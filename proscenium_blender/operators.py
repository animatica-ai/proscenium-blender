"""Blender operators for the Proscenium addon (MMCP client side).

Operators:
    proscenium.connect           — fetch /capabilities, populate model picker
    proscenium.generate          — build request, POST /generate, bake response
    proscenium.generate_pose     — single-frame variant (defined in step 8)
    proscenium.accept            — keep the generated motion, release source
    proscenium.reject            — restore the source action
    proscenium.cancel            — request cancellation of an in-flight gen

The generate operator is modal: a worker thread does the blocking POST while
a 0.1 s event-timer keeps the UI responsive. There is no SSE in MMCP v1, so
the modal shows an indeterminate progress bar (we can't know how far along
the server is).
"""

from __future__ import annotations

import threading

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy.types import Operator

from . import (
    constraints_ui,
    gltf_to_blender,
    mmcp_client,
    request_builder,
)


MOTION_ACTION_PREFIX = "Proscenium_Motion"
POSE_ACTION_NAME     = "Proscenium_Pose"

# Kept for back-compat with action references in older scenes — older bakes
# wrote to ``Proscenium_Generated``; the prefix filter (``Proscenium_``) in
# request_builder catches both old and new naming.
GENERATED_ACTION_NAME = MOTION_ACTION_PREFIX


_NLA_TRACK_PREFIX = "Proscenium: "


def _action_name_for_prompt(prompt: str) -> str:
    """Build an action name from a single block's prompt, clamped to fit
    Blender's 63-char action-name limit (with room left for ``.001`` auto
    suffixing).
    """
    label = (prompt or "").strip()
    name = f"{MOTION_ACTION_PREFIX}: {label}" if label else MOTION_ACTION_PREFIX
    MAX = 56
    if len(name) > MAX:
        name = name[:MAX - 1] + "…"
    return name


def _build_motion_action_name(prompt_blocks) -> str:
    """Descriptive Blender action name for a single-action motion bake.

    Picks the first enabled prompt block's text as the descriptive suffix.
    Used for the single-block / single-action fallback path; multi-block
    bakes call ``_action_name_for_prompt`` per block instead.
    """
    for b in prompt_blocks or ():
        if not getattr(b, "enabled", True):
            continue
        text = (b.prompt or "").strip()
        if text:
            return _action_name_for_prompt(text)
    return MOTION_ACTION_PREFIX


def _block_ranges_for_split(prompt_blocks, gen_start: int, gen_end: int):
    """Compute per-block ``(frame_start, frame_end, action_name)`` triples
    with half-gap expansion so NLA strips abut perfectly.

    Each enabled block claims half of the gap on each side (the rest going
    to its neighbor). The first block expands left to ``gen_start``, the
    last expands right to ``gen_end``, so no model output is discarded and
    no scene frame is uncovered.

    Returns ``[]`` if fewer than 2 enabled blocks are present (caller falls
    back to single-action bake in that case).
    """
    enabled = sorted(
        (b for b in prompt_blocks or () if getattr(b, "enabled", True)),
        key=lambda b: int(b.frame_start),
    )
    if len(enabled) < 2:
        return []

    ranges: list[tuple[int, int, str]] = []
    for i, b in enumerate(enabled):
        fs = int(b.frame_start)
        fe = int(b.frame_end)
        if i == 0:
            fs = min(fs, gen_start)
        else:
            prev_fe = int(enabled[i - 1].frame_end)
            # Midpoint between this block's start and previous block's end —
            # the +1 keeps strips non-overlapping when a gap has odd length.
            fs = (prev_fe + fs) // 2 + 1
        if i == len(enabled) - 1:
            fe = max(fe, gen_end)
        else:
            next_fs = int(enabled[i + 1].frame_start)
            fe = (fe + next_fs) // 2

        ranges.append((fs, fe, _action_name_for_prompt(b.prompt)))
    return ranges


def _push_actions_to_nla(armature_obj, actions) -> None:
    """Place every per-block action on a SINGLE shared NLA track named
    ``Proscenium: Motion``, in start-frame order.

    Strips share one track (instead of one track per strip) so playback is
    sequential: one block ends, the next plays. Putting each strip on its
    own track would stack them layered and play them simultaneously, which
    is the wrong default for a "play the timeline back end-to-end" workflow.
    Half-gap expansion in ``_block_ranges_for_split`` guarantees the strips
    don't overlap on the shared track.

    Wipes any prior ``Proscenium: ``-prefixed tracks first so a regenerate
    doesn't pile up duplicates.
    """
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    nla = armature_obj.animation_data.nla_tracks

    for track in list(nla):
        if track.name.startswith(_NLA_TRACK_PREFIX):
            nla.remove(track)

    if not actions:
        return

    track = nla.new()
    track.name = f"{_NLA_TRACK_PREFIX}Motion"

    # Sort by action.frame_range[0] so strips are added in timeline order;
    # NLA refuses out-of-order or overlapping strip insertions on the same
    # track, and an in-order pass is the safest contract.
    def _sort_key(a):
        try:
            return float(a.frame_range[0])
        except Exception:
            return 0.0

    for action in sorted(actions, key=_sort_key):
        try:
            start = int(action.frame_range[0])
        except Exception:
            start = 1
        strip = track.strips.new(name=action.name, start=start, action=action)
        # Blender 5.x ``strips.new`` returns a strip with ``influence=0`` by
        # default in some configurations — that's "the strip exists but
        # contributes nothing to the pose", which silently kills playback.
        # Force full influence so the strip drives the pose at 100%.
        strip.influence = 1.0
        # Disable blend-in/out ramps too — half-gap expansion already
        # places strips so they abut, no soft fade needed.
        strip.blend_in = 0.0
        strip.blend_out = 0.0


def _clear_proscenium_nla_tracks(armature_obj) -> None:
    """Remove every NLA track the addon owns. Called from Reject."""
    if armature_obj is None or armature_obj.animation_data is None:
        return
    nla = armature_obj.animation_data.nla_tracks
    for track in list(nla):
        if track.name.startswith(_NLA_TRACK_PREFIX):
            nla.remove(track)


def _root_location_data_path(armature_obj) -> str | None:
    """The fcurve data_path that drives the armature's root-bone location."""
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        return None
    root_bone = next(
        (pb for pb in armature_obj.pose.bones if pb.parent is None),
        None,
    )
    if root_bone is None:
        return None
    return f'pose.bones["{root_bone.name}"].location'


def _proscenium_motion_actions() -> list:
    """Every motion-bake action the addon owns (current and legacy names)."""
    return [
        a for a in bpy.data.actions
        if a.name.startswith(request_builder._GENERATED_ACTION_PREFIXES)
    ]


_INPLACE_CONSTRAINT_NAME = "Proscenium_InPlace"


def _apply_inplace_constraint(armature_obj, enabled: bool) -> None:
    """Add or remove a Limit Location constraint on the armature's root bone
    that pins bone-local X / Z to 0.

    Non-destructive — fcurves remain untouched, so flipping the toggle off
    restores the original travel without re-generating. Y is left
    unconstrained so vertical motion (jumps, crouches) still plays.

    Used for live preview-time toggling. At Accept, the constraint is
    "baked" by zeroing the X/Z fcurve values on the per-block actions and
    then removing the constraint, so the final motion data is genuinely
    travel-free without relying on a constraint persisting on the rig.
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        return
    root_bone = next(
        (pb for pb in armature_obj.pose.bones if pb.parent is None),
        None,
    )
    if root_bone is None:
        return

    existing = root_bone.constraints.get(_INPLACE_CONSTRAINT_NAME)
    if enabled:
        con = existing or root_bone.constraints.new('LIMIT_LOCATION')
        con.name = _INPLACE_CONSTRAINT_NAME
        # Pin X = 0 (bone-local).
        con.use_min_x = True
        con.use_max_x = True
        con.min_x = 0.0
        con.max_x = 0.0
        # Pin Z = 0.
        con.use_min_z = True
        con.use_max_z = True
        con.min_z = 0.0
        con.max_z = 0.0
        # Y unconstrained — vertical motion stays.
        con.use_min_y = False
        con.use_max_y = False
        con.owner_space = 'LOCAL'
        con.influence = 1.0
        con.mute = False
    else:
        if existing is not None:
            root_bone.constraints.remove(existing)


def _zero_root_xz_keyframes(action, armature_obj) -> int:
    """Set every root-bone X / Z location keyframe in ``action`` to 0.

    Used at Accept time when the In-place toggle is on, so the final per-
    block actions don't carry the original travel as inert keyframe data.
    Returns the number of fcurves modified.
    """
    target_path = _root_location_data_path(armature_obj)
    if target_path is None:
        return 0
    n = 0
    for fc in constraints_ui.iter_action_fcurves(action):
        if fc.data_path == target_path and fc.array_index in (0, 2):
            for kp in fc.keyframe_points:
                kp.co[1] = 0.0
                # Flatten handles too, otherwise easing curves might wobble
                # around the new 0.
                kp.handle_left[1] = 0.0
                kp.handle_right[1] = 0.0
            fc.update()
            n += 1
    return n


def _split_action_into_blocks(
    source_action,
    armature_obj,
    blocks,
):
    """Slice ``source_action``'s keyframes into N per-block actions.

    For each ``(frame_start, frame_end, action_name)`` in ``blocks``, builds
    a fresh action via the layered-Action API (mirrors the structure of the
    multi-block bake helper) and copies in only the source's keyframes that
    fall within ``[frame_start, frame_end]``. Keyframe ``type`` is preserved
    so previously-tagged ``KEYFRAME`` anchors keep their dopesheet styling
    after the split.

    Layered-API construction (instead of ``source_action.copy()`` + trim) is
    deliberate: it sidesteps the NLA-state corruption Blender 5.x exhibits
    when an action is touched-then-detached during keyframe writes. Fresh
    actions always evaluate cleanly on NLA strips.

    Returns the list of new actions in input order.
    """
    new_actions = []

    for fs, fe, action_name in blocks:
        new_a = bpy.data.actions.new(name=action_name)
        layer = new_a.layers.new(name="Layer")
        strip = layer.strips.new(type='KEYFRAME')
        slot = new_a.slots.new(id_type='OBJECT', name=armature_obj.name)
        cb = strip.channelbag(slot, ensure=True)

        for src_fc in constraints_ui.iter_action_fcurves(source_action):
            in_range = [
                (kp.co[0], kp.co[1], kp.type)
                for kp in src_fc.keyframe_points
                if fs <= kp.co[0] <= fe
            ]
            if not in_range:
                continue

            new_fc = cb.fcurves.new(
                data_path=src_fc.data_path,
                index=src_fc.array_index,
            )
            # Carry the mute flag — without this the In-place toggle's live
            # effect is lost the moment the user clicks Push to NLA (the
            # split would create fresh fcurves with mute=False, defeating
            # the user's intent).
            new_fc.mute = src_fc.mute
            n = len(in_range)
            new_fc.keyframe_points.add(n)
            flat = [0.0] * (2 * n)
            for i, (f, v, _t) in enumerate(in_range):
                flat[2 * i]     = float(f)
                flat[2 * i + 1] = float(v)
            new_fc.keyframe_points.foreach_set("co", flat)
            for kp_obj, (_, _, kp_type) in zip(new_fc.keyframe_points[:n], in_range):
                kp_obj.type = kp_type
            new_fc.update()

        new_a.use_fake_user = True
        new_actions.append(new_a)

    return new_actions


def _stash_quota_state(settings, exc) -> None:
    """If ``exc`` is a quota-exceeded MmcpError, mirror its message and
    upgrade URL onto the scene settings so the panel can render a
    persistent banner with an "Upgrade" action. No-op for other errors."""
    if not isinstance(exc, mmcp_client.MmcpError):
        return
    if exc.code != "quota_exceeded":
        return
    settings.quota_exceeded_message = exc.message or str(exc)
    settings.quota_upgrade_url = (exc.details or {}).get("upgrade_url", "")


def _clear_quota_state(settings) -> None:
    settings.quota_exceeded_message = ""
    settings.quota_upgrade_url = ""


# ═══════════════════════════════════════════════════════════════════════════
# Connect
# ═══════════════════════════════════════════════════════════════════════════

class PROSCENIUM_OT_connect(Operator):
    bl_idname = "proscenium.connect"
    bl_label = "Connect"
    bl_description = (
        "Fetch GET /capabilities from the configured MMCP server. "
        "Populates the model dropdown with what the server hosts"
    )

    def execute(self, context):
        url = mmcp_client.get_mmcp_url()
        try:
            client = mmcp_client.MmcpClient(url, timeout=30)
            caps = client.capabilities(refresh=True)
        except mmcp_client.MmcpError as exc:
            mmcp_client.clear_capabilities(error=str(exc))
            self.report({'ERROR'}, f"Cannot connect to {url}: {exc}")
            return {'CANCELLED'}
        except Exception as exc:                         # noqa: BLE001 — defensive
            mmcp_client.clear_capabilities(error=str(exc))
            self.report({'ERROR'}, f"Cannot connect to {url}: {exc}")
            return {'CANCELLED'}

        mmcp_client.store_capabilities(caps)
        models = [m.get("id") for m in caps.get("models", [])]

        settings = context.scene.proscenium
        if settings.model_id not in models and models:
            try:
                settings.model_id = models[0]
            except TypeError:
                pass

        for area in context.screen.areas:
            area.tag_redraw()

        self.report({'INFO'}, f"Connected. {len(models)} model(s): {', '.join(models) or '(none)'}")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# Generate
# ═══════════════════════════════════════════════════════════════════════════

class PROSCENIUM_OT_generate(Operator):
    bl_idname = "proscenium.generate"
    bl_label = "Generate Motion"
    bl_description = (
        "Build an MMCP request from the current scene (segments + constraints), "
        "POST it to /generate, and bake the returned glTF onto the target armature"
    )

    _timer = None
    _thread: threading.Thread | None = None
    _result: dict | None = None
    _error: Exception | None = None
    _anchor_frames: set[int] | None = None

    def execute(self, context):
        settings = context.scene.proscenium

        if settings.is_generating:
            self.report({'WARNING'}, "Already generating — wait or click Cancel")
            return {'CANCELLED'}

        arm = settings.target_armature
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set a target armature first")
            return {'CANCELLED'}

        model_caps = mmcp_client.cached_model(settings.model_id)
        if model_caps is None:
            self.report({'ERROR'}, "Connect to the server first (Connection panel → Connect)")
            return {'CANCELLED'}

        # Regenerate path: revert to the source action so the request reflects
        # the user's authored keyframes, not the generated preview.
        if settings.source_action_name:
            src = bpy.data.actions.get(settings.source_action_name)
            if src is not None:
                if arm.animation_data is None:
                    arm.animation_data_create()
                arm.animation_data.action = src

        try:
            req = request_builder.build_request(
                model_id=settings.model_id,
                model_caps=model_caps,
                armature_obj=arm,
                prompt_blocks=settings.prompt_blocks,
                settings=settings,
                scene=context.scene,
                constraint_objects=constraints_ui.walk_scene_constraints(context.scene),
            )
        except request_builder.BuildError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        # Save the source action for Accept / Reject (no-op if already saved).
        if not settings.source_action_name and arm.animation_data and arm.animation_data.action:
            settings.source_action_name = arm.animation_data.action.name

        # Remember which scene-frames the user actually keyed so we can tag
        # them ``'KEYFRAME'`` after the bake (while every other frame from
        # the generated timeline gets tagged ``'GENERATED'``). Three sources
        # contribute, all collapsed to scene-frame space:
        #   1. ``pose_keyframe`` constraints — frame index is timeline-
        #      relative (0 = request's first frame), so shift by gen_start.
        #   2. ``effector_target`` constraints — same timeline-relative
        #      indexing.
        #   3. Every keyframe on the source action's fcurves, regardless of
        #      channel. This catches location-only keys (e.g. root-bone
        #      path animation) and scale keys that the pose_keyframe
        #      sampler filters out (it only emits constraints from rotation
        #      fcurves), so a hand-authored Hips path stays visually
        #      distinguishable from the generated motion afterwards.
        gen_start, gen_end = request_builder.compute_frame_range(
            settings.prompt_blocks, arm, context.scene
        )
        self._gen_start_frame = gen_start

        anchor_frames: set[int] = set()
        for c in req.get("constraints", []):
            t = c.get("type")
            if t == "pose_keyframe":
                anchor_frames.add(int(c["frame"]) + gen_start)
            elif t == "effector_target":
                for f in c.get("frames", []) or ():
                    anchor_frames.add(int(f) + gen_start)
            # root_path frames come from evenly-spaced curve sampling, not
            # from user keyframes — skip them.

        src_action = (
            arm.animation_data.action
            if arm.animation_data and arm.animation_data.action
            else None
        )
        # Only motion-bake output is excluded — pose-generator output
        # (``Proscenium_Pose`` / legacy ``Proscenium_Poses``) is the user's
        # authored content (they chose to keep those poses as anchors), so
        # those keyframes should stay typed as ``KEYFRAME`` after the bake.
        if src_action is not None and not src_action.name.startswith(
            request_builder._GENERATED_ACTION_PREFIXES
        ):
            for fc in constraints_ui.iter_action_fcurves(src_action):
                for kp in fc.keyframe_points:
                    f = int(round(kp.co.x))
                    if gen_start <= f <= gen_end:
                        anchor_frames.add(f)

        self._anchor_frames = anchor_frames

        # Reset state and kick worker.
        settings.is_generating = True
        settings.cancel_requested = False
        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._worker,
            args=(mmcp_client.get_mmcp_url(), req),
            daemon=True,
        )
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    # ----- thread body -----------------------------------------------------
    def _worker(self, server_url: str, req: dict) -> None:
        try:
            client = mmcp_client.MmcpClient(server_url)
            self._result = client.generate(req)
        except Exception as exc:                         # noqa: BLE001 — surfaced to UI
            self._error = exc

    # ----- modal -----------------------------------------------------------
    def modal(self, context, event):
        settings = context.scene.proscenium

        if event.type == 'ESC' or settings.cancel_requested:
            self._cleanup(context)
            self.report({'INFO'}, "Generation cancelled (request still runs server-side)")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._thread is not None and self._thread.is_alive():
            return {'RUNNING_MODAL'}

        # Worker finished.
        if self._error is not None:
            self._cleanup(context)
            _stash_quota_state(context.scene.proscenium, self._error)
            self.report({'ERROR'}, f"Generation failed: {self._error}")
            return {'CANCELLED'}

        if self._result is None:
            self._cleanup(context)
            self.report({'ERROR'}, "Worker exited with no result")
            return {'CANCELLED'}

        # Successful run — clear any stale quota banner.
        _clear_quota_state(context.scene.proscenium)

        # Bake. Two paths:
        #   • 2+ enabled prompt blocks → split the response into one action
        #     per block, push to NLA. Lets the user iterate on individual
        #     blocks later (and matches Blender's "strip per beat" mental
        #     model). Control-rig handling is intentionally bypassed here;
        #     fall back to the single-action path for control rigs.
        #   • Otherwise → existing single-action bake (handles control rigs
        #     via the Mixamo operator hand-off).
        gen_start = getattr(self, "_gen_start_frame", context.scene.frame_start)
        gen_end_settings_scene_frame = context.scene.frame_end
        # Recompute gen_end via the same helper to stay in sync with what
        # was sent to the server.
        _, gen_end = request_builder.compute_frame_range(
            settings.prompt_blocks, settings.target_armature, context.scene
        )

        block_ranges = (
            _block_ranges_for_split(settings.prompt_blocks, gen_start, gen_end)
            if not request_builder.is_control_rig(settings.target_armature)
            else []
        )

        n_actions = 0
        skipped: list[str] = []
        try:
            # Always bake as a single action for preview — it scrubs cleanly
            # via the active-action / dopesheet display, no NLA stack
            # required. Multi-block scenes still get split into per-block
            # actions on Accept; we just stash the block ranges on the
            # armature so the Accept handler knows what to do.
            preview_name = (
                f"{MOTION_ACTION_PREFIX}: Preview"
                if block_ranges
                else _build_motion_action_name(settings.prompt_blocks)
            )
            action = gltf_to_blender.bake_gltf_to_armature(
                self._result,
                settings.target_armature,
                sample_index=0,
                action_name=preview_name,
                start_frame=gen_start,
                anchor_frames=getattr(self, "_anchor_frames", None),
            )
            n_actions = 1
            skipped = list(action.get("proscenium_skipped_joints") or [])

            # In-place mode: drop a Limit Location constraint on the root
            # so the character is pinned at bone-local xz=0 while still
            # playing vertical motion. Toggling the property afterwards
            # adds/removes the constraint live via its update callback;
            # this branch is just for the "toggle was already on when the
            # bake completed" case.
            if getattr(settings, "inplace", False):
                _apply_inplace_constraint(settings.target_armature, enabled=True)
            else:
                # Defensive: if a stale constraint lingers from a prior
                # session, clear it.
                _apply_inplace_constraint(settings.target_armature, enabled=False)

            arm = settings.target_armature
            if block_ranges:
                # Stash split metadata for Accept. Blender's ID-property
                # arrays are homogeneous numerics-only ("only floats, ints,
                # booleans and dicts are allowed in ID property arrays"),
                # which rules out a list-of-(int, int, str) shape. JSON-
                # encode into a string custom prop instead — survives save/
                # reload, decodes cheaply on Accept.
                import json as _json
                arm["proscenium_pending_block_ranges"] = _json.dumps([
                    [int(fs), int(fe), str(name)] for fs, fe, name in block_ranges
                ])
            else:
                # Single-block / control-rig path: drop any stale stash
                # from a prior multi-block preview that the user is
                # overwriting in place.
                if "proscenium_pending_block_ranges" in arm:
                    del arm["proscenium_pending_block_ranges"]
        except Exception as exc:                         # noqa: BLE001 — surfaced to UI
            self._cleanup(context)
            self.report({'ERROR'}, f"Bake failed: {exc}")
            return {'CANCELLED'}

        # Bake succeeded — keep the source-action reference so the
        # Accept / Reject preview UI knows what to fall back to.
        self._cleanup(context, preview=True)
        msg_suffix = f" ({n_actions} actions)" if n_actions > 1 else ""
        if skipped:
            self.report(
                {'WARNING'},
                f"Done — skipped {len(skipped)} unmatched joint(s){msg_suffix}",
            )
        else:
            self.report({'INFO'}, f"Generation complete{msg_suffix}")
        return {'FINISHED'}

    def _cleanup(self, context, *, preview: bool = False) -> None:
        # ``preview=True`` is set only by the success path so the Accept /
        # Reject UI can show. Every CANCELLED path leaves ``preview`` at
        # its default ``False`` and we drop the source reference here —
        # without this, a failed worker (e.g. a quota-exceeded 429) would
        # surface the preview UI even though no motion was baked.
        s = context.scene.proscenium
        if not preview:
            s.source_action_name = ""
            s.is_previewing = False
        else:
            # Source-action name may be empty for free-form generations
            # (no prior action to fall back to); ``is_previewing`` tracks
            # the preview UI independently so the panel still surfaces.
            s.is_previewing = True
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        s.is_generating = False
        s.cancel_requested = False


# ═══════════════════════════════════════════════════════════════════════════
# Cancel
# ═══════════════════════════════════════════════════════════════════════════

class PROSCENIUM_OT_cancel_generation(Operator):
    bl_idname = "proscenium.cancel"
    bl_label = "Cancel"
    bl_description = (
        "Stop waiting on the in-flight generation. The HTTP request continues "
        "server-side; the addon discards whatever comes back"
    )

    def execute(self, context):
        s = context.scene.proscenium
        if not s.is_generating:
            return {'CANCELLED'}
        s.cancel_requested = True
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# Preview: Accept / Reject
# ═══════════════════════════════════════════════════════════════════════════

class PROSCENIUM_OT_accept(Operator):
    bl_idname = "proscenium.accept"
    bl_label = "Push to NLA"
    bl_description = (
        "Commit the generated motion to the NLA stack. The preview action "
        "(single-block) or its per-block split (multi-block) is placed on a "
        "single shared NLA track named 'Proscenium: Motion'. The active "
        "action is cleared so NLA drives playback, and the source-action "
        "reference is released"
    )

    def execute(self, context):
        s = context.scene.proscenium
        arm = s.target_armature
        n_pushed = 0

        if arm is not None and arm.type == 'ARMATURE':
            import json as _json
            pending_raw = arm.get("proscenium_pending_block_ranges")
            try:
                pending = _json.loads(pending_raw) if pending_raw else []
            except (TypeError, ValueError):
                pending = []
            block_ranges = [
                (int(r[0]), int(r[1]), str(r[2]))
                for r in pending
                if len(r) >= 3
            ]

            preview_action = (
                arm.animation_data.action
                if arm.animation_data and arm.animation_data.action
                else None
            )

            actions_to_push: list = []

            if len(block_ranges) >= 2 and preview_action is not None:
                # Multi-block: build the per-block actions from the preview's
                # fcurves, drop the preview, push the splits.
                actions_to_push = _split_action_into_blocks(
                    preview_action, arm, block_ranges,
                )
                arm.animation_data.action = None
                bpy.data.actions.remove(preview_action)
            elif (
                preview_action is not None
                and preview_action.name.startswith(
                    request_builder._GENERATED_ACTION_PREFIXES
                )
            ):
                # Single-block: push the preview as-is (it's already the
                # final, prompt-named motion action). Detach from the
                # armature so NLA evaluation isn't shadowed by an active
                # action of the same content.
                preview_action.use_fake_user = True
                actions_to_push = [preview_action]
                arm.animation_data.action = None

            if actions_to_push:
                # If In place was on during preview, bake it: zero the X /
                # Z keyframes on every per-block action's root-bone
                # location fcurves, then remove the constraint. End state
                # is travel-free fcurve data on disk — survives across
                # regenerations, exports, and constraint stack edits.
                if bool(getattr(s, "inplace", False)):
                    for a in actions_to_push:
                        _zero_root_xz_keyframes(a, arm)

                _push_actions_to_nla(arm, actions_to_push)
                n_pushed = len(actions_to_push)

            # Always pull the constraint after Accept — its job is done
            # (either we baked the in-place state or the toggle was off).
            _apply_inplace_constraint(arm, enabled=False)

            if "proscenium_pending_block_ranges" in arm:
                del arm["proscenium_pending_block_ranges"]

        s.source_action_name = ""
        s.is_previewing = False
        if n_pushed:
            label = "strip" if n_pushed == 1 else "strips"
            self.report(
                {'INFO'},
                f"Pushed {n_pushed} {label} to NLA track 'Proscenium: Motion'",
            )
        else:
            self.report({'INFO'}, "Generated motion pushed to NLA")
        return {'FINISHED'}


class PROSCENIUM_OT_reject(Operator):
    bl_idname = "proscenium.reject"
    bl_label = "Reject"
    bl_description = (
        "Discard the generated motion and restore the source action so you "
        "can edit and regenerate"
    )

    def execute(self, context):
        s   = context.scene.proscenium
        arm = s.target_armature
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Target armature is gone")
            return {'CANCELLED'}

        # ``source`` is None for free-form generations (no prior action to
        # restore to). That's fine — we still need to clean up the preview
        # state, just without restoring anything.
        source = bpy.data.actions.get(s.source_action_name) if s.source_action_name else None

        if arm.animation_data is None:
            arm.animation_data_create()

        # Defensive: if the user manually assembled the per-block actions
        # onto NLA, strip our tracks before restoring the source action so
        # they don't keep playing on top.
        _clear_proscenium_nla_tracks(arm)

        # Drop any in-place constraint left over from preview state.
        _apply_inplace_constraint(arm, enabled=False)

        # Drop the pending-block-ranges stash — the preview action it
        # described is about to be orphaned and removed below.
        if "proscenium_pending_block_ranges" in arm:
            del arm["proscenium_pending_block_ranges"]

        # Restore the source action when we have one; otherwise just detach
        # the preview so the rig goes back to its un-animated state.
        arm.animation_data.action = source

        # Drop motion-bake orphans. Multi-block bakes mark each per-block
        # action with ``use_fake_user`` so Blender doesn't purge them while
        # the user is iterating; we have to look past that fake user when
        # deciding what's truly orphaned. Pose-generator actions
        # (``Proscenium_Pose`` / legacy ``Proscenium_Poses``) are user-
        # authored anchors and stay regardless of whether they're assigned.
        def _is_orphan(a):
            real_users = a.users - (1 if a.use_fake_user else 0)
            return real_users <= 0

        for ac in [a for a in bpy.data.actions
                   if a.name.startswith(request_builder._GENERATED_ACTION_PREFIXES)
                   and _is_orphan(a)]:
            bpy.data.actions.remove(ac)

        s.source_action_name = ""
        s.is_previewing = False
        if source is not None:
            self.report({'INFO'}, f"Restored source action {source.name!r}")
        else:
            self.report({'INFO'}, "Discarded preview")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# Pose generator (single keyframe at current frame, additive)
# ═══════════════════════════════════════════════════════════════════════════

class PROSCENIUM_OT_generate_pose(Operator):
    bl_idname = "proscenium.generate_pose"
    bl_label = "Generate Pose at Current Frame"
    bl_description = (
        "Generate a single pose from text and insert it as a keyframe at the "
        "current scene frame. Requires a server that advertises the 'pose' "
        "segment type (Animatica Cloud). Non-destructive — undo with Ctrl+Z"
    )

    prompt: StringProperty(
        name="Prompt",
        description="Text describing the pose to generate",
        default="a person stands in a neutral pose",
    )
    seed: IntProperty(name="Seed", default=42, min=0, max=999999)
    preserve_height: BoolProperty(
        name="Preserve height",
        description=(
            "Keep the root's current world height. When unchecked (default), "
            "the root's world Z is overridden by the generated pose's height "
            "(XY stays put), so a 'crouching' pose actually drops the character "
            "toward the floor and a 'jumping' pose lifts them"
        ),
        default=False,
    )

    # NOTE: keep these as plain class attributes (no type hints). Blender's
    # operator-registration walks ``__annotations__`` to resolve the class's
    # ``StringProperty``/``IntProperty``/``BoolProperty`` declarations into
    # RNA properties; under ``from __future__ import annotations`` every
    # annotation is a string and Blender's resolver chokes on the non-Property
    # ones (e.g. ``threading.Thread | None``), dropping ALL annotations for
    # the class — which makes the whole dialog go blank.
    _timer = None
    _thread = None
    _result = None
    _error = None
    _target_frame = 1

    @classmethod
    def poll(cls, context):
        # Hide the operator entirely when the connected model doesn't claim
        # 'pose' segment support — text-to-pose is a cloud-only capability.
        s = context.scene.proscenium
        model_caps = mmcp_client.cached_model(s.model_id) if s.model_id else None
        if model_caps is None:
            return False
        return "pose" in (model_caps.get("supported_segments") or [])

    # ----- UI --------------------------------------------------------------
    def invoke(self, context, event):
        s = context.scene.proscenium
        # Pre-fill from the most recent pose prompt the user submitted in
        # this scene; falls back to the property's default on first use.
        last = getattr(s, "last_pose_prompt", "")
        if last:
            self.prompt = last
        self.seed = int(s.seed)
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "prompt")
        layout.prop(self, "seed")
        layout.prop(self, "preserve_height")
        layout.label(text=f"Insert keyframe at frame {context.scene.frame_current}")

    # ----- entry -----------------------------------------------------------
    def execute(self, context):
        s = context.scene.proscenium
        if s.is_generating:
            self.report({'WARNING'}, "Already generating — wait or click Cancel")
            return {'CANCELLED'}

        arm = s.target_armature
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Set a target armature first")
            return {'CANCELLED'}

        model_caps = mmcp_client.cached_model(s.model_id)
        if model_caps is None:
            self.report({'ERROR'}, "Connect to the server first")
            return {'CANCELLED'}

        if "pose" not in (model_caps.get("supported_segments") or []):
            self.report({'ERROR'},
                        "Server does not advertise the 'pose' segment type. "
                        "Pose generation is an Animatica Cloud feature.")
            return {'CANCELLED'}

        # Persist the prompt to the scene so the next dialog open pre-fills
        # with what the user just submitted — kept here (after capability
        # checks pass) so a typo + cancel doesn't overwrite the previous
        # known-good prompt.
        s.last_pose_prompt = self.prompt

        # Send the user's own armature skeleton — the server retargets it to
        # the canonical on the way in and back again on the way out.
        request_skeleton = request_builder.armature_to_skeleton(arm)

        if not model_caps.get("supports_retargeting", True):
            canonical_joints = {j["name"] for j in model_caps["canonical_skeleton"]["joints"]}
            missing = canonical_joints - {pb.name for pb in arm.pose.bones}
            if missing:
                self.report({'ERROR'},
                            f"Server does not support retargeting and the armature is "
                            f"missing {len(missing)} canonical joint(s). Re-import via "
                            f"'Import canonical skeleton'")
                return {'CANCELLED'}
            request_skeleton = model_caps["canonical_skeleton"]

        # Single PoseSegment — server's specialized text-to-pose model returns
        # a 1-frame glTF directly. No client-side middle-frame extraction.
        req = {
            "protocol_version": request_builder.PROTOCOL_VERSION,
            "model":            s.model_id,
            "skeleton":         request_skeleton,
            "segments": [{
                "type":   "pose",
                "prompt": self.prompt,
            }],
            "options": {
                "diffusion_steps": request_builder.QUALITY_PRESETS.get(
                    s.quality_preset, int(s.custom_steps)
                ),
                "num_samples":     1,
                "seed":            int(self.seed) if int(self.seed) > 0 else None,
                "post_processing": bool(s.post_processing),
            },
        }

        self._target_frame = int(context.scene.frame_current)
        s.is_generating = True
        s.cancel_requested = False
        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._worker,
            args=(mmcp_client.get_mmcp_url(), req),
            daemon=True,
        )
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    # ----- thread body -----------------------------------------------------
    def _worker(self, server_url: str, req: dict) -> None:
        try:
            client = mmcp_client.MmcpClient(server_url)
            self._result = client.generate(req)
        except Exception as exc:                         # noqa: BLE001
            self._error = exc

    # ----- modal -----------------------------------------------------------
    def modal(self, context, event):
        s = context.scene.proscenium

        if event.type == 'ESC' or s.cancel_requested:
            self._cleanup(context)
            self.report({'INFO'}, "Pose generation cancelled")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._thread is not None and self._thread.is_alive():
            return {'RUNNING_MODAL'}

        if self._error is not None:
            self._cleanup(context)
            _stash_quota_state(context.scene.proscenium, self._error)
            self.report({'ERROR'}, f"Pose generation failed: {self._error}")
            return {'CANCELLED'}

        if self._result is None:
            self._cleanup(context)
            self.report({'ERROR'}, "Worker exited with no result")
            return {'CANCELLED'}

        # Successful run — clear any stale quota banner.
        _clear_quota_state(context.scene.proscenium)

        n_frames = gltf_to_blender.sample_frame_count(self._result, sample_index=0)
        if n_frames < 1:
            self._cleanup(context)
            self.report({'ERROR'}, "Response had no frames")
            return {'CANCELLED'}

        # PoseSegment yields exactly 1 frame from the server — no
        # middle-frame extraction needed; bake source_frame=0.
        try:
            written = gltf_to_blender.bake_single_pose(
                self._result,
                s.target_armature,
                source_frame=0,
                target_frame=self._target_frame,
                sample_index=0,
                root_translation="skip" if self.preserve_height else "height_only",
            )
        except Exception as exc:                         # noqa: BLE001
            self._cleanup(context)
            self.report({'ERROR'}, f"Bake failed: {exc}")
            return {'CANCELLED'}

        # Snap viewport to the freshly-keyframed pose.
        context.scene.frame_set(self._target_frame)

        self._cleanup(context)
        self.report({'INFO'}, f"Inserted pose: {written} channels @ frame {self._target_frame}")
        return {'FINISHED'}

    def _cleanup(self, context) -> None:
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        s = context.scene.proscenium
        s.is_generating = False
        s.cancel_requested = False


# ═══════════════════════════════════════════════════════════════════════════
# Auth — Animatica Cloud sign-in / sign-out
# ═══════════════════════════════════════════════════════════════════════════
#
# Auth is NOT part of the MMCP protocol. The cloud's proxy in front of
# /generate consumes the Bearer token; self-hosted servers ignore it.
# These operators only matter when the user is pointing at the cloud.

class PROSCENIUM_OT_signin(Operator):
    bl_idname = "proscenium.signin"
    bl_label = "Sign in to Animatica"
    bl_description = (
        "Exchange email + password for an Animatica session token. Only "
        "needed when pointing at Animatica Cloud — self-hosted servers "
        "don't require sign-in"
    )

    email: StringProperty(name="Email", default="")
    password: StringProperty(name="Password", default="", subtype='PASSWORD')

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=340)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "email")
        col.prop(self, "password")

    def execute(self, context):
        if not self.email or not self.password:
            self.report({'ERROR'}, "Email and password required")
            return {'CANCELLED'}
        try:
            data = mmcp_client.sign_in(self.email, self.password)
        except Exception as exc:                          # noqa: BLE001
            self.report({'ERROR'}, f"Sign-in failed: {exc}")
            return {'CANCELLED'}
        tier = data.get("tier", "")
        msg = f"Signed in as {data.get('email', self.email)}"
        if tier:
            msg += f" ({tier})"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class PROSCENIUM_OT_signout(Operator):
    bl_idname = "proscenium.signout"
    bl_label = "Sign out"
    bl_description = "Forget the cached Animatica session tokens"

    def execute(self, context):
        mmcp_client.sign_out()
        self.report({'INFO'}, "Signed out")
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# Quota / upgrade
# ═══════════════════════════════════════════════════════════════════════════

class PROSCENIUM_OT_open_upgrade(Operator):
    bl_idname = "proscenium.open_upgrade"
    bl_label = "Upgrade"
    bl_description = "Open the upgrade URL in your browser"

    def execute(self, context):
        url = (context.scene.proscenium.quota_upgrade_url or "").strip()
        if not url:
            self.report({'ERROR'}, "No upgrade URL available")
            return {'CANCELLED'}
        bpy.ops.wm.url_open(url=url)
        return {'FINISHED'}


class PROSCENIUM_OT_dismiss_quota(Operator):
    bl_idname = "proscenium.dismiss_quota"
    bl_label = "Dismiss"
    bl_description = "Hide the quota-exceeded banner"

    def execute(self, context):
        _clear_quota_state(context.scene.proscenium)
        return {'FINISHED'}


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

_classes = (
    PROSCENIUM_OT_connect,
    PROSCENIUM_OT_generate,
    PROSCENIUM_OT_generate_pose,
    PROSCENIUM_OT_accept,
    PROSCENIUM_OT_reject,
    PROSCENIUM_OT_cancel_generation,
    PROSCENIUM_OT_signin,
    PROSCENIUM_OT_signout,
    PROSCENIUM_OT_open_upgrade,
    PROSCENIUM_OT_dismiss_quota,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
