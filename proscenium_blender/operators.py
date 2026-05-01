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


def _build_motion_action_name(prompt_blocks) -> str:
    """Descriptive Blender action name for a motion bake.

    Picks the first enabled prompt block's text as the descriptive suffix,
    truncated so the full name stays under Blender's 64-char action-name
    limit. Falls back to ``MOTION_ACTION_PREFIX`` alone when no prompt is
    available (constraint-only generations). Subsequent regenerations get
    Blender's automatic ``.001`` / ``.002`` suffixes for free.
    """
    label = ""
    for b in prompt_blocks or ():
        if not getattr(b, "enabled", True):
            continue
        text = (b.prompt or "").strip()
        if text:
            label = text
            break
    if not label:
        return MOTION_ACTION_PREFIX
    name = f"{MOTION_ACTION_PREFIX}: {label}"
    # Blender allows up to 63 chars; clamp with an ellipsis a few chars under
    # so the auto-suffixer has room for ``.001``-style tags.
    MAX = 56
    if len(name) > MAX:
        name = name[:MAX - 1] + "…"
    return name


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

        # Bake.
        try:
            action = gltf_to_blender.bake_gltf_to_armature(
                self._result,
                settings.target_armature,
                sample_index=0,
                action_name=_build_motion_action_name(settings.prompt_blocks),
                start_frame=getattr(
                    self, "_gen_start_frame", context.scene.frame_start
                ),
                anchor_frames=getattr(self, "_anchor_frames", None),
            )
        except Exception as exc:                         # noqa: BLE001 — surfaced to UI
            self._cleanup(context)
            self.report({'ERROR'}, f"Bake failed: {exc}")
            return {'CANCELLED'}

        # Bake succeeded — keep the source-action reference so the
        # Accept / Reject preview UI knows what to fall back to.
        self._cleanup(context, preview=True)
        skipped = action.get("proscenium_skipped_joints") or []
        if skipped:
            self.report({'WARNING'}, f"Done — skipped {len(skipped)} unmatched joint(s)")
        else:
            self.report({'INFO'}, "Generation complete")
        return {'FINISHED'}

    def _cleanup(self, context, *, preview: bool = False) -> None:
        # ``preview=True`` is set only by the success path so the
        # Accept / Reject UI can show. Every CANCELLED path leaves
        # ``preview`` at its default ``False`` and we drop the source
        # reference here — without this, a failed worker (e.g. a
        # quota-exceeded 429) would surface the preview UI even though
        # no motion was baked.
        if not preview:
            context.scene.proscenium.source_action_name = ""
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
    bl_label = "Accept"
    bl_description = (
        "Keep the generated motion. The generated Proscenium action stays "
        "as the armature's active action; the source action reference is released"
    )

    def execute(self, context):
        context.scene.proscenium.source_action_name = ""
        self.report({'INFO'}, "Generated motion accepted")
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

        source = bpy.data.actions.get(s.source_action_name) if s.source_action_name else None
        if source is None:
            self.report({'WARNING'}, f"Source action {s.source_action_name!r} not found")
            return {'CANCELLED'}

        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = source

        # Drop any orphaned motion-bake outputs so they don't pile up.
        # Match on the motion-only prefix tuple — pose-generator actions
        # (``Proscenium_Pose`` / legacy ``Proscenium_Poses``) are user-
        # authored anchors and stay even when not currently assigned.
        for ac in [a for a in bpy.data.actions
                   if a.name.startswith(request_builder._GENERATED_ACTION_PREFIXES)
                   and a.users == 0]:
            bpy.data.actions.remove(ac)

        s.source_action_name = ""
        self.report({'INFO'}, f"Restored source action {source.name!r}")
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
