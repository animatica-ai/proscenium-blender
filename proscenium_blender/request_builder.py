"""Assemble a complete MMCP `GenerateRequest` from current Blender state.

This is the integration point: it pulls together everything from the rest of
the addon (capabilities cache, prompt blocks, constraint objects, settings)
and produces the request dict the ``mmcp_client`` POSTs to ``/generate``.
"""

from __future__ import annotations

from typing import Any

import bpy
from mathutils import Vector

from . import constraints_ui, coords


PROTOCOL_VERSION = "1.0"

QUALITY_PRESETS = {
    "STANDARD": 50,
    "HALF":     25,
    "QUARTER":  12,
}


class BuildError(Exception):
    """Raised when the current state can't be turned into a valid request."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_request(
    *,
    model_id: str,
    model_caps: dict[str, Any],
    armature_obj: bpy.types.Object,
    prompt_blocks: list,
    settings,
    scene: bpy.types.Scene,
    constraint_objects: dict[str, list[bpy.types.Object]],
) -> dict[str, Any]:
    """Build the request dict. Raises ``BuildError`` if state is incomplete."""

    if armature_obj is None or armature_obj.type != 'ARMATURE':
        raise BuildError("Set a target armature first")

    canonical = model_caps.get("canonical_skeleton") or {}
    canonical_joint_names = {j["name"] for j in canonical.get("joints", [])}
    supports_retargeting = bool(model_caps.get("supports_retargeting", False))

    # Build the request's skeleton from the user's armature. When the user
    # has imported the canonical skeleton, this matches it 1:1 and the
    # server skips the retarget hop. When they've picked any other rig, the
    # server's retarget pipeline will map it to canonical.
    request_skeleton = armature_to_skeleton(armature_obj)
    armature_bones = {pb.name for pb in armature_obj.pose.bones}

    if not supports_retargeting:
        # Legacy path for servers that can't retarget.
        if not canonical_joint_names:
            raise BuildError(f"Model {model_id!r} has no canonical_skeleton.joints")
        missing = canonical_joint_names - armature_bones
        if missing:
            raise BuildError(
                f"Armature {armature_obj.name!r} is missing {len(missing)} canonical joint(s) "
                f"(first few: {sorted(missing)[:5]}). "
                f"This server does not support retargeting — pick a rig that "
                f"mirrors the canonical skeleton, or use 'Import canonical skeleton'"
            )
        request_skeleton = canonical                  # echo verbatim

    frame_range = (int(scene.frame_start), int(scene.frame_end))
    segments = build_segments(prompt_blocks, frame_range)

    # Total timeline length matches the scene range so frames in constraints
    # land where the user authored them.
    total_frames = (
        sum(s["duration_frames"] for s in segments)
        if segments
        else (frame_range[1] - frame_range[0] + 1)
    )
    if total_frames < 1:
        raise BuildError("Scene frame range is empty")

    constraints = _collect_constraints(
        armature_obj=armature_obj,
        constraint_objects=constraint_objects,
        frame_range=frame_range,
        total_frames=total_frames,
    )

    if not segments and not constraints:
        raise BuildError(
            "No prompts or constraints to generate from. "
            "Add a prompt block on the timeline, draw a root path, or pin an effector"
        )

    valid_joint_names = {j["name"] for j in request_skeleton.get("joints", [])}
    _validate_constraint_joints(constraints, valid_joint_names)
    _validate_constraint_count(constraints, model_caps)

    request: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "model":            model_id,
        "skeleton":         request_skeleton,
        "options":          build_options(settings),
    }

    if segments:
        request["segments"] = segments
    else:
        request["duration_frames"] = total_frames

    if constraints:
        request["constraints"] = constraints

    return request


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

# Bone-following constraint types: when a deform bone carries one of
# these targeting a sibling bone, that's the universal signal that this
# rig has a control layer driving the deform layer (Mixamo Control Rig,
# Rigify, Auto-Rig Pro, custom). Used only for "is this a control rig?"
# detection — the deform set itself comes from ``bone.use_deform``, the
# flag rigs use to mark which bones skin the mesh.
_BONE_FOLLOWING_CONSTRAINTS = frozenset({
    "COPY_TRANSFORMS",
    "COPY_ROTATION",
    "COPY_LOCATION",
    "IK",
})


def detect_deform_bones(armature_obj: bpy.types.Object) -> set[str]:
    """Return the set of bone names that skin the mesh — i.e. bones with
    ``Bone.use_deform`` set. Every standard rig system (Mixamo, Rigify
    DEF-*, Auto-Rig Pro, custom) flags its deform bones with this; the
    flag is what drives vertex skinning, so it's the canonical "this is
    the actual character skeleton" signal that's stable across naming
    conventions and constraint structures.
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        return set()
    return {pb.name for pb in armature_obj.pose.bones if pb.bone.use_deform}


def is_control_rig(armature_obj: bpy.types.Object) -> bool:
    """``True`` when the armature's deform layer is driven from a separate
    control layer via constraints — what people mean by "control rig".

    Plain rigs (just the deform bones, animated directly) return False:
    no constraints means no control layer, so request building runs
    against the bones the user authored without filtering.
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        return False
    deform_names = detect_deform_bones(armature_obj)
    if not deform_names:
        return False
    bone_names = {pb.name for pb in armature_obj.pose.bones}
    for pb in armature_obj.pose.bones:
        if pb.name not in deform_names:
            continue
        for c in pb.constraints:
            if getattr(c, "mute", False):
                continue
            if getattr(c, "influence", 1.0) <= 0.0:
                continue
            if c.type not in _BONE_FOLLOWING_CONSTRAINTS:
                continue
            if getattr(c, "target", None) is armature_obj and getattr(c, "subtarget", "") in bone_names:
                return True
    return False


def _closest_deform_ancestor(pb: bpy.types.PoseBone, deform: set[str]):
    """Walk up parents until one is in ``deform``. Returns the PoseBone or
    None for the topmost deform bone in the chain."""
    p = pb.parent
    while p is not None and p.name not in deform:
        p = p.parent
    return p


def armature_to_skeleton(armature_obj: bpy.types.Object) -> dict[str, Any]:
    """Serialize the armature's rest layout to the MMCP `Skeleton` shape.

    Positions are expressed in MMCP frame (Y-up, meters). Heads are converted
    through ``armature.matrix_world`` before differencing, so rigs with a
    non-identity world transform (Mixamo's 90° + 0.01 scale is the common
    case) end up in the same world frame as the per-frame ``root_position``.
    Without this, offsets would live in armature-local frame while root pins
    live in world frame, and the 90° / scale would silently misalign them.

    When the armature has a control-rig setup (deform bones driven by
    constraints), only the deform bones are emitted — the server's bone
    classifier would otherwise see Ctrl_*/IK helpers/pole vectors and pick
    wrong slots. Parent links of deform bones get rewritten to skip over
    any non-deform intermediaries, keeping the rest hierarchy intact.
    """
    mw = armature_obj.matrix_world
    pose_bones = list(armature_obj.pose.bones)
    head_world_by_name: dict[str, Vector] = {
        pb.name: mw @ pb.bone.head_local for pb in pose_bones
    }

    deform = detect_deform_bones(armature_obj)
    use_deform_filter = is_control_rig(armature_obj)

    joints: list[dict[str, Any]] = []
    for pb in pose_bones:
        if use_deform_filter and pb.name not in deform:
            continue
        if use_deform_filter:
            parent = _closest_deform_ancestor(pb, deform)
        else:
            parent = pb.parent
        parent_name = parent.name if parent else None
        if parent is None:
            local = head_world_by_name[pb.name]
        else:
            local = head_world_by_name[pb.name] - head_world_by_name[parent.name]
        mx, my, mz = coords.blender_pos_to_mmcp(local)
        joints.append({
            "name":             pb.name,
            "parent":           parent_name,
            "rest_translation": [float(mx), float(my), float(mz)],
            "rest_rotation":    [0.0, 0.0, 0.0, 1.0],
        })
    return {
        "joints":            joints,
        "coordinate_system": "right_handed_y_up",
        "units":             "meters",
    }


def build_segments(prompt_blocks, frame_range: tuple[int, int]) -> list[dict[str, Any]]:
    """Convert the addon's ``PromptBlock`` collection into MMCP segments.

    Strategy:
      * Span the whole scene range so timeline-relative constraint frames
        line up with the user's authored frame numbers.
      * Each enabled block becomes a TextSegment (or UnconditionedSegment
        if the prompt is empty/whitespace).
      * Gaps before / between / after enabled blocks are filled with
        UnconditionedSegment so the model picks a sensible interpolation.
    """
    enabled: list[tuple[int, int, str]] = []
    for b in prompt_blocks:
        if not getattr(b, "enabled", True):
            continue
        s = max(int(b.frame_start), frame_range[0])
        e = min(int(b.frame_end),   frame_range[1])
        if e < s:
            continue
        enabled.append((s, e, (b.prompt or "").strip()))

    if not enabled:
        return []

    enabled.sort(key=lambda t: t[0])

    # Resolve overlaps by bumping the next block past the previous block's end.
    cleaned: list[tuple[int, int, str]] = []
    prev_end = frame_range[0] - 1
    for s, e, p in enabled:
        s = max(s, prev_end + 1)
        if s > e:
            continue
        cleaned.append((s, e, p))
        prev_end = e
    if not cleaned:
        return []

    segments: list[dict[str, Any]] = []
    cursor = frame_range[0]
    for s, e, prompt in cleaned:
        if s > cursor:
            segments.append({"type": "unconditioned", "duration_frames": s - cursor})
        if prompt:
            segments.append({
                "type":            "text",
                "prompt":          prompt,
                "duration_frames": e - s + 1,
            })
        else:
            # Empty/whitespace prompt promotes to unconditioned (TextSegment
            # would fail server-side validation on min_length=1).
            segments.append({"type": "unconditioned", "duration_frames": e - s + 1})
        cursor = e + 1

    if cursor <= frame_range[1]:
        segments.append({"type": "unconditioned", "duration_frames": frame_range[1] - cursor + 1})
    return segments


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

_GENERATED_ACTION_PREFIX = "Proscenium_Generated"


def _collect_constraints(
    *,
    armature_obj: bpy.types.Object,
    constraint_objects: dict[str, list[bpy.types.Object]],
    frame_range: tuple[int, int],
    total_frames: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for curve in constraint_objects.get("root_paths", []):
        c = constraints_ui.sample_root_path(curve, total_frames=total_frames)
        if c is not None:
            out.append(c)

    for empty in constraint_objects.get("effector_targets", []):
        c = constraints_ui.sample_effector_target(
            empty, frame_range=frame_range, total_frames=total_frames,
        )
        if c is not None:
            out.append(c)

    # Sample pose keyframes ONLY from a user-authored action, not from a
    # previous generation's bake. Otherwise regenerate ends up feeding the
    # model's own output back as constraints — feedback loop, garbled motion.
    src = (
        armature_obj.animation_data.action
        if armature_obj.animation_data and armature_obj.animation_data.action
        else None
    )
    if src is not None and not src.name.startswith(_GENERATED_ACTION_PREFIX):
        out.extend(
            constraints_ui.sample_pose_keyframes(
                armature_obj,
                source_action=src,
                frame_range=frame_range,
            )
        )

    # Anchor the motion's start to wherever the user placed the character.
    # Without this, the generated motion begins at the model's default root
    # (origin) regardless of the armature's world transform — pose keyframes
    # later in the timeline pull the body toward their targets but frame 0
    # stays stuck at (0, 0), which is what the user saw as "the first
    # keyframe gets reset to (0, 0)". Skip if another constraint already
    # pins frame 0.
    anchor = _start_anchor(armature_obj, out)
    if anchor is not None:
        out.append(anchor)

    return out


def _start_anchor(
    armature_obj: bpy.types.Object,
    existing: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if any(_pins_frame_zero(c) for c in existing):
        return None
    root_pb = next(
        (pb for pb in armature_obj.pose.bones if pb.parent is None),
        None,
    )
    if root_pb is None:
        return None
    root_world = (armature_obj.matrix_world @ root_pb.matrix).translation
    x, _, z = coords.blender_pos_to_mmcp(root_world)
    return {
        "type":         "root_path",
        "frames":       [0],
        "positions_xz": [[x, z]],
    }


def _pins_frame_zero(c: dict[str, Any]) -> bool:
    t = c.get("type")
    if t == "root_path":
        return 0 in (c.get("frames") or [])
    if t == "pose_keyframe":
        return c.get("frame") == 0 and c.get("root_position") is not None
    return False


def _validate_constraint_joints(
    constraints: list[dict[str, Any]],
    canonical_joint_names: set[str],
) -> None:
    for i, c in enumerate(constraints):
        if c["type"] == "effector_target":
            joint = c.get("joint", "")
            if joint not in canonical_joint_names:
                raise BuildError(
                    f"Constraint #{i} (effector_target) targets unknown joint {joint!r}. "
                    f"Allowed joints come from /capabilities.models[].canonical_skeleton.joints[].name"
                )
        elif c["type"] == "pose_keyframe":
            unknown = sorted(set(c.get("joint_rotations", {})) - canonical_joint_names)
            if unknown:
                raise BuildError(
                    f"Constraint #{i} (pose_keyframe) references unknown joints: {unknown[:5]}"
                )


def _validate_constraint_count(constraints: list[dict[str, Any]], model_caps: dict[str, Any]) -> None:
    limits = model_caps.get("limits") or {}
    cap = int(limits.get("max_constraints_per_request") or 0)
    if cap and len(constraints) > cap:
        raise BuildError(
            f"{len(constraints)} constraints exceeds the model's max of {cap} "
            f"(disable some keyframes / drop an effector pin)"
        )


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def build_options(settings) -> dict[str, Any]:
    steps = QUALITY_PRESETS.get(settings.quality_preset, int(settings.custom_steps))

    opts: dict[str, Any] = {
        "diffusion_steps":   int(steps),
        "num_samples":       1,                          # multi-sample UI is future work
        "seed":              int(settings.seed) if int(settings.seed) > 0 else None,
        "post_processing":   bool(settings.post_processing),
        "transition_frames": int(settings.num_transition_frames),
    }

    if settings.cfg_enabled:
        opts["guidance"] = {
            "type":   "separated",
            "weight": [float(settings.cfg_text), float(settings.cfg_constraint)],
        }
    return opts
