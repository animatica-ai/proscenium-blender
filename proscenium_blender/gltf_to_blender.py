"""Bake an MMCP glTF response onto a Blender armature.

The MMCP server returns a self-contained glTF 2.0 JSON document with one
animation per generated sample. Each animation has rotation channels per
joint (target.path == "rotation") plus one translation channel on the root
(target.path == "translation"). Buffers are embedded as ``data:`` URIs.

This module decodes those channels and writes them as Blender F-curve
keyframes on the target armature's pose bones. Coord conversion goes through
``coords`` (MMCP Y-up → Blender Z-up).

Public API:
    * ``bake_gltf_to_armature(gltf, armature_obj, sample_index=0, action_name=...)
        -> bpy.types.Action``  — main entry point.
    * ``count_samples(gltf) -> int``
    * ``read_extension_metadata(gltf) -> dict``
"""

from __future__ import annotations

import base64
import struct
from typing import Any

import bpy
from mathutils import Matrix, Quaternion, Vector

from . import coords


# MMCP world (right-handed Y-up) → Blender world (right-handed Z-up) as a 3×3
# basis-change matrix. Mapping: MMCP +X → Blender +X, MMCP +Y → Blender +Z,
# MMCP +Z → Blender -Y.  Used to conjugate rotation matrices that arrive in
# the model's MMCP world frame.
_MMCP_TO_BLENDER = Matrix((
    (1.0, 0.0,  0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0,  0.0),
))


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

ROTATION_PATH    = "rotation"
TRANSLATION_PATH = "translation"


def count_samples(gltf: dict[str, Any]) -> int:
    return len(gltf.get("animations") or [])


def sample_frame_count(gltf: dict[str, Any], sample_index: int = 0) -> int:
    """Number of frames in a sample (read from the MMCP_motion extension)."""
    meta = read_extension_metadata(gltf)
    samples = meta.get("samples") or []
    if 0 <= sample_index < len(samples):
        return int(samples[sample_index].get("num_frames", 0))
    return 0


def read_extension_metadata(gltf: dict[str, Any]) -> dict[str, Any]:
    return (gltf.get("extensions") or {}).get("MMCP_motion") or {}


def bake_gltf_to_armature(
    gltf: dict[str, Any],
    armature_obj: bpy.types.Object,
    *,
    sample_index: int = 0,
    action_name: str = "Proscenium_Motion",
    start_frame: int = 1,
    anchor_frames: set[int] | None = None,
) -> bpy.types.Action:
    """Write one sample from ``gltf`` as a fresh Action on ``armature_obj``.

    The new Action becomes the armature's active action, replacing whatever
    was there. Returns the new Action so callers can hook Accept/Reject.

    Joints in the response that don't exist on the armature are silently
    skipped — the addon's "Import canonical skeleton" path guarantees a 1:1
    name match for the supported flow.

    ``anchor_frames`` (scene-space frame indices) flags keyframes the user
    originally authored — those get typed ``'KEYFRAME'`` (yellow diamond
    in the dopesheet); every other frame is typed ``'GENERATED'`` (grey),
    so the two kinds are visually distinct. ``None`` leaves the default
    keyframe type for everything.
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        raise ValueError("bake target must be an ARMATURE object")

    animations = gltf.get("animations") or []
    if not animations:
        raise ValueError("response has no animations[]")
    if sample_index < 0 or sample_index >= len(animations):
        raise ValueError(f"sample_index {sample_index} out of range (0..{len(animations) - 1})")

    nodes = gltf.get("nodes") or []
    anim  = animations[sample_index]
    samplers = anim.get("samplers") or []
    channels = anim.get("channels") or []

    # Pre-decode each unique sampler.output once. Multiple channels can share
    # an input timestamps accessor, so dedupe.
    decoded_outputs: dict[int, list[tuple]] = {}
    decoded_inputs:  dict[int, list[float]] = {}

    def _input_for(sampler_idx: int) -> list[float]:
        s = samplers[sampler_idx]
        in_idx = s["input"]
        if in_idx not in decoded_inputs:
            decoded_inputs[in_idx] = _read_floats(gltf, in_idx, "SCALAR")
        return decoded_inputs[in_idx]

    def _quats_for(sampler_idx: int) -> list[tuple[float, float, float, float]]:
        s = samplers[sampler_idx]
        out_idx = s["output"]
        if out_idx not in decoded_outputs:
            floats = _read_floats(gltf, out_idx, "VEC4")
            decoded_outputs[out_idx] = [
                (floats[i], floats[i + 1], floats[i + 2], floats[i + 3])
                for i in range(0, len(floats), 4)
            ]
        return decoded_outputs[out_idx]

    def _vec3s_for(sampler_idx: int) -> list[tuple[float, float, float]]:
        s = samplers[sampler_idx]
        out_idx = s["output"]
        if out_idx not in decoded_outputs:
            floats = _read_floats(gltf, out_idx, "VEC3")
            decoded_outputs[out_idx] = [
                (floats[i], floats[i + 1], floats[i + 2])
                for i in range(0, len(floats), 3)
            ]
        return decoded_outputs[out_idx]

    # Make sure pose-bone rotation modes are quaternion (we're feeding quats).
    pose = armature_obj.pose
    for pb in pose.bones:
        pb.rotation_mode = 'QUATERNION'

    # Fresh Action.
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    new_action = bpy.data.actions.new(name=action_name)
    armature_obj.animation_data.action = new_action

    # Armature's world-rotation basis. The outbound side
    # (``armature_to_skeleton`` / ``sample_pose_keyframes``) lifts offsets
    # and rotations into Blender world frame, so we undo that here when
    # baking. No-op when ``matrix_world`` is identity.
    mw_rot = armature_obj.matrix_world.to_quaternion().to_matrix()
    mw_rot_t = mw_rot.transposed()

    skipped: list[str] = []

    # --- Rotation channels ---
    for ch in channels:
        target = ch.get("target") or {}
        if target.get("path") != ROTATION_PATH:
            continue
        node_idx = target.get("node")
        if node_idx is None or node_idx >= len(nodes):
            continue
        joint_name = nodes[node_idx].get("name", "")
        bone = pose.bones.get(joint_name)
        if bone is None:
            skipped.append(joint_name)
            continue

        sampler_idx = ch["sampler"]
        timestamps  = _input_for(sampler_idx)
        quats       = _quats_for(sampler_idx)

        # Inverse of the outbound chain in ``sample_pose_keyframes``:
        #   MMCP Y-up → Blender world → armature-local → bone-local rest.
        ML     = bone.bone.matrix_local.to_3x3()
        ML_inv = ML.transposed()   # ML is orthogonal → inverse = transpose

        for ts, q_mmcp in zip(timestamps, quats):
            qx, qy, qz, qw = q_mmcp
            R_mmcp          = Quaternion((qw, qx, qy, qz)).to_matrix()
            R_blender_world = _MMCP_TO_BLENDER @ R_mmcp @ _MMCP_TO_BLENDER.transposed()
            R_blender_arm   = mw_rot_t @ R_blender_world @ mw_rot
            R_bone          = ML_inv @ R_blender_arm @ ML
            bone.rotation_quaternion = R_bone.to_quaternion()
            bone.keyframe_insert(
                data_path="rotation_quaternion",
                frame=_frame_from_time(ts, gltf, start_frame),
            )

    # --- Root translation channel(s) ---
    for ch in channels:
        target = ch.get("target") or {}
        if target.get("path") != TRANSLATION_PATH:
            continue
        node_idx = target.get("node")
        if node_idx is None or node_idx >= len(nodes):
            continue
        joint_name = nodes[node_idx].get("name", "")
        bone = pose.bones.get(joint_name)
        if bone is None:
            skipped.append(joint_name)
            continue

        sampler_idx = ch["sampler"]
        timestamps  = _input_for(sampler_idx)
        positions   = _vec3s_for(sampler_idx)

        # MMCP world → Blender world → armature-local → bone-local offset.
        rest_head     = bone.bone.head_local             # armature-local rest
        ML            = bone.bone.matrix_local.to_3x3()  # armature → bone-local
        arm_world_inv = armature_obj.matrix_world.inverted()

        for ts, p_mmcp in zip(timestamps, positions):
            world     = Vector(coords.mmcp_pos_to_blender(p_mmcp))
            arm_local = arm_world_inv @ world
            delta     = arm_local - rest_head
            bone.location = ML.transposed() @ delta
            bone.keyframe_insert(
                data_path="location",
                frame=_frame_from_time(ts, gltf, start_frame),
            )

    if skipped:
        # Caller can decide whether to surface this as a report.
        new_action["proscenium_skipped_joints"] = sorted(set(skipped))

    # If the rig is a Mixamo control rig, hand off to the Mixamo addon's
    # battle-tested "Apply Animation to Control Rig" operator — it knows
    # the rig's IK/FK switching, Foot_IK_target chains, pole vectors, and
    # foot-roll cursor in detail. Our deform-bone keyframes act as the
    # source animation; the operator builds a temporary clean source rig
    # internally and bakes onto the control rig.
    from . import request_builder
    if request_builder.is_control_rig(armature_obj):
        timestamps = next(iter(decoded_inputs.values())) if decoded_inputs else []
        frames = [_frame_from_time(t, gltf, start_frame) for t in timestamps]
        if frames:
            _bake_to_control_rig(
                armature_obj,
                source_action=new_action,
                frame_start=min(frames),
                frame_end=max(frames),
            )

    if anchor_frames is not None:
        _tag_keyframe_types(new_action, anchor_frames)

    return new_action


def bake_gltf_to_actions_per_block(
    gltf: dict[str, Any],
    armature_obj: bpy.types.Object,
    *,
    blocks: list[tuple[int, int, str]],
    request_start_frame: int,
    sample_index: int = 0,
    anchor_frames: set[int] | None = None,
) -> list[bpy.types.Action]:
    """Slice one full-timeline gltf response into N per-block actions.

    The single request is unchanged — the model still sees every prompt
    segment at once, so its internal transition handling is preserved. We
    just route the resulting fcurve keyframes into N separate actions, one
    per ``blocks`` entry (typically one per enabled prompt block, with each
    block claiming half the gap on either side so strips abut on the NLA).

    ``blocks`` is ``[(frame_start, frame_end, action_name), ...]`` in the
    armature's scene frame space. ``request_start_frame`` is the scene
    frame the request's gltf time=0 maps to (i.e. the global ``gen_start``
    from ``compute_frame_range``); needed to convert gltf timestamps back
    to scene frames consistently across blocks.

    Returns the new actions in input order. The armature's active action is
    cleared so the caller can NLA-push the strips and let the timeline
    drive playback. Joints that don't exist on the rig are silently
    skipped (recorded on the first action's ``proscenium_skipped_joints``
    custom prop for the operator to surface).

    Control-rig handling is intentionally NOT applied here — this slice
    path targets straight skeletons. Callers should detect a control rig
    and fall back to ``bake_gltf_to_armature`` (single action + Mixamo
    operator hand-off).
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        raise ValueError("bake target must be an ARMATURE object")
    if not blocks:
        return []

    animations = gltf.get("animations") or []
    if not animations:
        raise ValueError("response has no animations[]")
    if sample_index < 0 or sample_index >= len(animations):
        raise ValueError(f"sample_index {sample_index} out of range (0..{len(animations) - 1})")

    nodes    = gltf.get("nodes") or []
    anim     = animations[sample_index]
    samplers = anim.get("samplers") or []
    channels = anim.get("channels") or []

    # Same dedup-on-output decoding as the single-action path.
    decoded_outputs: dict[int, list[tuple]] = {}
    decoded_inputs:  dict[int, list[float]] = {}

    def _input_for(idx: int) -> list[float]:
        s = samplers[idx]
        ii = s["input"]
        if ii not in decoded_inputs:
            decoded_inputs[ii] = _read_floats(gltf, ii, "SCALAR")
        return decoded_inputs[ii]

    def _quats_for(idx: int) -> list[tuple[float, float, float, float]]:
        s = samplers[idx]
        oi = s["output"]
        if oi not in decoded_outputs:
            f = _read_floats(gltf, oi, "VEC4")
            decoded_outputs[oi] = [
                (f[i], f[i + 1], f[i + 2], f[i + 3])
                for i in range(0, len(f), 4)
            ]
        return decoded_outputs[oi]

    def _vec3s_for(idx: int) -> list[tuple[float, float, float]]:
        s = samplers[idx]
        oi = s["output"]
        if oi not in decoded_outputs:
            f = _read_floats(gltf, oi, "VEC3")
            decoded_outputs[oi] = [
                (f[i], f[i + 1], f[i + 2])
                for i in range(0, len(f), 3)
            ]
        return decoded_outputs[oi]

    pose = armature_obj.pose
    for pb in pose.bones:
        pb.rotation_mode = 'QUATERNION'

    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()

    mw_rot   = armature_obj.matrix_world.to_quaternion().to_matrix()
    mw_rot_t = mw_rot.transposed()
    arm_world_inv = armature_obj.matrix_world.inverted()

    skipped: list[str] = []
    actions: list[bpy.types.Action] = []

    for fs, fe, action_name in blocks:
        action = bpy.data.actions.new(name=action_name)

        # Build the layered-Action structure manually instead of relying on
        # ``bone.keyframe_insert`` (which writes through the armature's
        # active action). Switching ``animation_data.action`` between the N
        # blocks during a single bake leaves Blender 5.x's NLA evaluator in
        # a stale state where strips referencing the just-touched actions
        # silently produce zero contribution. Writing directly to the
        # action's channelbag fcurves bypasses ``animation_data.action``
        # entirely, so each new action is born clean and its NLA strip
        # evaluates the moment we push it.
        layer = action.layers.new(name="Layer")
        strip = layer.strips.new(type='KEYFRAME')
        slot = action.slots.new(id_type='OBJECT', name=armature_obj.name)
        cb = strip.channelbag(slot, ensure=True)

        fcache: dict[tuple[str, int], bpy.types.FCurve] = {}

        def _fc(data_path: str, index: int) -> bpy.types.FCurve:
            key = (data_path, index)
            fc = fcache.get(key)
            if fc is None:
                fc = cb.fcurves.new(data_path=data_path, index=index)
                fcache[key] = fc
            return fc

        def _write_kps(fc: bpy.types.FCurve, frames: list[int], values: list[float]) -> None:
            """Bulk-add keyframes via ``foreach_set("co", ...)``."""
            if not frames:
                return
            n = len(frames)
            base = len(fc.keyframe_points)
            fc.keyframe_points.add(n)
            flat = [0.0] * (2 * n)
            for i, (f, v) in enumerate(zip(frames, values)):
                flat[2 * i] = float(f)
                flat[2 * i + 1] = float(v)
            # foreach_set walks the entire collection, so write into a fresh
            # buffer that mirrors the existing keyframes plus the new ones.
            full = [0.0] * (2 * (base + n))
            for i, kp in enumerate(fc.keyframe_points[:base]):
                full[2 * i]     = kp.co[0]
                full[2 * i + 1] = kp.co[1]
            for i, (f, v) in enumerate(zip(frames, values)):
                full[2 * (base + i)]     = float(f)
                full[2 * (base + i) + 1] = float(v)
            fc.keyframe_points.foreach_set("co", full)
            fc.update()

        # Rotation channels (one per non-root bone, plus root).
        for ch in channels:
            target = ch.get("target") or {}
            if target.get("path") != ROTATION_PATH:
                continue
            node_idx = target.get("node")
            if node_idx is None or node_idx >= len(nodes):
                continue
            joint_name = nodes[node_idx].get("name", "")
            bone = pose.bones.get(joint_name)
            if bone is None:
                skipped.append(joint_name)
                continue

            timestamps = _input_for(ch["sampler"])
            quats      = _quats_for(ch["sampler"])

            ML     = bone.bone.matrix_local.to_3x3()
            ML_inv = ML.transposed()

            data_path = f'pose.bones["{joint_name}"].rotation_quaternion'
            buf_w: list[float] = []
            buf_x: list[float] = []
            buf_y: list[float] = []
            buf_z: list[float] = []
            buf_f: list[int]   = []

            for ts, q_mmcp in zip(timestamps, quats):
                frame = _frame_from_time(ts, gltf, request_start_frame)
                if not (fs <= frame <= fe):
                    continue
                qx, qy, qz, qw = q_mmcp
                R_mmcp          = Quaternion((qw, qx, qy, qz)).to_matrix()
                R_blender_world = _MMCP_TO_BLENDER @ R_mmcp @ _MMCP_TO_BLENDER.transposed()
                R_blender_arm   = mw_rot_t @ R_blender_world @ mw_rot
                R_bone          = ML_inv @ R_blender_arm @ ML
                qb              = R_bone.to_quaternion()
                buf_f.append(frame)
                buf_w.append(qb.w)
                buf_x.append(qb.x)
                buf_y.append(qb.y)
                buf_z.append(qb.z)

            _write_kps(_fc(data_path, 0), buf_f, buf_w)
            _write_kps(_fc(data_path, 1), buf_f, buf_x)
            _write_kps(_fc(data_path, 2), buf_f, buf_y)
            _write_kps(_fc(data_path, 3), buf_f, buf_z)

        # Translation channels (root bone only in practice).
        for ch in channels:
            target = ch.get("target") or {}
            if target.get("path") != TRANSLATION_PATH:
                continue
            node_idx = target.get("node")
            if node_idx is None or node_idx >= len(nodes):
                continue
            joint_name = nodes[node_idx].get("name", "")
            bone = pose.bones.get(joint_name)
            if bone is None:
                skipped.append(joint_name)
                continue

            timestamps = _input_for(ch["sampler"])
            positions  = _vec3s_for(ch["sampler"])

            rest_head = bone.bone.head_local
            ML        = bone.bone.matrix_local.to_3x3()
            ML_T      = ML.transposed()

            data_path = f'pose.bones["{joint_name}"].location'
            buf_x: list[float] = []
            buf_y: list[float] = []
            buf_z: list[float] = []
            buf_f: list[int]   = []

            for ts, p_mmcp in zip(timestamps, positions):
                frame = _frame_from_time(ts, gltf, request_start_frame)
                if not (fs <= frame <= fe):
                    continue
                world     = Vector(coords.mmcp_pos_to_blender(p_mmcp))
                arm_local = arm_world_inv @ world
                delta     = arm_local - rest_head
                local_t   = ML_T @ delta
                buf_f.append(frame)
                buf_x.append(local_t.x)
                buf_y.append(local_t.y)
                buf_z.append(local_t.z)

            _write_kps(_fc(data_path, 0), buf_f, buf_x)
            _write_kps(_fc(data_path, 1), buf_f, buf_y)
            _write_kps(_fc(data_path, 2), buf_f, buf_z)

        if anchor_frames is not None:
            block_anchors = {f for f in anchor_frames if fs <= f <= fe}
            _tag_keyframe_types(action, block_anchors)

        actions.append(action)

    if skipped and actions:
        actions[0]["proscenium_skipped_joints"] = sorted(set(skipped))

    # We never touched ``animation_data.action`` (we wrote fcurves directly
    # via the layered-Action API), so there's nothing to clear here. The
    # caller is responsible for NLA pushing and active-action management.

    return actions


_BONE_FOLLOWING_CONSTRAINTS = frozenset({
    "COPY_TRANSFORMS",
    "COPY_ROTATION",
    "COPY_LOCATION",
    "IK",
})


def _project_point_onto_plane(point: Vector, plane_origin: Vector, plane_normal: Vector) -> Vector:
    n = plane_normal.normalized()
    return point - (point - plane_origin).dot(n) * n


def _ik_pole_position(b1, b2) -> Vector:
    """Geometric pole-vector position from a 2-bone IK chain (matches the
    method used by the Mixamo Rig addon's bake_anim). ``b1`` is the upper
    bone (e.g. arm/upleg), ``b2`` is the lower bone (forearm/leg).
    """
    plane_normal = b1.head - b2.tail
    midpoint = (b1.head + b2.tail) * 0.5
    prepole_dir = b2.head - midpoint
    pole_pos = b2.head + prepole_dir.normalized()
    pole_pos = _project_point_onto_plane(pole_pos, b2.head, plane_normal)
    return b2.head + (pole_pos - b2.head).normalized() * (b2.head - b1.head).magnitude * 1.7


def _trace_chain_end_to_deform(armature_obj, chain_end_pb, deform_names: set[str]):
    """For an IK chain whose end is ``chain_end_pb`` (a non-deform bone with
    an IK constraint), return the deform bone whose head sits at the IK
    target's intended position — i.e., the next bone after the deform
    shadow of ``chain_end_pb``. Used so a "Hand IK" control gets a full
    matrix spec instead of just rotation.

    Walks: chain_end_pb is COPY_TRANSFORMS-sourced by some deform bone D
    (e.g. mixamorig1:LeftForeArm); its child in the deform tree is the
    bone whose head is at the chain end's tail (mixamorig1:LeftHand).
    """
    for pb in armature_obj.pose.bones:
        if pb.name not in deform_names:
            continue
        for c in pb.constraints:
            if getattr(c, "mute", False) or getattr(c, "influence", 1.0) <= 0.0:
                continue
            if c.type not in _BONE_FOLLOWING_CONSTRAINTS:
                continue
            if (getattr(c, "target", None) is armature_obj
                    and getattr(c, "subtarget", "") == chain_end_pb.name):
                children_in_deform = [ch for ch in pb.children if ch.name in deform_names]
                if children_in_deform:
                    return children_in_deform[0].name
                return None
    return None


def _build_control_specs(armature_obj: bpy.types.Object) -> dict[str, tuple]:
    """Return a mapping ``control_bone_name → spec``, derived from the
    constraint graph, saying how to compute each control bone's target
    POSE matrix from the deform pose at any frame.

    Spec kinds:
      ``("matrix", deform_name)``         full deform matrix
      ``("location_head", deform_name)``  ctrl.head = deform.head
      ``("location_tail", deform_name)``  ctrl.head = deform.tail
      ``("pole", b1_name, b2_name)``      geometric pole position

    Two passes:
      1. Constraints on deform bones — Copy*/IK subtargets become specs.
      2. IK constraints anywhere in the rig — typical Mixamo Control Rig
         keeps the actual IK on intermediate non-deform bones (e.g.
         ForeArm_IK_Left). Their target/pole_subtargets need handling
         too, since a hand-IK control is *both* a rotation source for
         the deform hand AND an IK chain target. Upgrade rotation-only
         specs to full matrix specs in that case.
    """
    bone_names = {pb.name for pb in armature_obj.pose.bones}
    deform_names = {pb.name for pb in armature_obj.pose.bones if pb.bone.use_deform}

    specs: dict[str, tuple] = {}

    # Pass 1: constraints ON deform bones (the user's pose-feedback source).
    # COPY_ROTATION is upgraded to a full matrix spec — the control bone
    # typically sits at the deform bone's head anyway, and we need its
    # position correct for any IK chain that uses it as a target later.
    for pb in armature_obj.pose.bones:
        if pb.name not in deform_names:
            continue
        for c in pb.constraints:
            if getattr(c, "mute", False) or getattr(c, "influence", 1.0) <= 0.0:
                continue
            if c.type not in _BONE_FOLLOWING_CONSTRAINTS:
                continue
            target = getattr(c, "target", None)
            sub = getattr(c, "subtarget", "")
            if target is not armature_obj or sub not in bone_names:
                continue
            if c.type in ("COPY_TRANSFORMS", "COPY_ROTATION"):
                specs.setdefault(sub, ("matrix", pb.name))
            elif c.type == "COPY_LOCATION":
                specs.setdefault(sub, ("location_head", pb.name))
            elif c.type == "IK":
                specs.setdefault(sub, ("location_tail", pb.name))

    # Pass 2: IK constraints anywhere — covers IK chains hidden inside the
    # control rig (Mixamo's ForeArm_IK_Left has the IK; mixamorig1:LeftHand
    # only sees the COPY_ROTATION above). Upgrades / adds specs for IK
    # targets and adds pole-vector specs.
    for pb in armature_obj.pose.bones:
        for c in pb.constraints:
            if c.type != "IK" or getattr(c, "mute", False) or getattr(c, "influence", 1.0) <= 0.0:
                continue
            if getattr(c, "target", None) is not armature_obj:
                continue
            sub = getattr(c, "subtarget", "")
            if sub and sub in bone_names:
                deform_for_target = _trace_chain_end_to_deform(armature_obj, pb, deform_names)
                if deform_for_target is not None:
                    # Promote (or set) to full matrix from the bone whose
                    # head should align with the IK target — gives both
                    # rotation and the position the chain needs to settle.
                    specs[sub] = ("matrix", deform_for_target)
            pole = getattr(c, "pole_subtarget", "")
            if pole and pole in bone_names and pb.parent is not None:
                specs.setdefault(pole, ("pole", pb.parent.name, pb.name))

    # Pass 3: walk parents of every spec'd bone, lifting each non-deform
    # parent into the spec set with the same matrix target. Catches the
    # user-facing handles that wrap an internal target — e.g. Mixamo's
    # ``Ctrl_Foot_IK_Left`` is the bone the animator poses, but the
    # constraint graph only points at its child ``Foot_IK_Left``. With
    # this lift, the handle gets baked too and follows the generated
    # motion in the viewport. Only walks one parent at a time per pass,
    # so we don't lift a hierarchy-wide master controller into every
    # limb's bake.
    extra: dict[str, tuple] = {}
    for ctrl_name, spec in specs.items():
        if spec[0] != "matrix":
            continue
        pb = armature_obj.pose.bones.get(ctrl_name)
        if pb is None or pb.parent is None:
            continue
        parent_name = pb.parent.name
        if parent_name in deform_names or parent_name in specs or parent_name in extra:
            continue
        # Use the same deform source as the child — the child sits at this
        # parent's frame, so matching the parent to the same deform gives
        # the parent the desired world pose and leaves the child at local
        # identity (= correct cumulative).
        extra[parent_name] = ("matrix", spec[1])
    specs.update(extra)

    # Pass 4: drop bones whose pose at playback won't be governed by our
    # keyed matrix_basis. If a bone has any constraint other than
    # CHILD_OF (which we explicitly invert), the rig's constraint stack
    # will override the keyframes — TRACK_TO, COPY_LOCATION, TRANSFORM,
    # LIMIT_*, etc. all write directly to the visual transform regardless
    # of basis. Mixamo Control Rig stuffs these on internal helpers
    # (Foot_IK_Left, ToeEnd_Left, FootHeelMid_Left, Foot_IK_target_Left)
    # — leaving them unbaked lets the IK solver and foot-roll chain do
    # their job at playback, instead of fighting baked noise.
    _SAFE_CONSTRAINT_TYPES = {"CHILD_OF"}
    filtered: dict[str, tuple] = {}
    for ctrl_name, spec in specs.items():
        pb = armature_obj.pose.bones.get(ctrl_name)
        if pb is None:
            continue
        if any(
            (not getattr(c, "mute", False) and getattr(c, "influence", 1.0) > 0
             and c.type not in _SAFE_CONSTRAINT_TYPES)
            for c in pb.constraints
        ):
            continue
        filtered[ctrl_name] = spec
    return filtered


def _desired_pose_matrix(armature_obj, ctrl_pb, spec) -> Matrix:
    """The control bone's target matrix in *POSE* (armature-local) space at
    the currently-evaluated frame. Caller converts to LOCAL with
    ``armature.convert_space`` before writing the keyframe.
    """
    kind = spec[0]
    if kind == "matrix":
        return armature_obj.pose.bones[spec[1]].matrix.copy()
    if kind == "rotation":
        deform = armature_obj.pose.bones[spec[1]]
        out = ctrl_pb.matrix.copy()
        # Replace rotation, keep location/scale.
        rot = deform.matrix.to_quaternion().to_matrix().to_4x4()
        rot.translation = out.translation
        return rot
    if kind == "location_head":
        deform = armature_obj.pose.bones[spec[1]]
        out = ctrl_pb.matrix.copy()
        out.translation = deform.head
        return out
    if kind == "location_tail":
        deform = armature_obj.pose.bones[spec[1]]
        out = ctrl_pb.matrix.copy()
        out.translation = deform.tail
        return out
    if kind == "pole":
        b1 = armature_obj.pose.bones[spec[1]]
        b2 = armature_obj.pose.bones[spec[2]]
        return Matrix.Translation(_ik_pole_position(b1, b2))
    return ctrl_pb.matrix.copy()


_TEMP_TAG = "_proscenium_bake_"


def _bake_to_control_rig(
    armature_obj: bpy.types.Object,
    *,
    source_action: bpy.types.Action,
    frame_start: int,
    frame_end: int,
) -> None:
    """Bake deform-bone keyframes onto the Mixamo control rig.

    The deform fcurves are already on ``source_action`` (we wrote them
    earlier in ``bake_gltf_to_armature``). On a control rig those keys
    don't reach the screen — the Copy*/IK constraints make the deform
    bones follow control bones, not the other way round. So we add
    control-bone keyframes to the same action: at every frame in the
    range, evaluate where the deform bones want to be and project that
    back onto the control rig.

    Implementation: build a clean deform-only source armature out of a
    duplicate of ``armature_obj``, then run our ported version of the
    Mixamo addon's retarget-and-bake (``mixamo_bake``). The port writes
    into the supplied ``source_action`` instead of creating a new one,
    which the addon's ``mr.import_anim_to_rig`` operator does not allow.
    """
    if "mr_control_rig" not in armature_obj.data.keys():
        print("[proscenium] target armature is not flagged as a Mixamo control rig")
        return

    from . import mixamo_bake

    saved_active = bpy.context.view_layer.objects.active
    saved_mode = armature_obj.mode
    if saved_mode != 'OBJECT':
        bpy.context.view_layer.objects.active = armature_obj
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

    src_arm = None

    try:
        # 1. Duplicate the control rig into a source. ``bpy.ops.object.duplicate``
        # carries the action + slot assignment correctly on Blender 4.4+;
        # a manual ``data.copy()`` + ``animation_data_create()`` does not,
        # leaving fcurves unbound and the source evaluating as static.
        # ``select_all`` via the operator needs a 3D-View area context (the
        # multi-frame path has one because it's invoked from a button; the
        # single-pose path is called from a thread/timer and doesn't), so
        # do the deselect by hand. ``object.duplicate`` itself only needs
        # one selected object + an active.
        for o in bpy.context.view_layer.objects:
            try:
                o.select_set(False)
            except Exception:
                pass
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.duplicate(linked=False)
        src_arm = bpy.context.view_layer.objects.active
        src_arm.name = f"{armature_obj.name}_proscenium_src"
        src_arm["proscenium_temp_source"] = True

        # 2. Strip the duplicate down to deform-only. Required so the
        # bake's IK-pole branch (which keys off Ctrl_*Pole_* names) only
        # fires on the *target* rig where ``ik_data`` is populated.
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            ebones = src_arm.data.edit_bones
            to_delete = [eb for eb in ebones if not src_arm.data.bones[eb.name].use_deform]
            for eb in to_delete:
                ebones.remove(eb)
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')

        # 3. Mute Copy*/IK on the remaining (deform) bones so the source
        # evaluates from its own keyframes, not from the control bones
        # we just deleted.
        for pb in src_arm.pose.bones:
            for c in pb.constraints:
                if c.type in _BONE_FOLLOWING_CONSTRAINTS:
                    c.mute = True

        # 3b. Strip the source action's fcurves for any bone that no
        # longer exists on src_arm. Without this, when ``apply_anim_to_control_rig``
        # later re-creates helper bones with names like ``Ctrl_Foot_IK_Left``,
        # the leftover Ctrl_* fcurves drive matrix_basis on those new
        # helpers — corrupting the COPY_LOCATION read that the bake
        # depends on. (The original Mixamo addon doesn't hit this
        # because its source is a freshly imported FBX with no Ctrl_*
        # fcurves in its action.)
        if src_arm.animation_data and src_arm.animation_data.action:
            existing_bones = {pb.name for pb in src_arm.pose.bones}
            from . import mixamo_bake as _mb
            fcurves = _mb._action_fcurves(src_arm.animation_data.action, src_arm)
            if fcurves is not None:
                to_remove = []
                for fc in fcurves:
                    dp = fc.data_path
                    if not dp.startswith('pose.bones["'):
                        continue
                    bone_name = dp.split('"')[1]
                    if bone_name not in existing_bones:
                        to_remove.append(fc)
                for fc in to_remove:
                    try:
                        fcurves.remove(fc)
                    except Exception:
                        pass

        # 4. Hand off to our bake. It adds helper bones + retarget
        # constraints internally, runs the per-frame bake into
        # ``source_action``, and tears down the retarget rig before
        # returning.
        baked = mixamo_bake.apply_anim_to_control_rig(
            src_arm,
            armature_obj,
            action=source_action,
            frame_start=int(frame_start),
            frame_end=int(frame_end),
        )
        print(f"[proscenium] baked control bones: {baked}")
    except Exception as exc:  # noqa: BLE001 — best-effort delegate
        print(f"[proscenium] control-rig bake failed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: remove the source duplicate (and any internal copies
        # tagged with ``mix_to_del`` for parity with the addon).
        for o in list(bpy.data.objects):
            try:
                if o.get("mix_to_del") or o.get("proscenium_temp_source"):
                    bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass

        # Restore mode + active
        try:
            if armature_obj.mode != saved_mode and saved_mode in {'OBJECT', 'POSE', 'EDIT'}:
                bpy.context.view_layer.objects.active = armature_obj
                bpy.ops.object.mode_set(mode=saved_mode)
        except Exception:
            pass
        bpy.context.view_layer.objects.active = saved_active


def _undo_constraint_effects(
    armature_obj: bpy.types.Object,
    ctrl_pb: bpy.types.PoseBone,
    pose_mat: 'Matrix',
) -> 'Matrix':
    """Walk a control bone's *existing* constraints in reverse and unwind
    each one's effect on ``pose_mat``. The result is the pre-constraint
    armature-space pose — the matrix that, when multiplied through the
    bone hierarchy and run through the constraint stack at playback,
    will produce ``pose_mat`` again.

    Necessary because ``Object.convert_space(POSE → LOCAL)`` treats POSE
    as already-pre-constraint and computes LOCAL purely from the bone
    hierarchy. CHILD_OF (and similar transform-modifying constraints)
    aren't inverted automatically — without this compensation, every IK
    handle / pole vector with a CHILD_OF parent ends up offset at
    playback by exactly the target bone's pose.

    Skips our own temp constraints (tagged with ``_TEMP_TAG``); they're
    only there to set ``pb.matrix`` for the read and don't survive into
    the baked action.
    """
    out = pose_mat.copy()
    for c in reversed(list(ctrl_pb.constraints)):
        if getattr(c, "mute", False) or getattr(c, "influence", 1.0) <= 0.0:
            continue
        if c.name.startswith(_TEMP_TAG):
            continue
        if c.type == "CHILD_OF":
            target = getattr(c, "target", None)
            sub = getattr(c, "subtarget", "")
            if target is not armature_obj or not sub:
                continue
            tbone = armature_obj.pose.bones.get(sub)
            if tbone is None:
                continue
            # CHILD_OF eval: output = target.matrix @ inverse_matrix @ input
            # Inverting:    input  = inverse_matrix.inverted() @
            #                        target.matrix.inverted()  @ output
            out = c.inverse_matrix.inverted() @ tbone.matrix.inverted() @ out
        # Other transform-modifying constraints (Limit Location, Limit
        # Rotation, Transformation, etc.) could be added here in the
        # same shape if a rig needs them.
    return out


def _project_to_control_rig(
    armature_obj: bpy.types.Object,
    *,
    action: bpy.types.Action,
    frame_start: int,
    frame_end: int,
) -> None:
    """Project freshly-baked deform-bone keyframes onto the control rig.

    The math relies on the constraint stack being *active* during the
    per-frame evaluation: ``convert_space(POSE → LOCAL)`` derives a
    bone-local basis relative to the parent's *currently-evaluated*
    pose, and that basis only round-trips at playback if the parent is
    at the same evaluated pose then. The Mixamo Rig addon's ``bake_anim``
    leverages this with constraints already in place; we install
    temporary "deform → control" constraints on every spec'd control
    bone so the depsgraph propagates the target pose through the chain
    during the bake, then strip them.

    Pole vectors don't have a corresponding deform bone — they're solved
    by IK position. Computed geometrically each frame from the IK
    chain's two bones (matches Mixamo addon's ``get_ik_pole_pos``).
    """
    specs = _build_control_specs(armature_obj)
    if not specs:
        return

    saved_active = bpy.context.view_layer.objects.active
    saved_mode = armature_obj.mode
    bpy.context.view_layer.objects.active = armature_obj
    if armature_obj.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

    # Mute the original control→deform constraints so the deform bones
    # hold the just-baked rotation+translation keyframes during the
    # per-frame eval. Restored at the end.
    saved_mute: list[tuple[str, str, bool]] = []
    for pb in armature_obj.pose.bones:
        if not pb.bone.use_deform:
            continue
        for c in pb.constraints:
            if c.type in _BONE_FOLLOWING_CONSTRAINTS:
                saved_mute.append((pb.name, c.name, c.mute))
                c.mute = True

    # Install temp deform→control constraints. ``COPY_TRANSFORMS`` for
    # full-matrix specs, ``COPY_LOCATION`` (with head_tail) for the
    # location-only ones. Pole bones get no temp constraint — handled
    # geometrically per frame instead.
    temp_constraints: list[tuple[str, str]] = []
    for ctrl_name, spec in specs.items():
        if spec[0] == "pole":
            continue
        ctrl_pb = armature_obj.pose.bones.get(ctrl_name)
        if ctrl_pb is None:
            continue
        kind = spec[0]
        if kind == "matrix":
            c = ctrl_pb.constraints.new(type="COPY_TRANSFORMS")
        elif kind == "location_head":
            c = ctrl_pb.constraints.new(type="COPY_LOCATION")
            c.head_tail = 0.0
        elif kind == "location_tail":
            c = ctrl_pb.constraints.new(type="COPY_LOCATION")
            c.head_tail = 1.0
        else:
            continue
        c.name = f"{_TEMP_TAG}{ctrl_name}"
        c.target = armature_obj
        c.subtarget = spec[1]
        temp_constraints.append((ctrl_name, c.name))

    scene = bpy.context.scene
    saved_frame = scene.frame_current

    # Per-bone, per-frame matrix_basis collected first; written as
    # fcurves in one batch at the end (faster than keyframe_insert and
    # avoids stale handles on dense timelines).
    matrices: dict[str, list[tuple[int, Matrix]]] = {name: [] for name in specs}

    try:
        for f in range(int(frame_start), int(frame_end) + 1):
            scene.frame_set(f)
            bpy.context.view_layer.update()
            for ctrl_name, spec in specs.items():
                ctrl_pb = armature_obj.pose.bones.get(ctrl_name)
                if ctrl_pb is None:
                    continue
                if spec[0] == "pole":
                    b1 = armature_obj.pose.bones[spec[1]]
                    b2 = armature_obj.pose.bones[spec[2]]
                    pose_mat = Matrix.Translation(_ik_pole_position(b1, b2))
                else:
                    # Temp constraint already drove this bone; its
                    # ``matrix`` reflects the deform-bone target via the
                    # depsgraph evaluation, with parent chains resolved.
                    pose_mat = ctrl_pb.matrix.copy()
                # Undo any non-temp constraint effects (CHILD_OF mostly)
                # before converting to LOCAL — convert_space treats its
                # input as pre-constraint and won't unwind CHILD_OF on
                # its own, so an IK handle parented via CHILD_OF would
                # otherwise come out offset by the target bone's pose.
                pose_mat = _undo_constraint_effects(armature_obj, ctrl_pb, pose_mat)
                local_mat = armature_obj.convert_space(
                    pose_bone=ctrl_pb, matrix=pose_mat,
                    from_space="POSE", to_space="LOCAL",
                )
                matrices[ctrl_name].append((f, local_mat))
    finally:
        scene.frame_set(saved_frame)
        # Strip temp constraints first, then unmute originals — order
        # doesn't matter functionally but stays clean if anything throws.
        for ctrl_name, c_name in temp_constraints:
            ctrl_pb = armature_obj.pose.bones.get(ctrl_name)
            if ctrl_pb is None:
                continue
            c = ctrl_pb.constraints.get(c_name)
            if c is not None:
                ctrl_pb.constraints.remove(c)
        for deform_name, c_name, mute in saved_mute:
            pb = armature_obj.pose.bones.get(deform_name)
            if pb is None:
                continue
            c = pb.constraints.get(c_name)
            if c is not None:
                c.mute = mute
        if saved_mode != armature_obj.mode:
            bpy.ops.object.mode_set(mode=saved_mode)
        bpy.context.view_layer.objects.active = saved_active

    _write_control_keyframes(armature_obj, action, matrices)


def _write_control_keyframes(
    armature_obj: bpy.types.Object,
    action: bpy.types.Action,
    matrices: dict[str, list[tuple[int, 'Matrix']]],
) -> None:
    """Apply collected per-frame LOCAL matrices to each control bone and
    write rotation/location/scale fcurves in one batch per channel."""
    fcurves_container = _action_fcurves_container(armature_obj, action)
    if fcurves_container is None:
        return

    quat_prev: dict[str, 'Quaternion'] = {}
    euler_prev: dict[str, 'Euler'] = {}

    for bone_name, frames in matrices.items():
        if not frames:
            continue
        pb = armature_obj.pose.bones.get(bone_name)
        if pb is None:
            continue

        rot_mode = pb.rotation_mode
        location_keys: list[float] = [[], [], []]    # per axis: [f0,v0,f1,v1,...]
        scale_keys:    list[float] = [[], [], []]
        rot_keys: list[list[float]] = [[] for _ in range(4 if rot_mode == "QUATERNION" else 3)]

        for f, local_mat in frames:
            pb.matrix_basis = local_mat.copy()
            for i, v in enumerate(pb.location):
                location_keys[i].extend((f, v))
            for i, v in enumerate(pb.scale):
                scale_keys[i].extend((f, v))
            if rot_mode == "QUATERNION":
                q = pb.rotation_quaternion.copy()
                if bone_name in quat_prev:
                    q.make_compatible(quat_prev[bone_name])
                    pb.rotation_quaternion = q
                quat_prev[bone_name] = q
                for i, v in enumerate(q):
                    rot_keys[i].extend((f, v))
            elif rot_mode == "AXIS_ANGLE":
                for i, v in enumerate(pb.rotation_axis_angle):
                    rot_keys[i].extend((f, v))
            else:
                e = pb.rotation_euler.copy()
                if bone_name in euler_prev:
                    e.make_compatible(euler_prev[bone_name])
                    pb.rotation_euler = e
                euler_prev[bone_name] = e
                for i, v in enumerate(e):
                    rot_keys[i].extend((f, v))

        rot_prop = {
            "QUATERNION":  "rotation_quaternion",
            "AXIS_ANGLE":  "rotation_axis_angle",
        }.get(rot_mode, "rotation_euler")
        dp_loc = f'pose.bones["{bone_name}"].location'
        dp_rot = f'pose.bones["{bone_name}"].{rot_prop}'
        dp_scl = f'pose.bones["{bone_name}"].scale'

        for ax in range(3):
            _set_fcurve_keyframes(fcurves_container, dp_loc, ax, location_keys[ax])
            _set_fcurve_keyframes(fcurves_container, dp_scl, ax, scale_keys[ax])
        for ax in range(len(rot_keys)):
            _set_fcurve_keyframes(fcurves_container, dp_rot, ax, rot_keys[ax])


def _action_fcurves_container(armature_obj, action):
    """The fcurves collection on the armature's *active slot* — same
    helper logic as ``path_follow._active_fcurves`` but inlined here so
    this module stays standalone.
    """
    flat = getattr(action, "fcurves", None)
    if flat is not None:
        return flat
    slot = getattr(armature_obj.animation_data, "action_slot", None)
    if slot is None:
        return None
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            if not hasattr(strip, "channelbag"):
                continue
            cb = strip.channelbag(slot, ensure=True)
            if cb is not None:
                return cb.fcurves
    return None


def _set_fcurve_keyframes(fcurves, data_path: str, axis: int, key_values: list[float]) -> None:
    """Replace ``data_path[axis]``'s fcurve keyframes with the (frame, value)
    pairs flattened in ``key_values``. Single ``foreach_set`` write.
    """
    if not key_values:
        return
    fc = fcurves.find(data_path=data_path, index=axis)
    if fc is None:
        fc = fcurves.new(data_path=data_path, index=axis)
    # Drop any existing keys; we own the channel after the bake.
    while len(fc.keyframe_points):
        fc.keyframe_points.remove(fc.keyframe_points[-1], fast=True)
    n = len(key_values) // 2
    fc.keyframe_points.add(n)
    fc.keyframe_points.foreach_set("co", key_values)
    fc.update()


def _tag_keyframe_types(action: bpy.types.Action, anchor_frames: set[int]) -> None:
    """Walk every fcurve on ``action`` and set each keyframe point's
    ``type`` to ``'KEYFRAME'`` if its frame is in ``anchor_frames``, else
    ``'GENERATED'``. Makes the two kinds visually distinct in the dopesheet.
    """
    anchors = {int(round(f)) for f in anchor_frames}

    def _tag(fcurves):
        for fc in fcurves:
            for kp in fc.keyframe_points:
                kp.type = 'KEYFRAME' if int(round(kp.co[0])) in anchors else 'GENERATED'

    flat = getattr(action, "fcurves", None)
    if flat is not None:
        _tag(flat)
        return
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for slot in getattr(action, "slots", ()):
                cb = strip.channelbag(slot, ensure=False) if hasattr(strip, "channelbag") else None
                if cb is not None:
                    _tag(cb.fcurves)


# ---------------------------------------------------------------------------
# Single-pose insertion (pose generator path — additive, non-destructive)
# ---------------------------------------------------------------------------

def bake_single_pose(
    gltf: dict[str, Any],
    armature_obj: bpy.types.Object,
    *,
    source_frame: int,
    target_frame: int,
    sample_index: int = 0,
    root_translation: str = "skip",
) -> int:
    """Insert one keyframe per joint at ``target_frame`` in the armature's
    active action, reading rotations from ``gltf``'s ``sample_index`` at
    ``source_frame``.

    On a Mixamo control rig the deform-bone keyframes alone don't show
    up — the Copy*/IK constraints make the deform bones follow the
    control bones. After writing the deform keys we project them onto
    the control rig with a single-frame bake.

    Returns the number of bone channels updated. Does **not** touch or
    replace the armature's action — it's purely additive, so the user can
    undo with a single Ctrl+Z.
    """
    if armature_obj is None or armature_obj.type != 'ARMATURE':
        raise ValueError("bake target must be an ARMATURE object")

    animations = gltf.get("animations") or []
    if not animations or sample_index >= len(animations):
        raise ValueError("response is missing the requested sample")

    nodes    = gltf.get("nodes") or []
    anim     = animations[sample_index]
    samplers = anim.get("samplers") or []
    channels = anim.get("channels") or []

    # Armature must have an action to receive keyframes. Create one if needed.
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    if armature_obj.animation_data.action is None:
        armature_obj.animation_data.action = bpy.data.actions.new("Proscenium_Pose")

    pose = armature_obj.pose
    for pb in pose.bones:
        pb.rotation_mode = 'QUATERNION'

    # Undo the armature's ``matrix_world`` rotation on the way in; matches
    # the outbound conversion in ``sample_pose_keyframes``. No-op when
    # matrix_world is identity.
    mw_rot = armature_obj.matrix_world.to_quaternion().to_matrix()
    mw_rot_t = mw_rot.transposed()

    if root_translation not in ("skip", "height_only", "full"):
        raise ValueError(
            f"root_translation must be 'skip', 'height_only', or 'full', got {root_translation!r}"
        )

    written = 0
    for ch in channels:
        target = ch.get("target") or {}
        path = target.get("path")
        if path == TRANSLATION_PATH and root_translation == "skip":
            continue
        if path not in (ROTATION_PATH, TRANSLATION_PATH):
            continue

        node_idx = target.get("node")
        if node_idx is None or node_idx >= len(nodes):
            continue
        joint_name = nodes[node_idx].get("name", "")
        bone = pose.bones.get(joint_name)
        if bone is None:
            continue

        sampler = samplers[ch["sampler"]]

        ML = bone.bone.matrix_local.to_3x3()

        if path == ROTATION_PATH:
            quats = _read_floats(gltf, sampler["output"], "VEC4")
            if source_frame * 4 + 4 > len(quats):
                continue
            qx, qy, qz, qw = quats[source_frame * 4:(source_frame + 1) * 4]
            R_mmcp = Quaternion((qw, qx, qy, qz)).to_matrix()
            R_blender_world = _MMCP_TO_BLENDER @ R_mmcp @ _MMCP_TO_BLENDER.transposed()
            R_blender_arm = mw_rot_t @ R_blender_world @ mw_rot
            R_bone = ML.transposed() @ R_blender_arm @ ML
            bone.rotation_quaternion = R_bone.to_quaternion()
            bone.keyframe_insert(data_path="rotation_quaternion", frame=target_frame)
        else:
            vec3s = _read_floats(gltf, sampler["output"], "VEC3")
            if source_frame * 3 + 3 > len(vec3s):
                continue
            p_mmcp = tuple(vec3s[source_frame * 3:(source_frame + 1) * 3])
            pose_world = Vector(coords.mmcp_pos_to_blender(p_mmcp))

            if root_translation == "height_only":
                # Keep the bone's current world xy (whatever the user has
                # placed the rig at, including any prior keyframe at this
                # frame) and only override world z with the generated pose's
                # height. Lets a "crouching" pose drop toward the floor
                # without yanking the character to the canonical origin in xy.
                current_world = armature_obj.matrix_world @ bone.head
                target_world = Vector((current_world.x, current_world.y, pose_world.z))
            else:  # "full"
                target_world = pose_world

            arm_local = armature_obj.matrix_world.inverted() @ target_world
            delta = arm_local - bone.bone.head_local
            bone.location = ML.transposed() @ delta
            bone.keyframe_insert(data_path="location", frame=target_frame)

        written += 1

    if written:
        from . import request_builder
        if request_builder.is_control_rig(armature_obj):
            _bake_to_control_rig(
                armature_obj,
                source_action=armature_obj.animation_data.action,
                frame_start=int(target_frame),
                frame_end=int(target_frame),
            )

    return written


# ---------------------------------------------------------------------------
# Frame index helper
# ---------------------------------------------------------------------------

def _frame_from_time(timestamp_seconds: float, gltf: dict[str, Any], start_frame: int) -> int:
    """Convert a sampler timestamp (seconds) to a Blender frame index.

    The MMCP_motion extension publishes ``fps`` so the addon can place
    keyframes on integer scene frames regardless of Blender's frame rate
    setting.
    """
    fps = float(read_extension_metadata(gltf).get("fps") or 30.0)
    return int(round(timestamp_seconds * fps)) + start_frame


# ---------------------------------------------------------------------------
# glTF accessor decoding (SCALAR, VEC3, VEC4 of float32 only — that's what
# the MMCP server emits in v1).
# ---------------------------------------------------------------------------

_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_FLOAT_COMPONENT = 5126   # glTF componentType


def _read_floats(gltf: dict[str, Any], accessor_idx: int, expected_type: str) -> list[float]:
    accessors    = gltf.get("accessors") or []
    buffer_views = gltf.get("bufferViews") or []
    buffers      = gltf.get("buffers") or []

    accessor = accessors[accessor_idx]
    if accessor.get("componentType") != _FLOAT_COMPONENT:
        raise ValueError(
            f"accessor {accessor_idx}: expected float32 (5126), got {accessor.get('componentType')}"
        )
    if accessor.get("type") != expected_type:
        raise ValueError(
            f"accessor {accessor_idx}: expected type {expected_type}, got {accessor.get('type')}"
        )

    components = _TYPE_COMPONENTS[expected_type]
    count      = int(accessor["count"])
    n_floats   = count * components

    view = buffer_views[accessor["bufferView"]]
    buf  = buffers[view["buffer"]]
    raw  = _decode_buffer(buf)

    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    end   = start + n_floats * 4
    if end > len(raw):
        raise ValueError(
            f"accessor {accessor_idx}: range {start}..{end} exceeds buffer length {len(raw)}"
        )

    return list(struct.unpack(f"<{n_floats}f", raw[start:end]))


def _decode_buffer(buf: dict[str, Any]) -> bytes:
    uri = buf.get("uri")
    if not uri:
        # External buffer (.bin sidecar) — not produced by mmcp_server in v1.
        raise ValueError("external buffer URIs are not supported")
    if not uri.startswith("data:"):
        raise ValueError(f"non-data URI buffers not supported: {uri[:40]}…")
    _, b64 = uri.split(",", 1)
    return base64.b64decode(b64)
