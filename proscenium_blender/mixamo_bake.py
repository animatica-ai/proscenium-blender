"""Bake deform-bone animation onto a Mixamo Rig control armature.

Ported from the Mixamo Rig addon's ``mr.import_anim_to_rig`` operator
(``mixamo_rig.py:_import_anim`` and ``lib/animation.py:bake_anim``) so we
own the action handling. The addon's operator always calls
``bpy.data.actions.new("Action")``, which clobbers the action name we
want and rules out single-frame use (the operator bakes the action's
full frame range). Owning the bake lets us:

  * Keep our action name (e.g. ``Proscenium_Motion: <prompt>``).
  * Take an explicit ``frame_start``/``frame_end`` — works for the full
    multi-frame generate path *and* the single-frame text-to-pose path.
  * Skip the addon's ``redefine_source_rest_pose`` step. The addon needs
    it because its source armature (raw Mixamo FBX) has different rest
    rolls than the user's character. Our source is a duplicate of the
    user's own armature, so the deform rest pose already matches.

The high-level flow mirrors ``_import_anim``:
  1. Detect IK/FK switch state on each limb.
  2. Build a source-bone → control-bone name map (FK chains rotate from
     mixamorig deform bones; IK chains rotate from helper bones we
     create on the source that follow the deform IK chain via
     ``COPY_TRANSFORMS``).
  3. Add helper edit bones on the source armature for IK targets and
     IK chain tips, so pole positions can be derived geometrically and
     IK target chains can be matrix-baked.
  4. Add retarget constraints on the target's control bones (mostly
     ``COPY_ROTATION``, plus ``COPY_LOCATION`` on Hips and IK
     targets).
  5. Bake selected control bones over the requested frame range into
     the supplied action.
  6. Tear down all temp constraints and helper bones.

Public entry: :func:`apply_anim_to_control_rig`.
"""

from __future__ import annotations

import bpy
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------------
# Constants — pulled from the Mixamo addon's definitions/naming.py.
# Duplicated here so this module doesn't depend on the addon being
# importable (it is at runtime, but the addon source isn't on Python's
# import path; only its operators are exposed via bpy.ops.mr.*).
# ---------------------------------------------------------------------------

C_PREFIX = "Ctrl_"

ARM_NAMES = {
    "shoulder": "Shoulder",
    "arm_ik":   "Arm_IK",
    "arm_fk":   "Arm_FK",
    "forearm_ik": "ForeArm_IK",
    "forearm_fk": "ForeArm_FK",
    "pole_ik":  "ArmPole_IK",
    "hand_ik":  "Hand_IK",
    "hand_fk":  "Hand_FK",
}

LEG_NAMES = {
    "thigh_ik": "UpLeg_IK",
    "thigh_fk": "UpLeg_FK",
    "calf_ik":  "Leg_IK",
    "calf_fk":  "Leg_FK",
    "foot_fk":  "Foot_FK",
    "foot_ik":  "Foot_IK",
    "pole_ik":  "LegPole_IK",
}


# ---------------------------------------------------------------------------
# Geometry helpers — copied from ``lib/maths_geo.py`` (only the bits we use).
# ---------------------------------------------------------------------------

def _project_point_onto_plane(q: Vector, p: Vector, n: Vector) -> Vector:
    n = n.normalized()
    return q - ((q - p).dot(n)) * n


def _get_ik_pole_pos(b1, b2, axis: Vector) -> Vector:
    """``method=2`` from ``lib/maths_geo.py:get_ik_pole_pos`` — pole sits
    along ``axis`` (z-axis midpoint for legs, x-axis of forearm for arms),
    distance equal to the lower bone's length. The addon uses this same
    method for the per-frame bake; it gives a stable pole that matches the
    Mixamo control-rig conventions.
    """
    return b2.head + (axis.normalized() * (b2.tail - b2.head).magnitude)


# ---------------------------------------------------------------------------
# Slotted-action compatibility (Blender 4.4+ / 5.0).
# ---------------------------------------------------------------------------

def _has_slotted_actions() -> bool:
    return bpy.app.version >= (4, 4, 0)


def _ensure_action_slot(anim_data, action, datablock) -> None:
    """Make sure ``anim_data`` has a slot bound for ``datablock``.

    In Blender 4.4+, an action without an assigned slot doesn't drive
    anything — the fcurves resolve to nothing. ``anim_data.action = X``
    used to do this implicitly; on 4.4+ you need to either pick from
    ``action_suitable_slots`` or call ``fcurve_ensure_for_datablock`` to
    create one.
    """
    if not _has_slotted_actions():
        return
    if not hasattr(anim_data, "action_slot"):
        return
    try:
        if anim_data.action_slot is not None:
            return
        suitable = getattr(anim_data, "action_suitable_slots", None)
        if suitable and len(suitable) > 0:
            anim_data.action_slot = suitable[0]
            return
        if datablock is not None and hasattr(action, "fcurve_ensure_for_datablock"):
            data_path = "location"
            if (hasattr(datablock, "pose")
                    and hasattr(datablock.pose, "bones")
                    and len(datablock.pose.bones) > 0):
                data_path = (
                    f'pose.bones["{datablock.pose.bones[0].name}"].rotation_euler'
                )
            action.fcurve_ensure_for_datablock(datablock, data_path, index=0)
    except Exception:
        pass


def _action_fcurves(action, datablock):
    """Return the fcurve collection for ``action``, traversing the layered
    action API on Blender 5.0+ where ``action.fcurves`` was removed.
    """
    if action is None:
        return None
    if bpy.app.version >= (5, 0, 0):
        if not hasattr(action, "layers") or len(action.layers) == 0:
            # Action has no layers yet — touch one fcurve to create them.
            if datablock is not None and hasattr(action, "fcurve_ensure_for_datablock"):
                try:
                    bone_name = datablock.pose.bones[0].name
                    action.fcurve_ensure_for_datablock(
                        datablock, f'pose.bones["{bone_name}"].rotation_euler', index=0
                    )
                except Exception:
                    return None
        layer = action.layers[0]
        if not hasattr(layer, "strips") or len(layer.strips) == 0:
            return None
        strip = layer.strips[0]
        if not hasattr(strip, "channelbag"):
            return None
        slot = action.slots[0] if (hasattr(action, "slots") and len(action.slots) > 0) else None
        if slot is None:
            return None
        try:
            cbag = strip.channelbag(slot)
            if cbag is not None and hasattr(cbag, "fcurves"):
                return cbag.fcurves
        except Exception:
            return None
        return None
    return getattr(action, "fcurves", None)


def _ensure_fcurve(action, datablock, data_path: str, index: int):
    """Find or create the fcurve for ``data_path``/``index`` on ``action``."""
    if hasattr(action, "fcurve_ensure_for_datablock"):
        try:
            return action.fcurve_ensure_for_datablock(datablock, data_path, index=index)
        except Exception:
            pass
    fcurves = _action_fcurves(action, datablock)
    if fcurves is None:
        return None
    for fc in fcurves:
        if fc.data_path == data_path and fc.array_index == index:
            return fc
    if hasattr(fcurves, "new"):
        try:
            return fcurves.new(data_path, index=index)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Bake — port of ``lib/animation.py:bake_anim`` with two changes:
#   * ``action`` is supplied by the caller instead of being created inline.
#   * ``armature`` is supplied explicitly instead of read from the active
#     object.
# ---------------------------------------------------------------------------

def _bake_control_bones(
    armature,
    *,
    action,
    frame_start: int,
    frame_end: int,
    only_selected: bool,
    ik_data: dict,
) -> int:
    """Sample each (selected) pose bone's local matrix at every integer
    frame in ``[frame_start, frame_end]`` and write rotation/location/
    scale fcurves for it into ``action``. Returns the number of bones
    that contributed at least one keyframe.

    For ``Ctrl_*Pole_*`` bones the pose-space matrix is replaced with a
    geometric pole position derived from the IK chain bones in
    ``ik_data``. Then ``CHILD_OF`` on the pole control is compensated
    (the bake writes pre-CHILD_OF local space; the constraint is then a
    no-op at playback).

    Existing keyframes on the action at the same frames get overwritten
    by this fcurve-foreach_set path; existing keyframes at *other*
    frames are preserved (they're on different fcurves, in the case of
    the deform bones we read from, or different frames on the same
    fcurve if the user had pose keys on a control bone elsewhere).
    """
    scn = bpy.context.scene
    bones_data: list[tuple[int, dict[str, Matrix]]] = []

    def _is_selected(pb) -> bool:
        # Blender 5.0 removed Bone.select; only PoseBone.select exists.
        # Check the pose-bone form first, fall back to data-bone form.
        try:
            if hasattr(pb, "select"):
                return bool(pb.select)
        except Exception:
            pass
        try:
            return bool(pb.bone.select)
        except Exception:
            return False

    def _get_bones_matrix() -> dict[str, Matrix]:
        m: dict[str, Matrix] = {}
        for pb in armature.pose.bones:
            if only_selected and not _is_selected(pb):
                continue

            bmat = pb.matrix

            if pb.name.startswith("Ctrl_ArmPole") or pb.name.startswith("Ctrl_LegPole"):
                # IK pole: replace pose with geometric pole derived from
                # source armature's IK chain.
                src_arm = ik_data.get("src_arm")
                if src_arm is None:
                    continue

                kind = "Leg" if "Leg" in pb.name else ("Arm" if "Arm" in pb.name else "")
                side = pb.name.split("_")[-1]
                if not kind or kind + side not in ik_data:
                    continue

                b1_name, b2_name = ik_data[kind + side]
                b1 = src_arm.pose.bones.get(b1_name)
                b2 = src_arm.pose.bones.get(b2_name)
                if b1 is None or b2 is None:
                    continue

                if kind == "Leg":
                    axis = (b1.z_axis * 0.5) + (b2.z_axis * 0.5)
                else:  # Arm
                    axis = b2.x_axis if side == "Left" else -b2.x_axis

                try:
                    bmat = Matrix.Translation(_get_ik_pole_pos(b1, b2, axis))
                except AttributeError:
                    continue

                # CHILD_OF compensation — the pole control inherits from
                # an IK chain bone via Child Of. The constraint will be
                # re-applied at playback, so we have to pre-divide it
                # out of the matrix we store.
                co = pb.constraints.get("Child Of")
                if co and co.subtarget and co.influence == 1.0 and not co.mute:
                    sb = armature.pose.bones.get(co.subtarget)
                    if sb is not None:
                        bmat = sb.matrix_channel.inverted() @ bmat

            m[pb.name] = armature.convert_space(
                pose_bone=pb, matrix=bmat, from_space="POSE", to_space="LOCAL"
            )
        return m

    saved_frame = scn.frame_current
    for f in range(int(frame_start), int(frame_end) + 1):
        scn.frame_set(f)
        bpy.context.view_layer.update()
        bones_data.append((f, _get_bones_matrix()))

    # Make sure the action is bound to a slot before we add fcurves.
    if armature.animation_data is None:
        armature.animation_data_create()
    if armature.animation_data.action is not action:
        armature.animation_data.action = action
    _ensure_action_slot(armature.animation_data, action, armature)

    baked = 0

    LINEAR = (
        bpy.types.Keyframe.bl_rna.properties["interpolation"]
        .enum_items["LINEAR"].value
    )

    for pb in armature.pose.bones:
        if only_selected and not _is_selected(pb):
            continue

        keyframes: dict[tuple[str, int], list[float]] = {}

        def store(prop: str, idx: int, frame: int, val: float) -> None:
            key = (f'pose.bones["{pb.name}"].{prop}', idx)
            keyframes.setdefault(key, []).extend((frame, val))

        rot_mode = pb.rotation_mode
        euler_prev = None
        quat_prev = None

        for f, mats in bones_data:
            if pb.name not in mats:
                continue
            pb.matrix_basis = mats[pb.name].copy()

            for i, v in enumerate(pb.location):
                store("location", i, f, v)

            if rot_mode == "QUATERNION":
                if quat_prev is not None:
                    q = pb.rotation_quaternion.copy()
                    q.make_compatible(quat_prev)
                    pb.rotation_quaternion = q
                    quat_prev = q
                else:
                    quat_prev = pb.rotation_quaternion.copy()
                for i, v in enumerate(pb.rotation_quaternion):
                    store("rotation_quaternion", i, f, v)
            elif rot_mode == "AXIS_ANGLE":
                for i, v in enumerate(pb.rotation_axis_angle):
                    store("rotation_axis_angle", i, f, v)
            else:  # XYZ Euler etc.
                if euler_prev is not None:
                    e = pb.rotation_euler.copy()
                    e.make_compatible(euler_prev)
                    pb.rotation_euler = e
                    euler_prev = e
                else:
                    euler_prev = pb.rotation_euler.copy()
                for i, v in enumerate(pb.rotation_euler):
                    store("rotation_euler", i, f, v)

            for i, v in enumerate(pb.scale):
                store("scale", i, f, v)

        if not keyframes:
            continue
        baked += 1

        for (data_path, index), pairs in keyframes.items():
            fc = _ensure_fcurve(action, armature, data_path, index)
            if fc is None:
                continue
            n = len(pairs) // 2
            # Preserve any existing keys on this fcurve at frames OUTSIDE
            # the bake range; remove keys INSIDE the range so foreach_set
            # below replaces them cleanly. (Without this, frames we bake
            # would coexist with stale keys at the same frame number,
            # producing flicker.)
            try:
                bake_frames = set(int(pairs[i]) for i in range(0, len(pairs), 2))
                kp = fc.keyframe_points
                for i in range(len(kp) - 1, -1, -1):
                    if int(kp[i].co[0]) in bake_frames:
                        kp.remove(kp[i], fast=True)
            except Exception:
                pass
            fc.keyframe_points.add(n)
            # add() appends — write the new keys onto the tail.
            existing = len(fc.keyframe_points) - n
            for i in range(n):
                fc.keyframe_points[existing + i].co = (pairs[i * 2], pairs[i * 2 + 1])
                fc.keyframe_points[existing + i].interpolation = "LINEAR"
            try:
                fc.update()
            except Exception:
                pass
            try:
                grp = action.groups.get(pb.name) or action.groups.new(pb.name)
                fc.group = grp
            except Exception:
                pass

    scn.frame_set(saved_frame)
    return baked


# ---------------------------------------------------------------------------
# Setup helpers — port of the matrix-collection / helper-bone / retarget-
# constraint logic from ``_import_anim``. These run on a *prepared* source
# armature: a deform-only duplicate of the user's rig with its Copy*/IK
# constraints muted. The caller is responsible for that setup (it's already
# done in ``gltf_to_blender._bake_to_control_rig``).
# ---------------------------------------------------------------------------

def _detect_mixamo_prefix(arm) -> tuple[bool, str]:
    """Return (use_prefix, prefix) — ``("mixamorig:",)`` for typical
    Mixamo-named rigs, ``("",)`` for raw control-rig deform bones (some
    workflows keep the bare ``Hips``/``Spine``/``LeftArm`` names).
    """
    for b in arm.data.bones:
        if b.name.startswith("mixamorig") and ":" in b.name:
            return True, b.name.split(":")[0] + ":"
    return False, ""


def _build_bones_map(
    src_prefix: str,
    *,
    arm_left_kin: str,
    arm_right_kin: str,
    leg_left_kin: str,
    leg_right_kin: str,
) -> dict[str, str]:
    """Source-bone-name → target-control-bone-name. FK chains are driven
    by the equivalent mixamo deform bone (rotation only). IK chains are
    driven by helper bones we add to the source rig at the IK target /
    pole positions (those names equal the target control name; matched
    1:1 below).
    """
    def s(n):  # source name
        return src_prefix + n

    m: dict[str, str] = {}

    # Spine + head
    m[s("Hips")]   = C_PREFIX + "Hips"
    m[s("Spine")]  = C_PREFIX + "Spine"
    m[s("Spine1")] = C_PREFIX + "Spine1"
    m[s("Spine2")] = C_PREFIX + "Spine2"
    m[s("Neck")]   = C_PREFIX + "Neck"
    m[s("Head")]   = C_PREFIX + "Head"
    m[s("LeftShoulder")]  = C_PREFIX + "Shoulder_Left"
    m[s("RightShoulder")] = C_PREFIX + "Shoulder_Right"

    # Arms
    if arm_left_kin == "FK":
        m[s("LeftArm")]     = C_PREFIX + "Arm_FK_Left"
        m[s("LeftForeArm")] = C_PREFIX + "ForeArm_FK_Left"
        m[s("LeftHand")]    = C_PREFIX + "Hand_FK_Left"
    else:
        m[C_PREFIX + "Hand_IK_Left"] = C_PREFIX + "Hand_IK_Left"
    if arm_right_kin == "FK":
        m[s("RightArm")]     = C_PREFIX + "Arm_FK_Right"
        m[s("RightForeArm")] = C_PREFIX + "ForeArm_FK_Right"
        m[s("RightHand")]    = C_PREFIX + "Hand_FK_Right"
    else:
        m[C_PREFIX + "Hand_IK_Right"] = C_PREFIX + "Hand_IK_Right"

    # Fingers
    for side, side_short in (("Left", "Left"), ("Right", "Right")):
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            for j in (1, 2, 3):
                m[s(f"{side}Hand{finger}{j}")] = C_PREFIX + f"{finger}{j}_{side_short}"

    # Legs
    if leg_left_kin == "FK":
        m[s("LeftUpLeg")]   = C_PREFIX + "UpLeg_FK_Left"
        m[s("LeftLeg")]     = C_PREFIX + "Leg_FK_Left"
        m[C_PREFIX + "Foot_FK_Left"] = C_PREFIX + "Foot_FK_Left"
        m[s("LeftToeBase")] = C_PREFIX + "Toe_FK_Left"
    else:
        m[C_PREFIX + "Foot_IK_Left"] = C_PREFIX + "Foot_IK_Left"
        m[s("LeftToeBase")] = C_PREFIX + "Toe_IK_Left"
    if leg_right_kin == "FK":
        m[s("RightUpLeg")]   = C_PREFIX + "UpLeg_FK_Right"
        m[s("RightLeg")]     = C_PREFIX + "Leg_FK_Right"
        m[C_PREFIX + "Foot_FK_Right"] = C_PREFIX + "Foot_FK_Right"
        m[s("RightToeBase")] = C_PREFIX + "Toe_FK_Right"
    else:
        m[C_PREFIX + "Foot_IK_Right"] = C_PREFIX + "Foot_IK_Right"
        m[s("RightToeBase")] = C_PREFIX + "Toe_IK_Right"

    return m


def _select_only(obj) -> None:
    """Make ``obj`` the only selected + active object.

    Uses direct API rather than ``bpy.ops.object.select_all(...)`` so the
    call works from contexts without a 3D-View area (modal timers,
    background threads). ``mode_set`` etc. don't need an area; the
    select operator does.
    """
    for o in bpy.context.view_layer.objects:
        try:
            o.select_set(False)
        except Exception:
            pass
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def apply_anim_to_control_rig(
    src_arm,
    tar_arm,
    *,
    action,
    frame_start: int,
    frame_end: int,
) -> int:
    """Bake the per-frame deform pose of ``src_arm`` onto the control
    bones of ``tar_arm`` and write the result into ``action``.

    ``src_arm`` is expected to be a *prepared* source: deform-only
    skeleton with mixamorig (or unprefixed) bone names and an animation
    that resolves on its own pose bones — i.e. already-baked keyframes,
    or a constraint chain that doesn't depend on the bones we strip.
    The caller (``gltf_to_blender._bake_to_control_rig``) builds
    this from a duplicate of ``tar_arm`` with control bones removed and
    the deform bones' Copy*/IK constraints muted.

    ``action`` is the action that receives the new control-bone fcurves.
    Existing keyframes on it are preserved except for those at frames
    inside ``[frame_start, frame_end]`` on the same fcurves we're
    overwriting (see ``_bake_control_bones``).

    Returns the number of control bones that received keyframes.
    """
    use_prefix, prefix = _detect_mixamo_prefix(src_arm)

    # Mark source for cleanup (callers also tag with "proscenium_temp_source",
    # but the cleanup loop tolerates either marker).
    src_arm["mix_to_del"] = True

    def s(n):  # source bone name
        return prefix + n if use_prefix else n

    # --- IK/FK switch state from existing control bone properties ---
    def _ik_state(bone_name: str) -> str:
        pb = tar_arm.pose.bones.get(bone_name)
        if pb is None:
            return "FK"
        try:
            return "IK" if pb["ik_fk_switch"] < 0.5 else "FK"
        except (KeyError, TypeError):
            return "FK"

    arm_left_kin  = _ik_state(C_PREFIX + ARM_NAMES["hand_ik"] + "_Left")
    arm_right_kin = _ik_state(C_PREFIX + ARM_NAMES["hand_ik"] + "_Right")
    leg_left_kin  = _ik_state(C_PREFIX + LEG_NAMES["foot_ik"] + "_Left")
    leg_right_kin = _ik_state(C_PREFIX + LEG_NAMES["foot_ik"] + "_Right")

    bones_map = _build_bones_map(
        prefix if use_prefix else "",
        arm_left_kin=arm_left_kin,
        arm_right_kin=arm_right_kin,
        leg_left_kin=leg_left_kin,
        leg_right_kin=leg_right_kin,
    )

    # --- Collect target rest-pose data (helper-bone matrices, IK chains) ---
    _select_only(tar_arm)
    bpy.ops.object.mode_set(mode="EDIT")

    ctrl_matrices: dict[str, tuple[Matrix, str]] = {}
    ik_bones_data: dict[str, tuple[str, str, dict[str, tuple]]] = {}

    kinematics = {
        "HandLeft":  ("Hand", arm_left_kin,  "Left"),
        "HandRight": ("Hand", arm_right_kin, "Right"),
        "FootLeft":  ("Foot", leg_left_kin,  "Left"),
        "FootRight": ("Foot", leg_right_kin, "Right"),
    }
    for slot_id, (kind, kin, side) in kinematics.items():
        ctrl_name = C_PREFIX + kind + "_" + kin + "_" + side
        ctrl_eb = tar_arm.data.edit_bones.get(ctrl_name)
        if ctrl_eb is None:
            continue
        mix_bone_name = s(side + kind)  # e.g. mixamorig:LeftHand
        ctrl_matrices[ctrl_name] = (ctrl_eb.matrix.copy(), mix_bone_name)

        if kin == "IK":
            chain_names = (["UpLeg_IK_" + side, "Leg_IK_" + side]
                           if kind == "Foot"
                           else ["Arm_IK_" + side, "ForeArm_IK_" + side])
            ik1 = tar_arm.data.edit_bones.get(chain_names[0])
            ik2 = tar_arm.data.edit_bones.get(chain_names[1])
            if ik1 is None or ik2 is None:
                continue
            ik_bones_data[slot_id] = (
                kind,
                side,
                {
                    "ik1": (ik1.name, ik1.head.copy(), ik1.tail.copy(), ik1.roll),
                    "ik2": (ik2.name, ik2.head.copy(), ik2.tail.copy(), ik2.roll),
                },
            )

    # --- Source: apply transforms (rotation+scale) and rescale location curves ---
    bpy.ops.object.mode_set(mode="OBJECT")
    _select_only(src_arm)
    bpy.context.view_layer.update()

    scale_fac = src_arm.scale[0]
    try:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        bpy.context.evaluated_depsgraph_get().update()
    except Exception:
        pass

    if scale_fac != 1.0 and src_arm.animation_data and src_arm.animation_data.action:
        src_action = src_arm.animation_data.action
        src_fcurves = _action_fcurves(src_action, src_arm)
        if src_fcurves is not None:
            for fc in src_fcurves:
                dp = fc.data_path
                if dp.startswith("pose.bones") and dp.endswith(".location"):
                    for k in fc.keyframe_points:
                        k.co[1] *= scale_fac

    # --- Add helper bones on source (IK target shadows + IK chain shadows) ---
    bpy.ops.object.mode_set(mode="EDIT")
    eb = src_arm.data.edit_bones
    for ctrl_name, (mat, parent_name) in ctrl_matrices.items():
        helper = eb.new(ctrl_name)
        helper.head = Vector((0.0, 0.0, 0.0))
        helper.tail = Vector((0.0, 0.0, 0.1))
        helper.matrix = mat
        parent = eb.get(parent_name)
        if parent is not None:
            helper.parent = parent

    for slot_id, (kind, side, ikb) in ik_bones_data.items():
        for key in ("ik1", "ik2"):
            bname, bhead, btail, broll = ikb[key]
            if bname in eb:
                continue
            helper = eb.new(bname)
            helper.head = bhead
            helper.tail = btail
            helper.roll = broll

    # --- Add COPY_TRANSFORMS on IK helpers so they follow the deform chain ---
    bpy.ops.object.mode_set(mode="POSE")
    bake_ik_data: dict = {"src_arm": src_arm}

    for slot_id, (kind, side, ikb) in ik_bones_data.items():
        b1_name = ikb["ik1"][0]
        b2_name = ikb["ik2"][0]
        b1_pb = src_arm.pose.bones.get(b1_name)
        b2_pb = src_arm.pose.bones.get(b2_name)
        if b1_pb is None or b2_pb is None:
            continue

        if kind == "Foot":
            chain = (s(side + "UpLeg"), s(side + "Leg"))
            bake_ik_data["Leg" + side] = chain
        else:  # Hand
            chain = (s(side + "Arm"), s(side + "ForeArm"))
            bake_ik_data["Arm" + side] = chain

        for pb, sub in ((b1_pb, chain[0]), (b2_pb, chain[1])):
            cns = pb.constraints.new("COPY_TRANSFORMS")
            cns.name = "Copy Transforms"
            cns.target = src_arm
            cns.subtarget = sub

    # --- Add retarget constraints on target's control bones ---
    _select_only(tar_arm)
    bpy.ops.object.mode_set(mode="POSE")
    # Direct deselect instead of ``bpy.ops.pose.select_all`` so we don't
    # need a 3D-View area in the context (modal/timer paths lack one).
    for pb in tar_arm.pose.bones:
        try:
            pb.bone.select = False
        except Exception:
            pass
        try:
            pb.select = False
        except Exception:
            pass
    bpy.context.view_layer.update()

    for src_name, tar_name in bones_map.items():
        src_pb = src_arm.pose.bones.get(src_name)
        tar_pb = tar_arm.pose.bones.get(tar_name)
        if src_pb is None or tar_pb is None:
            continue

        cns = tar_pb.constraints.new("COPY_ROTATION")
        cns.name = "Copy Rotation_retarget"
        cns.target = src_arm
        cns.subtarget = src_name

        if "Hips" in src_name:
            cns = tar_pb.constraints.new("COPY_LOCATION")
            cns.name = "Copy Location_retarget"
            cns.target = src_arm
            cns.subtarget = src_name
            cns.owner_space = cns.target_space = "LOCAL"

        is_ik_target = (
            (leg_left_kin  == "IK" and "Foot_IK_Left"  in src_name)
            or (leg_right_kin == "IK" and "Foot_IK_Right" in src_name)
            or (arm_left_kin  == "IK" and "Hand_IK_Left"  in src_name)
            or (arm_right_kin == "IK" and "Hand_IK_Right" in src_name)
        )
        if is_ik_target:
            cns = tar_pb.constraints.new("COPY_LOCATION")
            cns.name = "Copy Location_retarget"
            cns.target = src_arm
            cns.subtarget = src_name
            cns.target_space = cns.owner_space = "POSE"

            side_suffix = "_Left" if "Left" in src_name else "_Right"
            pole_kind = ARM_NAMES["pole_ik"] if "Hand" in src_name else LEG_NAMES["pole_ik"]
            pole_name = C_PREFIX + pole_kind + side_suffix
            pole_pb = tar_arm.pose.bones.get(pole_name)
            if pole_pb is not None:
                tar_arm.data.bones.active = pole_pb.bone
                try:
                    pole_pb.bone.select = True
                except Exception:
                    pass
                try:
                    pole_pb.select = True
                except Exception:
                    pass

        tar_arm.data.bones.active = tar_pb.bone
        try:
            tar_pb.bone.select = True
        except Exception:
            pass
        try:
            tar_pb.select = True
        except Exception:
            pass

    bpy.context.view_layer.update()

    # --- Bake into the supplied action ---
    baked = _bake_control_bones(
        tar_arm,
        action=action,
        frame_start=frame_start,
        frame_end=frame_end,
        only_selected=True,
        ik_data=bake_ik_data,
    )

    # --- Tear down retarget constraints ---
    for tar_name in set(bones_map.values()):
        pb = tar_arm.pose.bones.get(tar_name)
        if pb is None:
            continue
        for c in list(pb.constraints):
            if c.name.endswith("_retarget"):
                pb.constraints.remove(c)
    for slot_id in ik_bones_data:
        side_suffix = "_Left" if "Left" in slot_id else "_Right"
        pole_kind = ARM_NAMES["pole_ik"] if "Hand" in slot_id else LEG_NAMES["pole_ik"]
        # No retarget constraint on poles, but they were left selected.
        pole_pb = tar_arm.pose.bones.get(C_PREFIX + pole_kind + side_suffix)
        if pole_pb is not None:
            try:
                pole_pb.bone.select = False
            except Exception:
                pass

    return baked
