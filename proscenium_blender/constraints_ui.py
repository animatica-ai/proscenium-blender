"""Constraint authoring with native Blender objects.

The MMCP protocol has three constraint primitives. Each one is exposed in the
addon as a Blender object the artist can see and edit in the viewport:

  root_path        →  a Bezier curve named ``Proscenium_RootPath_*`` with
                      a custom prop ``proscenium_is_root_path = True``.
  effector_target  →  an Empty with ``proscenium_target_joint = "<name>"``,
                      location-keyframed across the timeline.
  pose_keyframe    →  derived from the target armature's existing keyframes
                      in its source action; no dedicated object.

This module owns:
  * The operators that create / focus / remove the constraint objects.
  * A scene-walker that returns the lists of root-path curves and effector
    empties active in the current scene.
  * Sampling helpers that turn each Blender object into the MMCP constraint
    dict shape (``{"type": "...", "frames": [...], ...}``).

The actual assembly of the request body lives in ``request_builder.py`` —
this module is purely UI + sampling.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator
from mathutils import Matrix, Vector
from mathutils.geometry import interpolate_bezier

from . import constants, coords


# MMCP world (right-handed Y-up) → Blender world (right-handed Z-up). Used
# to conjugate Blender-frame rotation matrices back into MMCP frame when
# sampling pose keyframes for the request.
_MMCP_TO_BLENDER = Matrix((
    (1.0, 0.0,  0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0,  0.0),
))


# Bones Kimodo's EndEffector constraint can pin in "generate" (fill_mode)
# mode: the 5 EE chains plus the root. When ``sample_pose_keyframes``
# detects a keyed bone outside this set (spine, neck, arms mid-chain, etc.),
# we switch fill_mode to "rest" so the server pins every joint position via
# FullBodyConstraintSet — otherwise those off-chain keyframes would be
# silently dropped.
_EE_CHAIN_BONES = frozenset({
    "Hips",
    "LeftFoot", "LeftToeBase",
    "RightFoot", "RightToeBase",
    "LeftHand", "LeftHandMiddleEnd",
    "RightHand", "RightHandMiddleEnd",
})


# ---------------------------------------------------------------------------
# Action API compatibility
# ---------------------------------------------------------------------------

def iter_action_fcurves(action):
    """Yield every F-curve in an Action, regardless of Blender version.

    Pre-Blender 4.4: ``action.fcurves`` is a flat collection.
    Blender 4.4+ moved to a layered Action with slots/strips/channelbags;
    the flat ``.fcurves`` attribute was removed in 5.x.
    """
    if action is None:
        return
    flat = getattr(action, "fcurves", None)
    if flat is not None:
        yield from flat
        return
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            slots = getattr(action, "slots", ())
            for slot in slots:
                cb = strip.channelbag(slot, ensure=False) if hasattr(strip, "channelbag") else None
                if cb is None:
                    continue
                yield from cb.fcurves


# ---------------------------------------------------------------------------
# Scene walker
# ---------------------------------------------------------------------------

def walk_scene_constraints(scene: bpy.types.Scene) -> dict[str, list[bpy.types.Object]]:
    """Find every Blender object in the scene that the addon should treat as
    an MMCP constraint. Returns a dict keyed by primitive type."""
    root_paths: list[bpy.types.Object] = []
    effectors:  list[bpy.types.Object] = []
    for obj in scene.objects:
        if obj.get(constants.PROP_IS_ROOT_PATH) and obj.type == 'CURVE':
            root_paths.append(obj)
        elif obj.get(constants.PROP_TARGET_JOINT) and obj.type == 'EMPTY':
            effectors.append(obj)
    return {"root_paths": root_paths, "effector_targets": effectors}


# ---------------------------------------------------------------------------
# Operators — Add / Remove / Focus
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_add_root_path(Operator):
    bl_idname = "proscenium.add_root_path"
    bl_label = "Add Root Path"
    bl_description = (
        "Create a Bezier curve on the floor plane that the character will follow. "
        "Sample density and 'follow direction' (heading) are editable on the curve "
        "object's properties"
    )
    # Note: no ``UNDO`` flag. With UNDO enabled on a slotted-action scene,
    # Blender's undo snapshot diff was observed to drop unreferenced fcurves
    # — including the root bone's vertical-axis keyframes — after the op
    # completed. The bake itself never writes that axis, so this was purely
    # an undo-system artifact; dropping UNDO keeps the channel intact.
    bl_options = {'REGISTER'}

    match_direction: bpy.props.BoolProperty(
        name="Follow Direction",
        description="Derive heading_radians from the curve tangent so the character faces along the path",
        default=True,
    )
    sample_density: bpy.props.IntProperty(
        name="Sample Every N Frames",
        description="One root_path constraint frame per N timeline frames",
        default=10, min=1, max=60,
    )

    def execute(self, context):
        scene = context.scene

        # Seed the curve from root-bone location keyframes when available —
        # one spline control point per existing keyframe, preserving the
        # user's authored timing exactly. Fall back to a default 2 m line
        # (densified to a handful of handles) only when the armature has no
        # root-location keys at all.
        keyframe_points = _root_keyframe_points(scene)
        if keyframe_points:
            control_points = keyframe_points
        else:
            control_points = _densify_points(
                _default_root_points(),
                min_count=max(4, constants.ROOT_PATH_CONTROL_POINTS),
            )

        # 2D curve: Blender stores only the XY plane (floor). That matches
        # MMCP's root_path primitive, which is a smooth_root xz trajectory —
        # the protocol has no notion of vertical root motion.
        curve_data = bpy.data.curves.new("Proscenium_RootPath", type='CURVE')
        curve_data.dimensions = '2D'
        curve_data.resolution_u = 12
        spline = curve_data.splines.new('BEZIER')
        spline.bezier_points.add(len(control_points) - 1)
        for pt, pos in zip(spline.bezier_points, control_points):
            pt.co                = pos
            pt.handle_left_type  = 'AUTO'
            pt.handle_right_type = 'AUTO'

        existing = sum(1 for o in scene.objects if o.get(constants.PROP_IS_ROOT_PATH))
        obj_name = f"Proscenium_RootPath_{existing + 1:02d}"
        obj = bpy.data.objects.new(obj_name, curve_data)
        obj[constants.PROP_IS_ROOT_PATH]    = True
        obj[constants.PROP_MATCH_DIRECTION] = self.match_direction
        obj[constants.PROP_SAMPLE_DENSITY]  = self.sample_density
        scene.collection.objects.link(obj)

        _select_only(context, obj)
        self.report({'INFO'}, f"Added {obj_name} ({len(control_points)} control points)")
        return {'FINISHED'}


def _default_root_points() -> list[Vector]:
    """Fallback seed: a 2 m line along +X on the floor."""
    n = max(2, constants.ROOT_PATH_CONTROL_POINTS)
    return [Vector((i / (n - 1) * 2.0, 0.0, 0.0)) for i in range(n)]


def _densify_points(points: list[Vector], *, min_count: int) -> list[Vector]:
    """Return ``points`` augmented with linearly-interpolated midpoints so
    the total is at least ``min_count``. Preserves every original point —
    midpoints get inserted into the currently-longest segment, one at a
    time, until the count is reached.
    """
    pts = list(points)
    while len(pts) < min_count and len(pts) >= 2:
        longest_i, longest_len = 0, -1.0
        for i in range(len(pts) - 1):
            d = (pts[i + 1] - pts[i]).length
            if d > longest_len:
                longest_len = d
                longest_i = i
        mid = (pts[longest_i] + pts[longest_i + 1]) * 0.5
        pts.insert(longest_i + 1, mid)
    return pts


def _root_keyframe_points(scene: bpy.types.Scene) -> list[Vector]:
    """World xz of the root bone at each root-location keyframe, projected
    onto the ground plane (z=0). Returns an empty list if the target armature
    has no root location keys in the scene range — the operator then falls
    back to a default line.
    """
    settings = scene.proscenium
    arm = settings.target_armature
    if arm is None or arm.type != 'ARMATURE':
        return []
    action = arm.animation_data.action if arm.animation_data else None
    if action is None:
        return []

    root_pb = next((pb for pb in arm.pose.bones if pb.parent is None), None)
    if root_pb is None:
        return []

    frame_range = (int(scene.frame_start), int(scene.frame_end))
    times: set[int] = set()
    for fc in iter_action_fcurves(action):
        if "location" not in fc.data_path:
            continue
        if _bone_name_from_data_path(fc.data_path) != root_pb.name:
            continue
        for kp in fc.keyframe_points:
            f = int(round(kp.co.x))
            if frame_range[0] <= f <= frame_range[1]:
                times.add(f)
    if len(times) < 2:
        return []

    saved = scene.frame_current
    try:
        points: list[Vector] = []
        for f in sorted(times):
            scene.frame_set(f)
            world = (arm.matrix_world @ root_pb.matrix).translation
            points.append(Vector((world.x, world.y, 0.0)))
        return points
    finally:
        scene.frame_set(saved)


def _joint_items_callback(self, context):
    """Joint names from the target armature, for the effector-target popup."""
    settings = context.scene.proscenium
    arm = settings.target_armature
    if arm is None or arm.type != 'ARMATURE':
        return [("", "(set a target armature first)", "")]
    items = []
    for pb in arm.pose.bones:
        items.append((pb.name, pb.name, ""))
    return items or [("", "(armature has no pose bones)", "")]


class PROSCENIUM_OT_add_effector_target(Operator):
    bl_idname = "proscenium.add_effector_target"
    bl_label = "Add Effector Pin"
    bl_description = (
        "Pin a named joint to a moving Blender Empty. Each location keyframe "
        "on the empty becomes one effector_target constraint frame"
    )
    bl_options = {'REGISTER', 'UNDO'}

    joint: EnumProperty(
        name="Joint",
        items=_joint_items_callback,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=240)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "joint")

    def execute(self, context):
        if not self.joint:
            self.report({'ERROR'}, "Pick a joint")
            return {'CANCELLED'}

        settings = context.scene.proscenium
        arm = settings.target_armature
        if arm is None:
            self.report({'ERROR'}, "Set a target armature first")
            return {'CANCELLED'}

        bone = arm.pose.bones.get(self.joint)
        # bone.head is in armature-local space. Empties have no parent, so
        # empty.location is world-space — transform through the armature's
        # world matrix to keep the pin at the bone's actual viewport position
        # (otherwise a translated/rotated armature puts the empty at the wrong
        # spot, which looks like a "rotation" offset).
        if bone is not None:
            start_pos = arm.matrix_world @ bone.head
        else:
            start_pos = Vector((0.0, 1.0, 0.0))

        empty = bpy.data.objects.new(f"Proscenium_{self.joint}_Target", None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = constants.EFFECTOR_EMPTY_SIZE
        empty.location           = start_pos

        empty[constants.PROP_TARGET_JOINT] = self.joint
        # Drop a colour swatch on Object so the user can spot it in viewport.
        if self.joint in constants.EFFECTOR_COLORS:
            empty.color = constants.EFFECTOR_COLORS[self.joint]
            empty.show_name = True

        context.scene.collection.objects.link(empty)
        # Seed with one keyframe at the current frame so it's a valid constraint.
        empty.keyframe_insert(data_path="location", frame=context.scene.frame_current)

        _select_only(context, empty)
        self.report({'INFO'}, f"Added effector pin for {self.joint}")
        return {'FINISHED'}


class PROSCENIUM_OT_remove_constraint_object(Operator):
    bl_idname = "proscenium.remove_constraint_object"
    bl_label = "Remove Constraint Object"
    bl_description = "Delete the named constraint object from the scene"
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj is None:
            return {'CANCELLED'}
        bpy.data.objects.remove(obj, do_unlink=True)
        for area in context.screen.areas:
            area.tag_redraw()
        return {'FINISHED'}


class PROSCENIUM_OT_focus_constraint_object(Operator):
    bl_idname = "proscenium.focus_constraint_object"
    bl_label = "Focus Constraint Object"
    bl_description = "Select and view-frame the named constraint object"
    bl_options = {'REGISTER', 'UNDO'}

    name: bpy.props.StringProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(self.name)
        if obj is None:
            return {'CANCELLED'}
        _select_only(context, obj)
        return {'FINISHED'}


def _select_only(context, obj: bpy.types.Object) -> None:
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


# ---------------------------------------------------------------------------
# Sampling — Blender object → MMCP constraint dict
# ---------------------------------------------------------------------------

def sample_root_path(
    curve_obj: bpy.types.Object,
    *,
    total_frames: int,
) -> dict[str, Any] | None:
    """Sample a Bezier curve into a `root_path` constraint dict.

    The curve is treated as a static spatial trajectory; the character walks
    along it over the request's timeline. We sample at evenly-spaced frame
    indices in the request frame range. Returns ``None`` if the curve has no
    spline data (corrupt) or fewer than 2 points.
    """
    if total_frames < 2:
        return None
    spline = curve_obj.data.splines[0] if curve_obj.data.splines else None
    if spline is None or len(spline.bezier_points) < 2:
        return None

    local_polyline = _bezier_to_polyline(spline, segments_per_segment=12)
    if len(local_polyline) < 2:
        return None

    # Transform the sampled points through the curve object's world matrix so
    # moving the curve in the viewport moves the path the character follows.
    mw = curve_obj.matrix_world
    polyline = [mw @ p for p in local_polyline]

    density = max(1, int(curve_obj.get(constants.PROP_SAMPLE_DENSITY, 10)))
    frames = list(range(0, total_frames, density))
    if frames[-1] != total_frames - 1:
        frames.append(total_frames - 1)

    positions_xz: list[tuple[float, float]] = []
    headings:     list[float] = []

    last_idx = len(polyline) - 1
    for f in frames:
        t = f / (total_frames - 1)               # 0..1 along the curve
        idx = max(0, min(last_idx, int(round(t * last_idx))))
        world = polyline[idx]
        x_mmcp, _, z_mmcp = coords.blender_pos_to_mmcp(world)
        positions_xz.append((x_mmcp, z_mmcp))

        if curve_obj.get(constants.PROP_MATCH_DIRECTION):
            tangent = _tangent_at(polyline, idx)
            tx, _, tz = coords.blender_pos_to_mmcp(tangent)
            # heading = 0 faces MMCP +Z; positive rotation about +Y takes +Z toward -X.
            headings.append(math.atan2(-tx, tz))

    constraint: dict[str, Any] = {
        "type": "root_path",
        "frames": frames,
        "positions_xz": [list(p) for p in positions_xz],
    }
    if headings:
        constraint["heading_radians"] = headings
    return constraint


def sample_effector_target(
    empty_obj: bpy.types.Object,
    *,
    frame_range: tuple[int, int],
    total_frames: int,
) -> dict[str, Any] | None:
    """Convert an effector-target empty into an `effector_target` constraint.

    Walks the empty's location fcurve keyframes within ``frame_range`` and
    emits one constraint frame per keyframe. Frame indices are translated
    from Blender frames to the request's timeline (0..total_frames-1).
    Returns ``None`` if the empty has no keyframes in range.
    """
    joint = (empty_obj.get(constants.PROP_TARGET_JOINT) or "").strip()
    if not joint:
        return None

    frames_in_range: set[int] = set()
    if empty_obj.animation_data and empty_obj.animation_data.action:
        for fc in iter_action_fcurves(empty_obj.animation_data.action):
            if fc.data_path != "location":
                continue
            for kp in fc.keyframe_points:
                f = int(round(kp.co.x))
                if frame_range[0] <= f <= frame_range[1]:
                    frames_in_range.add(f)

    if not frames_in_range:
        # No keyframes in range — fall back to the current static location at
        # a single frame (frame_range start) so the user gets a usable pin
        # without having to keyframe.
        frames_in_range.add(frame_range[0])

    sorted_frames = sorted(frames_in_range)
    positions: list[list[float]] = []

    scene = bpy.context.scene
    original = scene.frame_current
    try:
        for f in sorted_frames:
            scene.frame_set(f)
            world = empty_obj.matrix_world.translation
            mx, my, mz = coords.blender_pos_to_mmcp(world)
            positions.append([mx, my, mz])
    finally:
        scene.frame_set(original)

    return {
        "type": "effector_target",
        "joint": joint,
        "frames": [_to_timeline_frame(f, frame_range) for f in sorted_frames],
        "positions": positions,
    }


def _evaluated_local_basis(pb: bpy.types.PoseBone) -> 'Matrix':
    """The bone-local matrix_basis equivalent computed from the *evaluated*
    pose-bone matrix, so constraint-driven bones (control-rig setups) report
    a non-identity basis even though they were never directly keyed.

    For a freely-keyed bone this returns exactly ``pb.matrix_basis``; for a
    bone driven by a Copy*/IK constraint it returns the basis that, applied
    against the bone's rest, would reproduce ``pb.matrix``.
    """
    if pb.parent is None:
        return pb.bone.matrix_local.inverted() @ pb.matrix
    parent_offset = pb.bone.matrix_local.inverted() @ pb.parent.bone.matrix_local
    return parent_offset @ pb.parent.matrix.inverted() @ pb.matrix


def sample_pose_keyframes(
    armature_obj: bpy.types.Object,
    *,
    source_action: bpy.types.Action,
    frame_range: tuple[int, int],
) -> list[dict[str, Any]]:
    """Walk ``source_action`` for frames with rotation keyframes and emit one
    pose_keyframe constraint per such frame, capturing every pose bone's
    current rotation at that frame.

    Control-rig aware: when the armature is detected as a control rig
    (deform bones driven by Copy*/IK constraints), keyframes are collected
    from any bone (the user typically keys *control* bones), and rotations
    are sampled on the *deform* bones via their evaluated pose. That way
    posing a Mixamo/Rigify/ARP control rig produces the same MMCP request
    as if the user had keyed the deform skeleton directly.

    The armature's current action is temporarily set to ``source_action``
    so the evaluation matches what the user authored.
    """
    if armature_obj is None or source_action is None:
        return []

    # Lazy-import to avoid request_builder ↔ constraints_ui circular at
    # module load — both modules are pulled into __init__'s register flow.
    from . import request_builder

    root_pb = next(
        (pb for pb in armature_obj.pose.bones if pb.parent is None),
        None,
    )

    deform_bones = request_builder.detect_deform_bones(armature_obj)
    is_control_rig = request_builder.is_control_rig(armature_obj)

    # pose_keyframes describe a BODY POSE. If no bone has rotation keyframes,
    # there's no pose to pin — just root motion — and the root_path
    # constraint (from the path curve) handles that channel. A rig whose
    # only keyframes are root-bone location keys (the path_follow sync case)
    # therefore emits zero pose_keyframes; the root location flows through
    # root_path instead.
    interesting_frames: set[int] = set()
    keyed_bones: set[str] = set()
    for fc in iter_action_fcurves(source_action):
        bone_name = _bone_name_from_data_path(fc.data_path)
        if bone_name is None:
            continue
        if "rotation" not in fc.data_path:
            continue
        keyed_bones.add(bone_name)
        for kp in fc.keyframe_points:
            f = int(round(kp.co.x))
            if frame_range[0] <= f <= frame_range[1]:
                interesting_frames.add(f)
    if not interesting_frames or not keyed_bones:
        return []

    # In control-rig mode the user keys controls; the deform bones (what
    # we serialize) get their pose from the constraint stack. Sample those.
    # In direct mode the user keys the deform bones themselves; sample the
    # bones that actually have rotation keyframes.
    sample_bone_names = deform_bones if is_control_rig else keyed_bones

    # Temporarily attach the source action so frame evaluation hits the
    # user's authored poses.
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    saved_action = armature_obj.animation_data.action
    armature_obj.animation_data.action = source_action

    scene = bpy.context.scene
    original = scene.frame_current

    # Per-bone rotation delta, converted to MMCP world frame in three steps:
    #   1. ``ML @ R_basis @ ML.T``   — bone-local-rest → armature-local
    #   2. ``mw_rot @ _ @ mw_rot.T`` — armature-local → Blender world
    #                                  (no-op when armature is at identity;
    #                                   required for rigs like Mixamo that
    #                                   carry a 90° + 0.01 matrix_world).
    #   3. ``S_inv @ _ @ S``         — Blender Z-up → MMCP Y-up
    # The inverse of this chain lives in ``gltf_to_blender.py`` on the bake
    # path and must stay in sync.
    S        = _MMCP_TO_BLENDER
    S_inv    = S.transposed()
    mw_rot   = armature_obj.matrix_world.to_quaternion().to_matrix()
    mw_rot_t = mw_rot.transposed()

    # Heuristic: if the user keyed bones that live outside Kimodo's
    # end-effector chains (spine, neck, shoulders, arms, etc.), they clearly
    # want those poses pinned. Control-rig setups always go through full
    # body — there's no easy mapping from keyed control bones to canonical
    # EE groups, and the user's intent is to author the whole pose.
    fill_mode = (
        "rest"
        if (is_control_rig or any(b not in _EE_CHAIN_BONES for b in keyed_bones))
        else "generate"
    )

    constraints: list[dict[str, Any]] = []
    try:
        for f in sorted(interesting_frames):
            scene.frame_set(f)
            # Force depsgraph re-evaluation — ``scene.frame_set`` alone
            # doesn't always propagate through drivers / constraint stacks /
            # NLA, and the evaluated matrices then read stale values from
            # the previous frame.
            bpy.context.view_layer.update()
            joint_rotations: dict[str, list[float]] = {}
            for pb in armature_obj.pose.bones:
                if pb.name not in sample_bone_names:
                    continue
                R_basis = _evaluated_local_basis(pb).to_3x3()
                ML      = pb.bone.matrix_local.to_3x3()
                R_blender_arm = ML @ R_basis @ ML.transposed()
                R_blender_world = mw_rot @ R_blender_arm @ mw_rot_t
                R_mmcp = S_inv @ R_blender_world @ S
                w, x, y, z = R_mmcp.to_quaternion()
                joint_rotations[pb.name] = [x, y, z, w]

            if not joint_rotations:
                continue

            entry: dict[str, Any] = {
                "type":            "pose_keyframe",
                "frame":           _to_timeline_frame(f, frame_range),
                "joint_rotations": joint_rotations,
                "fill_mode":       fill_mode,
            }
            # Always send the root's **world**-space position so the server
            # has per-frame root motion to retarget. Without this, every
            # pose_keyframe defaults to the canonical standing root on the
            # server and the generated character's root never moves —
            # observable as "body animates but root stays nailed at MMCP
            # origin." Uses the full ``armature.matrix_world @ pose_bone.matrix``
            # chain so this captures both:
            #   * the armature object's world transform (user drags the rig
            #     around in object mode),
            #   * the root bone's pose matrix (user keyframes location in
            #     pose mode).
            #
            # When the rig is a control rig, the *deform* root bone is what
            # the server will see — pull its evaluated head world position,
            # not the armature's outer root pose bone.
            sample_root = root_pb
            if is_control_rig and root_pb is not None and root_pb.name not in deform_bones:
                # Find the topmost deform bone; that's the deform skeleton's root.
                for pb in armature_obj.pose.bones:
                    if pb.name in deform_bones:
                        cur = pb
                        while cur.parent is not None and cur.parent.name in deform_bones:
                            cur = cur.parent
                        sample_root = cur
                        break
            if sample_root is not None:
                root_world = (armature_obj.matrix_world @ sample_root.matrix).translation
                root_mmcp = coords.blender_pos_to_mmcp(root_world)
                entry["root_position"] = list(root_mmcp)
            constraints.append(entry)
    finally:
        scene.frame_set(original)
        if saved_action is not None:
            armature_obj.animation_data.action = saved_action

    return constraints


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bezier_to_polyline(spline, *, segments_per_segment: int = 12) -> list[Vector]:
    """Sample a single Bezier spline into a flat list of world-space points."""
    pts = spline.bezier_points
    polyline: list[Vector] = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg = interpolate_bezier(
            a.co, a.handle_right, b.handle_left, b.co, segments_per_segment + 1
        )
        if i > 0:
            seg = seg[1:]   # avoid duplicating segment endpoints
        polyline.extend(seg)
    return polyline


def _tangent_at(polyline: list[Vector], idx: int) -> Vector:
    n = len(polyline)
    if n < 2:
        return Vector((0.0, 1.0, 0.0))
    if idx <= 0:
        v = polyline[1] - polyline[0]
    elif idx >= n - 1:
        v = polyline[-1] - polyline[-2]
    else:
        v = polyline[idx + 1] - polyline[idx - 1]
    return v.normalized() if v.length else Vector((0.0, 1.0, 0.0))


def _to_timeline_frame(blender_frame: int, frame_range: tuple[int, int]) -> int:
    return max(0, blender_frame - frame_range[0])


_BONE_DATA_PATH_RE = re.compile(r'pose\.bones\["([^"]+)"\]')


def _bone_name_from_data_path(data_path: str) -> str | None:
    """Return the bone name referenced by a pose-bone fcurve data path.

    Typical shapes: ``pose.bones["Hips"].rotation_quaternion``,
    ``pose.bones["LeftArm"].rotation_euler``.
    """
    m = _BONE_DATA_PATH_RE.search(data_path)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    PROSCENIUM_OT_add_root_path,
    PROSCENIUM_OT_add_effector_target,
    PROSCENIUM_OT_remove_constraint_object,
    PROSCENIUM_OT_focus_constraint_object,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
