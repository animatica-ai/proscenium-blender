"""Root-path curve ↔ root-bone location keyframes.

Mental model: each Bezier control point on a ``proscenium_is_root_path``
curve is one keyframe on the target armature's **root pose bone**
``location`` fcurve. Add a control point → new keyframe. Drag a
control point → the keyframe's value updates. Delete one → the
keyframe is gone.

A persistent ``depsgraph_update_post`` handler watches for changes to
the curve (point count, point positions, or the curve object's world
transform) and re-bakes the root bone's location fcurve. Frames are
distributed evenly across the scene's active range, so N control points
→ N evenly spaced keyframes between ``frame_start`` and ``frame_end``.

We keyframe the **root pose bone** (not the armature object) because:
  * It's the canonical "root motion" channel in Blender — compatible
    with existing authoring workflow and NLA stacks.
  * The armature object's location may already be non-identity (Mixamo
    rigs ship at (0, −3, 0) and scale 0.01); we don't want to fight it.
  * ``sample_pose_keyframes`` on the outbound path reads the root bone's
    final world head via ``armature.matrix_world @ root_pb.matrix``,
    which picks these keyframes up automatically.

For each control point in world space, the per-frame bone-local
``location`` is computed so the root bone's world head lands *exactly*
on the control point:

    M = armature.matrix_world @ root_pb.bone.matrix_local   # 4x4
    root_pb.location = M.inverted() @ control_point_world
"""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent
from mathutils import Vector

from . import constants


_ACTION_NAME = "Proscenium_Path"
_EPSILON     = 1e-4


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def _find_root_path_curve(scene: bpy.types.Scene) -> bpy.types.Object | None:
    for obj in scene.objects:
        if obj.type == 'CURVE' and obj.get(constants.PROP_IS_ROOT_PATH):
            return obj
    return None


def _active_fcurves(arm: bpy.types.Object, action: bpy.types.Action):
    """Return the single fcurves collection that the *armature is actually
    evaluating* — the one that drives its pose this frame.

    Pre-4.4: flat ``action.fcurves`` — one collection, shared.
    4.4+ slotted actions: the action holds N channelbags (one per slot);
    the armature picks the one matching its ``action_slot``. Writing to any
    other slot's channelbag has no visible effect, and iterating across all
    of them can delete or clobber fcurves that belong to a different rig
    tracked by the same action (multi-character FBX imports). Limit reads
    and writes to the active slot.
    """
    flat = getattr(action, "fcurves", None)
    if flat is not None:
        return flat
    slot = getattr(arm.animation_data, "action_slot", None)
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


def _get_or_create_fcurve(fcurves, data_path: str, axis: int):
    """Return the fcurve at ``data_path[axis]``, creating it if missing.
    ``fcurves`` is the active-slot collection from ``_active_fcurves``.
    """
    fc = fcurves.find(data_path=data_path, index=axis)
    if fc is None:
        fc = fcurves.new(data_path=data_path, index=axis)
    return fc


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def _control_points_world(curve_obj: bpy.types.Object) -> list[Vector]:
    spline = curve_obj.data.splines[0] if curve_obj.data.splines else None
    if spline is None:
        return []
    mw = curve_obj.matrix_world
    return [mw @ p.co.to_3d() for p in spline.bezier_points]


def _frames_for_points(scene: bpy.types.Scene, n: int) -> list[int]:
    """N control points → N integer frames evenly spread across the scene's
    active frame range. Single point → pinned at ``frame_start``.
    """
    if n <= 0:
        return []
    if n == 1:
        return [scene.frame_start]
    span = max(1, scene.frame_end - scene.frame_start)
    return [scene.frame_start + int(round(i * span / (n - 1))) for i in range(n)]


def _fcurve_matches(fc, frames: list[int], values: list[float]) -> bool:
    """True iff ``fc`` already has exactly these keyframes (same count,
    same frames, same values). Used to short-circuit writes and avoid
    kicking off another depsgraph update (infinite loop).
    """
    kps = fc.keyframe_points
    if len(kps) != len(frames):
        return False
    for kp, f, v in zip(kps, frames, values):
        if abs(kp.co[0] - f) > _EPSILON or abs(kp.co[1] - v) > _EPSILON:
            return False
    return True


def _write_fcurve(fc, frames: list[int], values: list[float]) -> None:
    kps = fc.keyframe_points
    # Clear via remove() in reverse — ``kps.clear()`` is not part of the RNA.
    while len(kps):
        kps.remove(kps[-1], fast=True)
    kps.add(len(frames))
    for i, (f, v) in enumerate(zip(frames, values)):
        kp = kps[i]
        kp.co = (f, v)
        kp.handle_left_type = 'AUTO_CLAMPED'
        kp.handle_right_type = 'AUTO_CLAMPED'
    fc.update()


def _root_pose_bone(arm: bpy.types.Object) -> bpy.types.PoseBone | None:
    return next((pb for pb in arm.pose.bones if pb.parent is None), None)


def _world_to_root_basis_location(
    arm: bpy.types.Object,
    root_pb: bpy.types.PoseBone,
    world_point: Vector,
) -> Vector:
    """Solve for the root bone's bone-local ``location`` value such that the
    root bone's world head position lands on ``world_point``.

    Blender places the root bone's armature-local head at
    ``bone.matrix_local @ matrix_basis.translation``, and the world head at
    ``armature.matrix_world @ that``. Let ``M = armature.matrix_world @
    bone.matrix_local``; then ``M @ root_pb.location == world_point`` solves
    to ``root_pb.location = M.inv() @ world_point``.
    """
    M = arm.matrix_world @ root_pb.bone.matrix_local
    return M.inverted() @ world_point


_VERTICAL_BONE_AXIS = 1
"""Bone-local Y is always the bone's length axis in Blender — from bone
head to tail. For the root/hips bone, that direction points "up" in the
character's rest frame, which is the height channel we never want the
path to touch. This holds across rig conventions (Z-up, Y-up, Mixamo
post-90°-X, etc.) because it's about the bone's local frame, not the
world orientation. Do not infer it from ``arm.matrix_world`` — a Y-up rig
with an identity world transform would otherwise resolve to axis 2."""


def sync_path_to_armature(
    arm: bpy.types.Object,
    scene: bpy.types.Scene,
    curve: bpy.types.Object,
) -> bool:
    """Bake the curve's control points as root-bone ``location`` keyframes.

    The path drives the horizontal world plane only — the bone-local axis
    that maps onto world Z (rig's height axis) is never keyframed, so the
    model's generated height (jumps, crouches, incline) and any authored
    pose keyframes on the root's vertical channel survive untouched.

    Returns True when any fcurve was written, False when everything was
    already in sync (used by the handler to suppress noisy re-entry).
    """
    points = _control_points_world(curve)
    if not points:
        return False

    root_pb = _root_pose_bone(arm)
    if root_pb is None:
        return False

    if arm.animation_data is None:
        arm.animation_data_create()
    if arm.animation_data.action is None:
        arm.animation_data.action = bpy.data.actions.new(name=_ACTION_NAME)
    action = arm.animation_data.action

    # Work exclusively against the armature's active-slot channelbag.
    # Iterating every slot in a multi-character action would delete/modify
    # fcurves that belong to other rigs sharing the same action.
    fcurves = _active_fcurves(arm, action)
    if fcurves is None:
        return False

    vertical_axis = _VERTICAL_BONE_AXIS

    # target.z goes into the skipped vertical bone-local axis, so its value
    # is irrelevant for the horizontal axes we actually write.
    targets = [Vector((p.x, p.y, 0.0)) for p in points]

    frames     = _frames_for_points(scene, len(points))
    basis_locs = [_world_to_root_basis_location(arm, root_pb, t) for t in targets]

    # ``pose.bones["name"]`` — the name is quoted on the data_path so
    # colons in Mixamo bone names don't get mis-parsed.
    data_path = f'pose.bones["{root_pb.name}"].location'

    # The path only writes the two horizontal bone-local axes. Anything
    # else on the action (object-level transforms, the vertical root axis,
    # rotations, scale, other bones) is left completely alone — the path
    # is purely additive for horizontal motion; height and orientation
    # come from wherever else they're already driven.

    written = False
    for axis in range(3):
        if axis == vertical_axis:
            continue
        fc = _get_or_create_fcurve(fcurves, data_path, axis)
        values = [v[axis] for v in basis_locs]
        if _fcurve_matches(fc, frames, values):
            continue
        _write_fcurve(fc, frames, values)
        written = True
    return written


# ---------------------------------------------------------------------------
# Depsgraph handler
# ---------------------------------------------------------------------------

@persistent
def _on_depsgraph_update(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph) -> None:
    settings = getattr(scene, "proscenium", None)
    if settings is None or not getattr(settings, "preview_path_snap", False):
        return
    arm = settings.target_armature
    if arm is None or arm.type != 'ARMATURE':
        return
    curve = _find_root_path_curve(scene)
    if curve is None:
        return

    curve_names = {curve.name, curve.data.name}
    touched = any(
        getattr(u.id, "name", None) in curve_names
        for u in depsgraph.updates
    )
    if not touched:
        return

    # ``sync_path_to_armature`` no-ops when the keyframes already match the
    # curve, which keeps us from re-entering via the update we just caused.
    sync_path_to_armature(arm, scene, curve)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _purge_stale_handlers(handler_list, fn_name: str) -> int:
    """Remove every handler in ``handler_list`` named ``fn_name``.

    Addon reload rebuilds the module — the newly imported function has a
    different id() than the one registered previously, so ``fn not in
    handler_list`` is always True after reload and a fresh copy gets
    appended on top of the stale ones. Stale handlers from prior loads
    keep running, and any bug they had keeps firing forever. Identifying
    by name instead of identity catches them across reloads.
    """
    removed = 0
    for h in list(handler_list):
        if getattr(h, "__name__", None) == fn_name:
            handler_list.remove(h)
            removed += 1
    return removed


def register() -> None:
    _purge_stale_handlers(bpy.app.handlers.depsgraph_update_post, "_on_depsgraph_update")
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister() -> None:
    _purge_stale_handlers(bpy.app.handlers.depsgraph_update_post, "_on_depsgraph_update")
