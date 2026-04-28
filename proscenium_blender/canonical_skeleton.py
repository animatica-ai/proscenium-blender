"""Build a Blender Armature from an MMCP canonical skeleton.

The MMCP server publishes ``canonical_skeleton`` in
``/capabilities.models[].canonical_skeleton``. Each joint has a name, a
parent name, a local-space ``rest_translation`` (meters, MMCP frame) and a
``rest_rotation`` quaternion (typically identity for Kimodo skeletons).

This module turns that JSON into a real Blender armature so the user has
something to animate. Round-trip safe: the names and parent links match
exactly what the server expects to receive back, so generation works without
a retargeting layer (the MMCP v1 server publishes
``supports_retargeting: false``).

Bone construction strategy:
  * head = accumulated rest_translation from root to this joint
  * tail = the **first listed child**'s head — matches the model's
    convention of which direction is "up the chain" for that joint. Using
    the centroid of children for branching joints (Hips → spine + legs)
    makes the Hips bone point straight down, which then misinterprets
    every local rotation the model produces as a 180° tilt. Picking the
    first child puts the bone's local +Y along the spine for Hips, along
    the neck for Chest, along the head-end for Head — matching how the
    model authored its rest pose.
  * leaf joints: tail = head + (+Z * 0.05)  (just for visibility)
  * roll = 0   (rest_rotation is baked into the pose, not the bone roll —
    Blender bone rolls don't round-trip cleanly through quaternion data)

All positions are converted MMCP → Blender at import time via ``coords``.
"""

from __future__ import annotations

from typing import Any

import bpy
from bpy.props import BoolProperty
from mathutils import Vector

from . import body_mesh, coords, mmcp_client


LEAF_TAIL_LENGTH = 0.05   # metres, +Y in MMCP frame (= +Z in Blender)


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_import_canonical_skeleton(bpy.types.Operator):
    bl_idname = "proscenium.import_canonical_skeleton"
    bl_label = "Import Canonical Skeleton"
    bl_description = (
        "Build a Blender armature from the selected MMCP model's "
        "canonical_skeleton. The new armature becomes the target for "
        "generation — its joint names match the server's expectations exactly"
    )
    bl_options = {'REGISTER', 'UNDO'}

    with_body: BoolProperty(
        name="Include body mesh",
        description=(
            "Also import the SOMA77 reference body mesh, skinned to the "
            "imported armature. Weights for joints not present on the "
            "armature (fingers, jaw) get redistributed to their nearest "
            "ancestor — fingers don't curl, but the body shape is preserved"
        ),
        default=True,
    )

    def execute(self, context):
        settings = context.scene.proscenium
        model_id = settings.model_id

        if not model_id:
            self.report({'ERROR'}, "Pick a model first (Connection panel → Connect → Model)")
            return {'CANCELLED'}

        model = mmcp_client.cached_model(model_id)
        if model is None:
            self.report({'ERROR'}, f"Model {model_id!r} not in the cached capabilities; reconnect first")
            return {'CANCELLED'}

        skel = model.get("canonical_skeleton") or {}
        joints = skel.get("joints") or []
        if not joints:
            self.report({'ERROR'}, f"Model {model_id!r} has no canonical_skeleton.joints")
            return {'CANCELLED'}

        try:
            arm_obj, floor_lift = build_armature_from_canonical(model_id, joints, context)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        # Wire it as the generation target.
        settings.target_armature = arm_obj

        body_loaded = False
        if (
            self.with_body
            and body_mesh.asset_available()
            and body_mesh.looks_like_kimodo_skeleton(arm_obj)
        ):
            try:
                body_obj = body_mesh.load_body_mesh(
                    arm_obj, context,
                    canonical_joints=joints,
                    floor_lift=floor_lift,
                )
                body_loaded = body_obj is not None
            except Exception as exc:                                 # noqa: BLE001
                # Mesh is a nice-to-have; never fail the armature import on it.
                self.report({'WARNING'}, f"Imported armature but body mesh failed: {exc}")

        msg = f"Imported {model_id} ({len(joints)} joints)"
        if body_loaded:
            msg += " with body mesh"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_armature_from_canonical(
    model_id: str,
    joints: list[dict[str, Any]],
    context,
) -> tuple[bpy.types.Object, float]:
    """Create + link a Blender Armature object whose bones mirror ``joints``.

    Returns ``(armature_object, floor_lift)``. The lift is the +Z offset
    applied to every bone head/tail so the lowest joint sits on Blender's
    z=0 plane; downstream code that wants to align other geometry with the
    rest pose (e.g. the body mesh) needs the same value.

    Always creates a new object — re-importing the same model produces
    ``model_id.001``, ``.002``, etc.
    """
    name_to_local: dict[str, tuple[float, float, float]] = {}
    name_to_parent: dict[str, str | None] = {}
    children_of: dict[str, list[str]] = {}
    order: list[str] = []

    for j in joints:
        name = j.get("name")
        if not name:
            raise ValueError("joint missing 'name'")
        parent = j.get("parent")
        rt = j.get("rest_translation") or [0.0, 0.0, 0.0]
        if len(rt) != 3:
            raise ValueError(f"joint {name!r}: rest_translation must be length 3")
        name_to_local[name]  = (float(rt[0]), float(rt[1]), float(rt[2]))
        name_to_parent[name] = parent
        children_of.setdefault(parent, []).append(name)
        order.append(name)

    # Resolve global rest positions in MMCP frame. ``joints`` is guaranteed
    # parents-before-children by the spec, so a single forward pass is enough.
    name_to_global_mmcp: dict[str, tuple[float, float, float]] = {}
    for name in order:
        local = name_to_local[name]
        parent = name_to_parent[name]
        if parent is None:
            name_to_global_mmcp[name] = local
        else:
            px, py, pz = name_to_global_mmcp[parent]
            lx, ly, lz = local
            name_to_global_mmcp[name] = (px + lx, py + ly, pz + lz)

    # Lift the whole skeleton so the lowest joint sits on the floor (Blender
    # z = 0). The model's root_positions stream encodes absolute root height
    # (~1 m for SOMA), so once an animation is baked the character pops to
    # the right place — but the *rest* pose without that offset would put
    # feet below the floor. Adding the lift to every joint's head/tail keeps
    # them aligned during baking too (the bake's `delta = world - rest_head`
    # accounts for it automatically).
    lowest_blender_z = min(
        coords.mmcp_pos_to_blender(p)[2] for p in name_to_global_mmcp.values()
    )
    floor_lift = max(0.0, -lowest_blender_z)

    # Build the armature.
    arm_data = bpy.data.armatures.new(f"{model_id}_data")
    arm_obj  = bpy.data.objects.new(model_id, arm_data)
    context.scene.collection.objects.link(arm_obj)

    # Edit mode is the only place EditBones can be created.
    prev_active = context.view_layer.objects.active
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        edit_bones = arm_data.edit_bones

        for name in order:
            head_mmcp = name_to_global_mmcp[name]
            head = Vector(coords.mmcp_pos_to_blender(head_mmcp))
            head.z += floor_lift

            children = children_of.get(name, [])
            if children:
                # Take the first listed child as the bone's "primary"
                # direction. The joints[] array is in the model's intended
                # order (parents-before-children, primary chain first), so
                # for Hips this picks Spine1 (not LeftLeg/RightLeg), which
                # keeps the bone's local +Y aligned with what the model
                # treats as the rest direction.
                primary = children[0]
                tail = Vector(coords.mmcp_pos_to_blender(name_to_global_mmcp[primary]))
                tail.z += floor_lift
                if (tail - head).length < 1e-4:
                    tail = head + Vector((0, 0, LEAF_TAIL_LENGTH))
            else:
                tail = head + Vector((0, 0, LEAF_TAIL_LENGTH))

            bone = edit_bones.new(name)
            bone.head = head
            bone.tail = tail
            bone.roll = 0.0

        # Wire parent links in a second pass so child bones exist.
        for name in order:
            parent = name_to_parent[name]
            if parent is not None:
                edit_bones[name].parent = edit_bones[parent]
                # Optional connect: only if parent's tail coincides with this
                # bone's head (otherwise we'd snap the bone). Loose-link
                # everything for v1 — preserves rest_translation faithfully.
                edit_bones[name].use_connect = False
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = prev_active

    # Stash the model id on the armature so the rest of the addon can verify
    # the rig matches the server's canonical skeleton without name-matching.
    arm_obj["proscenium_canonical_model"] = model_id

    return arm_obj, floor_lift
