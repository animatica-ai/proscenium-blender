"""Build a Blender mesh from the bundled SOMA77 skin asset.

The Kimodo training pipeline ships a SMPL-X-derived skinned mesh at
``kimodo/assets/skeletons/somaskel77/skin_standard.npz`` — 18 056 verts,
36 108 tris, 77 joints with LBS weights. We bundle a copy under
``proscenium_blender/assets/somaskel77_skin.npz`` so users can preview
their generation on a real body, not a stick figure.

The mesh is rigged to SOMA77 (full skeleton), but the canonical
skeleton the user imports through MMCP today is SOMA30 (a 30-joint
subset — fingers, jaw, and a few extra spine joints are absent). We
handle the gap by **weight redistribution**: any vertex weight that
points at a joint not present on the imported armature gets walked up
the SOMA77 parent chain to its nearest ancestor that *is* on the
armature. Visible cost: fingers don't curl independently and the jaw
doesn't open, neither of which Kimodo's canonical animates anyway.

This module owns one public function, ``load_body_mesh(arm_obj, …)``,
which creates and links a child mesh object beneath the armature with
an Armature modifier wired up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import bpy
import numpy as np

from . import coords


_ASSET_PATH = Path(__file__).parent / "assets" / "somaskel77_skin.npz"


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def load_body_mesh(
    arm_obj: bpy.types.Object,
    context: bpy.types.Context,
    *,
    canonical_joints: list[dict],
    floor_lift: float = 0.0,
) -> bpy.types.Object | None:
    """Create the SOMA77 body mesh, parent it to ``arm_obj``, and bind it
    via an Armature modifier with weights redistributed to whatever
    bones the armature actually has.

    Returns the mesh object, or ``None`` if the asset isn't present
    (defensive — should never happen in a packaged plugin install).

    The bundled mesh is bound to SMPL-X's A-pose, but the armature is
    built from the canonical_skeleton's T-pose rest. We re-pose the
    vertices A→T per-bone via LBS so the mesh and armature agree at
    rest while the armature stays in the T-pose frame the model was
    trained against (so generated joint rotations interpret correctly).

    ``floor_lift`` matches the same value the canonical-skeleton import
    uses to put the lowest joint on Blender's z=0 plane.
    """
    if not _ASSET_PATH.exists():
        return None

    data = np.load(_ASSET_PATH)
    vertices_a_pose_mmcp = data["bind_vertices"].astype(np.float64)
    faces         = data["faces"].astype(np.int64)
    j_names       = [str(n) for n in data["rig_joint_names"]]
    bind_xforms   = data["bind_rig_transform"].astype(np.float64)   # (J, 4, 4) A-pose
    lbs_indices   = data["lbs_indices"].astype(np.int32)
    lbs_weights   = data["lbs_weights"].astype(np.float32)
    edges         = data["rig_joint_connections"].astype(np.int64)

    # SOMA77 parent map: index → parent index. Used to walk weights up
    # the chain when a joint isn't on the user's armature.
    parent_of = _build_parent_table(edges, len(j_names))

    # Joints actually on the armature — anything else has its weight
    # walked up to the nearest ancestor that IS present.
    armature_bones = {b.name for b in arm_obj.data.bones}

    # Re-pose vertices from SMPL-X A-pose to canonical T-pose using LBS.
    vertices_t_pose_mmcp = _repose_a_to_t(
        vertices_a_pose_mmcp,
        lbs_indices,
        lbs_weights,
        j_names,
        bind_xforms,
        parent_of,
        armature_bones,
        canonical_joints,
    )

    # Convert vertices from MMCP Y-up → Blender Z-up + apply floor lift.
    vertices_blender = np.empty_like(vertices_t_pose_mmcp)
    for i, v in enumerate(vertices_t_pose_mmcp):
        bx, by, bz = coords.mmcp_pos_to_blender(v)
        vertices_blender[i] = (bx, by, bz + floor_lift)

    # Build the Mesh data block.
    me = bpy.data.meshes.new(f"{arm_obj.name}_body")
    me.from_pydata(
        vertices_blender.tolist(),
        [],
        faces.tolist(),
    )
    me.update()
    me.validate(verbose=False)

    body = bpy.data.objects.new(f"{arm_obj.name}_body", me)
    context.scene.collection.objects.link(body)

    # One vertex group per joint that ends up with non-zero weight after
    # redistribution. Created lazily on first use.
    groups: dict[str, bpy.types.VertexGroup] = {}

    def _vg(name: str) -> bpy.types.VertexGroup:
        if name not in groups:
            groups[name] = body.vertex_groups.new(name=name)
        return groups[name]

    # Resolve each (joint_index, joint_name) → effective armature bone
    # once up front (LBS indices repeat across vertices, so caching the
    # walk dramatically cuts work).
    resolved_bone: list[str | None] = []
    for j_idx in range(len(j_names)):
        bone = _walk_to_armature_bone(j_idx, j_names, parent_of, armature_bones)
        resolved_bone.append(bone)

    # Apply weights — accumulate via ADD so multiple SOMA77 joints
    # collapsing onto the same armature bone sum cleanly.
    n_verts = len(vertices_blender)
    for vi in range(n_verts):
        for k in range(lbs_indices.shape[1]):
            w = float(lbs_weights[vi, k])
            if w <= 0.0:
                continue
            j_idx = int(lbs_indices[vi, k])
            bone = resolved_bone[j_idx]
            if bone is None:
                continue
            _vg(bone).add([vi], w, "ADD")

    # Parent + Armature modifier.
    body.parent = arm_obj
    body.parent_type = "OBJECT"
    body.matrix_parent_inverse = arm_obj.matrix_world.inverted()

    mod = body.modifiers.new(name="Armature", type="ARMATURE")
    mod.object = arm_obj
    mod.use_vertex_groups = True
    mod.use_bone_envelopes = False

    # Tag for traceability — the panel can detect "this armature already
    # has a body" and skip re-import.
    body["proscenium_body_for"] = arm_obj.name

    return body


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_parent_table(edges: np.ndarray, n_joints: int) -> list[int]:
    """Convert ``rig_joint_connections`` (parent, child) pairs into a
    flat ``index → parent_index`` table. Root's parent is ``-1``.

    SOMA77's edge list has one row per non-root joint, so the result is
    a length-``n_joints`` list with exactly one ``-1`` entry.
    """
    parent_of = [-1] * n_joints
    for edge in edges:
        parent_idx, child_idx = int(edge[0]), int(edge[1])
        parent_of[child_idx] = parent_idx
    return parent_of


def _walk_to_armature_bone(
    j_idx: int,
    j_names: list[str],
    parent_of: list[int],
    armature_bones: set[str],
) -> str | None:
    """Return the nearest ancestor whose name is on the armature, or
    ``None`` if the chain reaches the root without finding one.

    A SOMA77 finger joint walked up this chain lands on the wrist; a
    jaw bone lands on the head. That's the right semantics for keeping
    the mesh skinned when the armature has fewer joints — the lost
    detail is articulation we don't animate anyway.
    """
    cur = j_idx
    while cur >= 0:
        name = j_names[cur]
        if name in armature_bones:
            return name
        cur = parent_of[cur]
    return None


def _walk_to_armature_index(
    j_idx: int,
    j_names: list[str],
    parent_of: list[int],
    armature_bones: set[str],
) -> int:
    """``_walk_to_armature_bone`` but returns the SMPL-X index. Used by
    the A→T re-pose where we need the resolved joint's bind transform,
    not just its name."""
    cur = j_idx
    while cur >= 0:
        if j_names[cur] in armature_bones:
            return cur
        cur = parent_of[cur]
    return -1


def _repose_a_to_t(
    vertices_a_pose: np.ndarray,        # (N, 3) MMCP A-pose
    lbs_indices: np.ndarray,            # (N, K) int
    lbs_weights: np.ndarray,            # (N, K) float
    j_names: list[str],                 # length J (SMPL-X)
    bind_xforms: np.ndarray,            # (J, 4, 4) SMPL-X A-pose
    parent_of: list[int],               # length J
    armature_bones: set[str],
    canonical_joints: list[dict],
) -> np.ndarray:
    """Re-pose mesh vertices from SMPL-X's A-pose bind to the canonical
    skeleton's T-pose rest, via LBS.

    For each weight, the vertex is unbound from the SMPL-X joint's
    A-pose and re-bound to the **resolved armature joint's** T-pose
    (resolved = walk SMPL-X parents until we hit a joint that exists
    on the armature; weight redistribution to fingers/jaw etc.).

    Per-bone transform: ``M = T_pose_world[resolved] @ inv(A_pose_world[resolved])``
    where the T-pose translation comes from cumulative ``rest_translation``
    in ``canonical_joints`` and the rotation is identity (canonical rest
    rotations are identity for SOMA30).
    """
    # T-pose world position for each canonical joint (cumulative).
    t_pose_pos: dict[str, np.ndarray] = {}
    for j in canonical_joints:
        name = j["name"]
        parent = j.get("parent")
        rt = j.get("rest_translation") or [0.0, 0.0, 0.0]
        local = np.array([float(rt[0]), float(rt[1]), float(rt[2])], dtype=np.float64)
        if parent is None or parent not in t_pose_pos:
            t_pose_pos[name] = local
        else:
            t_pose_pos[name] = t_pose_pos[parent] + local

    # Per-SMPL-X-joint M (4x4): mapping vertex from A-pose world to
    # T-pose world via the resolved armature joint.
    n_smplx = len(j_names)
    M_per_joint = np.tile(np.eye(4, dtype=np.float64), (n_smplx, 1, 1))
    for i in range(n_smplx):
        resolved = _walk_to_armature_index(i, j_names, parent_of, armature_bones)
        if resolved < 0:
            continue
        resolved_name = j_names[resolved]
        if resolved_name not in t_pose_pos:
            continue
        a_inv = np.linalg.inv(bind_xforms[resolved])
        t = np.eye(4)
        t[:3, 3] = t_pose_pos[resolved_name]
        M_per_joint[i] = t @ a_inv

    # Vectorised LBS: (N, K, 4, 4) gathered, transform homogeneous (N, 4),
    # weight, sum.
    n_verts = vertices_a_pose.shape[0]
    v_h = np.concatenate([vertices_a_pose, np.ones((n_verts, 1))], axis=1)  # (N, 4)
    M_per_slot = M_per_joint[lbs_indices]                                   # (N, K, 4, 4)
    v_per_slot = np.einsum('ikab,ib->ika', M_per_slot, v_h)                 # (N, K, 4)
    v_t_h = (v_per_slot * lbs_weights[:, :, None]).sum(axis=1)              # (N, 4)
    weight_sum = lbs_weights.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    return (v_t_h[:, :3] / weight_sum).astype(np.float64)


def has_body_mesh(arm_obj: bpy.types.Object) -> bool:
    """Cheap check: does any object in the same scene already claim
    ``arm_obj`` as its proscenium-body parent?"""
    target = arm_obj.name
    for obj in bpy.data.objects:
        if obj.get("proscenium_body_for") == target:
            return True
    return False


def asset_available() -> bool:
    """``True`` if the bundled npz is present — controls whether the
    import operator surfaces the body checkbox at all."""
    return _ASSET_PATH.exists()


# Joints expected on the armature for body skinning. Used by the import
# operator to decide whether body-loading is worth offering — the SOMA30
# canonical exposes most of these directly, custom rigs may not.
SOMA_BODY_JOINTS = (
    "Hips",
    "Spine", "Spine1", "Spine2",
    "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
)


def looks_like_kimodo_skeleton(arm_obj: bpy.types.Object, threshold: int = 12) -> bool:
    """Heuristic: does this armature look like one Kimodo's canonical
    body mesh would skin to? We match by bone-name overlap rather than
    exact-set so retargeted / extended SOMA variants still qualify.
    """
    bones = {b.name for b in arm_obj.data.bones}
    overlap = sum(1 for j in SOMA_BODY_JOINTS if j in bones)
    return overlap >= threshold
