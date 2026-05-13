"""Blender version differences the addon papers over."""

from __future__ import annotations

import bpy


def pose_bone_is_selected(pb: bpy.types.PoseBone) -> bool:
    """Whether ``pb`` is selected for posing.

    Blender 5.0 stores pose selection on ``PoseBone.select``. Older releases
    used ``Bone.select`` on ``pb.bone``.
    """
    try:
        return bool(pb.select)
    except AttributeError:
        return bool(pb.bone.select)


def pose_bone_select_set(pb: bpy.types.PoseBone, value: bool) -> None:
    """Set pose selection on ``pb`` (API-compatible across Blender versions)."""
    try:
        pb.select = value
    except AttributeError:
        pb.bone.select = value
