# SPDX-License-Identifier: GPL-3.0-or-later
"""
Proscenium for Blender — AI Motion Generation Addon
====================================================

Select an armature with a few keyframes, click Generate, and the server
fills in the motion using a backend MMCP-compatible motion model.

The addon is ML-free: all generation, retargeting, and keyframe
optimisation runs on the backend server.
"""

bl_info = {
    "name": "Proscenium — AI Motion Generation",
    "author": "Animatica",
    "version": (0, 3, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Proscenium",
    "description": "AI motion generation — select armature, set keyframes, generate",
    "category": "Animation",
}

import bpy
from bpy.app.handlers import persistent

from . import properties
from . import operators
from . import canonical_skeleton
from . import constraints_ui
from . import panels
from . import path_follow
from . import timeline_overlay
from . import timeline_operators


# ---------------------------------------------------------------------------
# Persistent handlers — save/load prompt blocks onto the target armature
# ---------------------------------------------------------------------------

@persistent
def _proscenium_save_pre(dummy):
    """Before saving the .blend, persist the current prompt_blocks onto the
    target armature's custom properties so they survive file reloads and
    per-armature switching."""
    for scene in bpy.data.scenes:
        settings = getattr(scene, "proscenium", None)
        if settings is None:
            continue
        arm = settings.target_armature
        if arm is None:
            continue
        try:
            properties.save_blocks_to_armature(arm, settings)
        except Exception:
            pass


@persistent
def _proscenium_load_post(dummy):
    """After loading a .blend, hydrate settings.prompt_blocks from the
    target armature's custom properties (or seed defaults)."""
    for scene in bpy.data.scenes:
        settings = getattr(scene, "proscenium", None)
        if settings is None:
            continue
        arm = settings.target_armature
        if arm is None:
            continue
        try:
            properties.load_blocks_from_armature(arm, settings)
            settings.previous_target_armature = arm
        except Exception:
            pass


def _reset_runtime_flags() -> None:
    """Clear the ``is_generating`` / ``cancel_requested`` flags on every scene.

    Reloading the addon kills any in-flight modal operator without giving it
    a chance to run ``_cleanup``, which leaves the flags stuck at True and
    the UI showing a "Generating…" state that can never clear. Reset them on
    every register() so reloading is a clean slate.

    ``bpy.data`` can be a restricted ``_RestrictData`` proxy (no ``scenes``
    attribute) when register() runs during startup or through the script
    context of the MCP bridge. In that case, defer the reset until the next
    app tick via a one-shot timer.
    """
    try:
        scenes = bpy.data.scenes
    except AttributeError:
        bpy.app.timers.register(_reset_runtime_flags, first_interval=0.0)
        return
    for scene in scenes:
        s = getattr(scene, "proscenium", None)
        if s is None:
            continue
        try:
            s.is_generating = False
            s.cancel_requested = False
            if hasattr(s, "generation_progress"):
                s.generation_progress = 0.0
        except (AttributeError, ReferenceError):
            pass


def _purge_stale_handlers(handler_list, fn_name: str) -> None:
    """Drop every previously-registered copy of ``fn_name`` before a
    fresh append. Addon reload creates a new function object each time,
    so the usual ``if fn not in handler_list`` guard never matches and
    stale copies accumulate — each still firing with whatever behavior
    it had at the time of registration. Match by name, not identity."""
    for h in list(handler_list):
        if getattr(h, "__name__", None) == fn_name:
            handler_list.remove(h)


def register():
    properties.register()
    operators.register()
    bpy.utils.register_class(canonical_skeleton.PROSCENIUM_OT_import_canonical_skeleton)
    constraints_ui.register()
    panels.register()
    path_follow.register()
    timeline_operators.register()
    timeline_overlay.register_draw_handler()

    _reset_runtime_flags()

    # Install persistent handlers, purging any stale copies from prior loads.
    _purge_stale_handlers(bpy.app.handlers.save_pre, "_proscenium_save_pre")
    bpy.app.handlers.save_pre.append(_proscenium_save_pre)
    _purge_stale_handlers(bpy.app.handlers.load_post, "_proscenium_load_post")
    bpy.app.handlers.load_post.append(_proscenium_load_post)


def unregister():
    _purge_stale_handlers(bpy.app.handlers.save_pre, "_proscenium_save_pre")
    _purge_stale_handlers(bpy.app.handlers.load_post, "_proscenium_load_post")

    timeline_overlay.unregister_draw_handler()
    timeline_operators.unregister()
    path_follow.unregister()
    panels.unregister()
    constraints_ui.unregister()
    bpy.utils.unregister_class(canonical_skeleton.PROSCENIUM_OT_import_canonical_skeleton)
    operators.unregister()
    properties.unregister()
