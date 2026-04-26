"""Timeline interaction operators for Proscenium prompt-block strips.

Uses keymap-based approach: clicking on a strip automatically selects/drags
it without needing to activate a modal operator first.

Provides:
- PROSCENIUM_OT_timeline_strip_action: click/drag on strips (keymap-driven)
- PROSCENIUM_OT_timeline_strip_add_click: double-click empty area to add strip
- PROSCENIUM_OT_timeline_strip_context_menu: right-click context menu
- PROSCENIUM_OT_timeline_strip_delete: Delete/Backspace to remove active strip
- PROSCENIUM_OT_edit_strip_prompt: popup dialog for editing a strip's prompt
- draw_timeline_header(): appended to DOPESHEET_HT_header for + / - buttons
- register_keymaps() / unregister_keymaps()
"""

import time

import bpy
from bpy.props import IntProperty, StringProperty

from .timeline_overlay import (
    hit_test_strips,
    hit_test_lane_resize,
    pixel_to_frame,
    find_neighbors,
    find_gap,
    inline_edit_state,
    get_strip_height,
    set_strip_height,
    STRIP_Y_OFFSET,
    LANE_PADDING,
    MIN_STRIP_HEIGHT,
    MAX_STRIP_HEIGHT,
)


# Module-level double-click tracking (persists across operator invocations)
_last_click_time: float = 0.0
_last_click_idx: int = -1
_DOUBLE_CLICK_THRESHOLD: float = 0.35  # seconds

# Keymap storage
_addon_keymaps = []


def _is_in_lane(mouse_y):
    """Check if mouse Y is within the Proscenium strip lane area."""
    lane_y0 = STRIP_Y_OFFSET - LANE_PADDING
    lane_y1 = STRIP_Y_OFFSET + get_strip_height() + LANE_PADDING
    return lane_y0 <= mouse_y <= lane_y1


def _timeline_poll(context):
    """Common poll for timeline operators."""
    return (
        context.area is not None
        and context.area.type == "DOPESHEET_EDITOR"
        and hasattr(context, "space_data")
        and context.space_data is not None
        and context.space_data.mode == "TIMELINE"
        and hasattr(context.scene, "proscenium")
    )


# ---------------------------------------------------------------------------
# Main strip interaction operator (triggered by keymap on LEFTMOUSE)
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_timeline_strip_action(bpy.types.Operator):
    """Click/drag Proscenium timeline strips.

    Automatically invoked by keymap when clicking in the Timeline.
    If the click lands on a strip, the operator handles select + drag.
    If not, it returns PASS_THROUGH so normal timeline scrubbing works.
    """

    bl_idname = "proscenium.timeline_strip_action"
    bl_label = "Proscenium Strip Action"
    bl_options = {"REGISTER", "UNDO"}

    # State for the current drag
    _state: str = "IDLE"
    _active_idx: int = -1
    _zone: str = ""
    _original_start: int = 0
    _original_end: int = 0
    _drag_offset: int = 0

    @classmethod
    def poll(cls, context):
        return (
            _timeline_poll(context)
            and len(context.scene.proscenium.prompt_blocks) > 0
        )

    def invoke(self, context, event):
        global _last_click_time, _last_click_idx

        # --- Lane resize (top edge of lane) ---
        if hit_test_lane_resize(
            context, event.mouse_region_x, event.mouse_region_y
        ):
            self._state = "DRAGGING_LANE_RESIZE"
            self._resize_start_y = event.mouse_region_y
            self._resize_start_h = get_strip_height()
            context.window.cursor_set("MOVE_Y")
            context.window_manager.modal_handler_add(self)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        hit = hit_test_strips(context, event.mouse_region_x, event.mouse_region_y)

        if hit["index"] is None:
            # Miss — let Blender handle normal timeline scrubbing
            return {"PASS_THROUGH"}

        idx = hit["index"]
        zone = hit["zone"]
        props = context.scene.proscenium
        fr = props.prompt_blocks[idx]

        # --- Double-click detection ---
        now = time.time()
        if (
            idx == _last_click_idx
            and (now - _last_click_time) < _DOUBLE_CLICK_THRESHOLD
        ):
            # Double-click → inline prompt editing on the strip
            props.active_block_index = idx
            _last_click_time = 0.0
            _last_click_idx = -1
            bpy.ops.proscenium.timeline_strip_inline_edit(
                "INVOKE_DEFAULT", index=idx,
            )
            return {"FINISHED"}

        _last_click_time = now
        _last_click_idx = idx

        # --- Select strip ---
        props.active_block_index = idx

        # --- Begin drag ---
        self._active_idx = idx
        self._original_start = fr.frame_start
        self._original_end = fr.frame_end

        if zone == "edge_start":
            self._state = "DRAGGING_EDGE_START"
        elif zone == "edge_end":
            self._state = "DRAGGING_EDGE_END"
        else:
            self._state = "DRAGGING_BODY"
            mouse_frame = pixel_to_frame(context, event.mouse_region_x)
            self._drag_offset = mouse_frame - fr.frame_start

        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if context.area is None:
            return {"CANCELLED"}

        props = context.scene.proscenium

        # Cancel drag
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._cancel_drag(context)
            return {"CANCELLED"}

        # Dispatch by state
        if self._state == "DRAGGING_LANE_RESIZE":
            return self._handle_lane_resize(context, event)
        elif self._state == "DRAGGING_BODY":
            return self._handle_body_drag(context, event, props)
        elif self._state in ("DRAGGING_EDGE_START", "DRAGGING_EDGE_END"):
            return self._handle_edge_drag(context, event, props)

        return {"PASS_THROUGH"}

    # -- Edge drag --------------------------------------------------------

    def _handle_edge_drag(self, context, event, props):
        if event.type == "MOUSEMOVE":
            fr = props.prompt_blocks[self._active_idx]
            new_frame = pixel_to_frame(context, event.mouse_region_x)
            left_end, right_start = find_neighbors(
                props.prompt_blocks, self._active_idx
            )

            if self._state == "DRAGGING_EDGE_START":
                low = max(1, left_end)
                fr.frame_start = max(low, min(new_frame, fr.frame_end - 1))
            else:  # DRAGGING_EDGE_END
                hi = right_start if right_start is not None else new_frame
                fr.frame_end = min(hi, max(fr.frame_start + 1, new_frame))

            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            context.window.cursor_set("DEFAULT")
            return {"FINISHED"}

        return {"RUNNING_MODAL"}

    # -- Body drag --------------------------------------------------------

    def _handle_body_drag(self, context, event, props):
        if event.type == "MOUSEMOVE":
            fr = props.prompt_blocks[self._active_idx]
            mouse_frame = pixel_to_frame(context, event.mouse_region_x)
            duration = self._original_end - self._original_start

            new_start = max(1, mouse_frame - self._drag_offset)
            new_end = new_start + duration

            # Clamp to neighbors
            left_end, right_start = find_neighbors(
                props.prompt_blocks, self._active_idx
            )
            if new_start < left_end:
                new_start = left_end
                new_end = new_start + duration
            if right_start is not None and new_end > right_start:
                new_end = right_start
                new_start = new_end - duration

            new_start = max(1, new_start)
            fr.frame_start = new_start
            fr.frame_end = new_start + duration
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            context.window.cursor_set("DEFAULT")
            return {"FINISHED"}

        return {"RUNNING_MODAL"}

    # -- Lane resize -----------------------------------------------------

    def _handle_lane_resize(self, context, event):
        if event.type == "MOUSEMOVE":
            delta = event.mouse_region_y - self._resize_start_y
            set_strip_height(self._resize_start_h + delta)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            context.window.cursor_set("DEFAULT")
            context.area.tag_redraw()
            return {"FINISHED"}

        return {"RUNNING_MODAL"}

    # -- Cancel -----------------------------------------------------------

    def _cancel_drag(self, context):
        if self._state == "DRAGGING_LANE_RESIZE":
            set_strip_height(
                getattr(self, "_resize_start_h", get_strip_height())
            )
            context.window.cursor_set("DEFAULT")
            context.area.tag_redraw()
            return

        props = context.scene.proscenium
        if 0 <= self._active_idx < len(props.prompt_blocks):
            fr = props.prompt_blocks[self._active_idx]
            fr.frame_start = self._original_start
            fr.frame_end = self._original_end
        context.window.cursor_set("DEFAULT")
        context.area.tag_redraw()

    def cancel(self, context):
        self._cancel_drag(context)


# ---------------------------------------------------------------------------
# Double-click on empty area to add strip at that position
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_timeline_strip_add_click(bpy.types.Operator):
    """Double-click on empty timeline area to add a new strip at that position."""

    bl_idname = "proscenium.timeline_strip_add_click"
    bl_label = "Add Strip at Click"
    bl_options = {"REGISTER", "UNDO"}

    # Module-level tracking for double-click on empty
    _last_empty_click_time: float = 0.0

    @classmethod
    def poll(cls, context):
        return _timeline_poll(context)

    def invoke(self, context, event):
        # Only react if click is in the lane area
        if not _is_in_lane(event.mouse_region_y):
            return {"PASS_THROUGH"}

        # Check if there's already a strip here — if so, pass through
        hit = hit_test_strips(context, event.mouse_region_x, event.mouse_region_y)
        if hit["index"] is not None:
            return {"PASS_THROUGH"}

        # Double-click detection on empty area
        now = time.time()
        cls = type(self)
        if (now - cls._last_empty_click_time) < _DOUBLE_CLICK_THRESHOLD:
            cls._last_empty_click_time = 0.0
            # Create strip at click position
            return self._add_strip_at(context, event)

        cls._last_empty_click_time = now
        return {"PASS_THROUGH"}

    def _add_strip_at(self, context, event):
        """Add a new strip at the click position, fitting into the available gap."""
        props = context.scene.proscenium
        click_frame = pixel_to_frame(context, event.mouse_region_x)
        scene_start = max(1, context.scene.frame_start)
        scene_end = context.scene.frame_end

        # Find the gap that contains the click position
        from .timeline_overlay import get_sorted_blocks
        sorted_items = get_sorted_blocks(props.prompt_blocks)

        # Determine gap boundaries around the click
        gap_start = scene_start
        gap_end = scene_end
        for _idx, s_start, s_end in sorted_items:
            if s_end <= click_frame:
                gap_start = max(gap_start, s_end)
            if s_start > click_frame and s_start < gap_end:
                gap_end = s_start
            # Click is inside an existing strip — no room
            if s_start <= click_frame < s_end:
                self.report({"WARNING"}, "No room for a new strip here")
                return {"CANCELLED"}

        gap_length = gap_end - gap_start
        if gap_length < 2:
            self.report({"WARNING"}, "No room for a new strip here")
            return {"CANCELLED"}

        # Fill the entire gap
        new_start = gap_start
        new_end = gap_end

        new_range = props.prompt_blocks.add()
        new_range.prompt = ""
        new_range.frame_start = new_start
        new_range.frame_end = new_end
        new_range.enabled = True

        new_idx = len(props.prompt_blocks) - 1
        props.active_block_index = new_idx

        context.area.tag_redraw()

        # Start inline editing on the new strip (same as double-click)
        bpy.ops.proscenium.timeline_strip_inline_edit(
            "INVOKE_DEFAULT", index=new_idx,
        )

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Add strip between nearest keyframes of source armature
# ---------------------------------------------------------------------------

def _get_armature_keyframes(armature):
    """Collect all unique keyframe frame numbers from an armature's action.

    Uses the Blender 5.0 slotted-actions API (channelbag) to access fcurves.
    """
    from bpy_extras import anim_utils

    frames: set[int] = set()
    if not armature or not armature.animation_data or not armature.animation_data.action:
        return sorted(frames)

    action = armature.animation_data.action
    slot = armature.animation_data.action_slot
    if slot is None:
        return sorted(frames)

    channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)
    if channelbag is None:
        return sorted(frames)

    for fc in channelbag.fcurves:
        for kp in fc.keyframe_points:
            frames.add(int(kp.co[0]))
    return sorted(frames)


class PROSCENIUM_OT_add_strip_between_keyframes(bpy.types.Operator):
    """Add a new strip spanning between the two nearest keyframes around the
    click position on the source armature's timeline."""

    bl_idname = "proscenium.add_strip_between_keyframes"
    bl_label = "Add Strip Between Keyframes"
    bl_options = {"REGISTER", "UNDO"}

    frame: IntProperty(
        name="Frame",
        description="Frame around which to find bracketing keyframes",
        default=1,
    )

    @classmethod
    def poll(cls, context):
        return _timeline_poll(context)

    def execute(self, context):
        props = context.scene.proscenium
        armature = props.target_armature

        if not armature:
            self.report({"WARNING"}, "No source armature selected")
            return {"CANCELLED"}

        kf_list = _get_armature_keyframes(armature)
        if len(kf_list) < 2:
            self.report({"WARNING"}, "Source armature has fewer than 2 keyframes")
            return {"CANCELLED"}

        click = self.frame

        # Find the two keyframes that bracket the click position
        kf_before = None
        kf_after = None
        for kf in kf_list:
            if kf <= click:
                kf_before = kf
            if kf >= click and kf_after is None:
                kf_after = kf

        # Edge cases: click before first keyframe or after last
        if kf_before is None:
            kf_before = kf_list[0]
        if kf_after is None:
            kf_after = kf_list[-1]

        # If click lands exactly on a keyframe, expand to neighbors
        if kf_before == kf_after:
            idx = kf_list.index(kf_before)
            if idx > 0:
                kf_before = kf_list[idx - 1]
            elif idx < len(kf_list) - 1:
                kf_after = kf_list[idx + 1]

        new_start = kf_before
        new_end = kf_after

        if new_start >= new_end:
            self.report({"WARNING"}, "Could not determine keyframe range")
            return {"CANCELLED"}

        # Clamp to scene bounds
        new_start = max(1, new_start)
        new_end = min(context.scene.frame_end, new_end)

        # Find the gap around the click position and intersect with
        # the keyframe range so the strip fits without overlapping.
        from .timeline_overlay import get_sorted_blocks
        gap_start = max(1, context.scene.frame_start)
        gap_end = context.scene.frame_end
        for _idx, s_start, s_end in get_sorted_blocks(props.prompt_blocks):
            if s_end <= click and s_end > gap_start:
                gap_start = s_end
            if s_start > click and s_start < gap_end:
                gap_end = s_start

        # Intersect keyframe range with gap
        new_start = max(new_start, gap_start)
        new_end = min(new_end, gap_end)

        if new_start >= new_end:
            self.report({"WARNING"}, "No room for a strip between these keyframes")
            return {"CANCELLED"}

        new_range = props.prompt_blocks.add()
        new_range.prompt = ""
        new_range.frame_start = new_start
        new_range.frame_end = new_end
        new_range.enabled = True

        props.active_block_index = len(props.prompt_blocks) - 1

        self.report({"INFO"}, f"Added strip {new_start}–{new_end} (between keyframes)")

        # Redraw all timeline areas
        for area in context.screen.areas:
            if area.type == "DOPESHEET_EDITOR":
                area.tag_redraw()

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Delete strip (Backspace / Delete key)
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_timeline_strip_delete(bpy.types.Operator):
    """Delete the active Proscenium strip (Backspace/Delete).

    Only activates when the mouse cursor is inside the strip lane area.
    Otherwise passes through so Blender can handle keyframe deletion.
    """

    bl_idname = "proscenium.timeline_strip_delete"
    bl_label = "Delete Proscenium Strip"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            _timeline_poll(context)
            and len(context.scene.proscenium.prompt_blocks) > 0
        )

    def invoke(self, context, event):
        # Only intercept if the mouse is hovering over the strip lane
        if not _is_in_lane(event.mouse_region_y):
            return {"PASS_THROUGH"}

        props = context.scene.proscenium
        idx = props.active_block_index

        if not (0 <= idx < len(props.prompt_blocks)):
            return {"PASS_THROUGH"}

        fr = props.prompt_blocks[idx]
        self.report({"INFO"}, f"Deleted strip: {fr.prompt or '(no prompt)'}")
        props.prompt_blocks.remove(idx)

        # Adjust active index
        if len(props.prompt_blocks) == 0:
            props.active_block_index = 0
        elif props.active_block_index >= len(props.prompt_blocks):
            props.active_block_index = len(props.prompt_blocks) - 1

        context.area.tag_redraw()
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Right-click context menu
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_timeline_strip_context_menu(bpy.types.Operator):
    """Right-click context menu for Proscenium timeline strips."""

    bl_idname = "proscenium.timeline_strip_context_menu"
    bl_label = "Proscenium Strip Menu"

    @classmethod
    def poll(cls, context):
        return _timeline_poll(context)

    def invoke(self, context, event):
        # Only show menu if click is in the lane area
        if not _is_in_lane(event.mouse_region_y):
            return {"PASS_THROUGH"}

        # Check if right-click hit a strip
        hit = hit_test_strips(context, event.mouse_region_x, event.mouse_region_y)

        if hit["index"] is not None:
            # Select the strip
            context.scene.proscenium.active_block_index = hit["index"]

        # Store click position for "Add Strip Here"
        self._click_frame = pixel_to_frame(context, event.mouse_region_x)
        self._hit_index = hit["index"]

        wm = context.window_manager
        wm.popup_menu(self._draw_menu, title="Proscenium Strip")
        return {"FINISHED"}

    def _draw_menu(self, menu, context):
        layout = menu.layout
        props = context.scene.proscenium

        if self._hit_index is not None and 0 <= self._hit_index < len(props.prompt_blocks):
            fr = props.prompt_blocks[self._hit_index]
            layout.label(text=f"Strip: {fr.prompt or '(no prompt)'}")
            layout.separator()

            # Edit prompt
            op = layout.operator(
                "proscenium.edit_strip_prompt",
                text="Edit Prompt",
                icon="TEXT",
            )
            op.index = self._hit_index

            # Toggle enabled
            toggle_text = "Disable" if fr.enabled else "Enable"
            toggle_icon = "HIDE_ON" if fr.enabled else "HIDE_OFF"
            op = layout.operator(
                "proscenium.timeline_strip_toggle_enabled",
                text=toggle_text,
                icon=toggle_icon,
            )
            op.index = self._hit_index

            # Regenerate just this range (deactivated — not yet supported by API)
            regen_row = layout.row()
            regen_row.enabled = False
            regen_row.operator(
                "proscenium.regenerate_block",
                text="Regenerate Range",
                icon="FILE_REFRESH",
            )

            layout.separator()

            # Delete
            layout.operator(
                "proscenium.timeline_strip_delete",
                text="Delete Strip",
                icon="TRASH",
            )

            layout.separator()

        # Add between keyframes (needs source armature with keyframes)
        has_kf = (
            props.target_armature
            and len(_get_armature_keyframes(props.target_armature)) >= 2
        )
        kf_row = layout.row()
        kf_row.enabled = has_kf
        op = kf_row.operator(
            "proscenium.add_strip_between_keyframes",
            text="Add Strip Between Keyframes",
            icon="KEYFRAME_HLT",
        )
        op.frame = self._click_frame

        # Add strip in first available gap
        layout.operator(
            "proscenium.add_prompt_block",
            text="Add Strip in Gap",
            icon="ADD",
        )


# ---------------------------------------------------------------------------
# Toggle strip enabled (for context menu)
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_timeline_strip_toggle_enabled(bpy.types.Operator):
    """Toggle enabled state of a strip."""

    bl_idname = "proscenium.timeline_strip_toggle_enabled"
    bl_label = "Toggle Strip Enabled"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(name="Strip Index", default=0)

    @classmethod
    def poll(cls, context):
        return _timeline_poll(context)

    def execute(self, context):
        props = context.scene.proscenium
        if 0 <= self.index < len(props.prompt_blocks):
            fr = props.prompt_blocks[self.index]
            fr.enabled = not fr.enabled
            state = "enabled" if fr.enabled else "disabled"
            self.report({"INFO"}, f"Strip '{fr.prompt}' {state}")

            # Force redraw all timeline areas — context.area may point
            # to the popup menu rather than the actual timeline editor.
            for area in context.screen.areas:
                if area.type == "DOPESHEET_EDITOR":
                    area.tag_redraw()
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Inline prompt editing (triggered by double-click on strip)
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_timeline_strip_inline_edit(bpy.types.Operator):
    """Edit strip prompt text directly on the timeline strip."""

    bl_idname = "proscenium.timeline_strip_inline_edit"
    bl_label = "Inline Edit Strip Prompt"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    index: IntProperty(name="Strip Index", default=0)

    @classmethod
    def poll(cls, context):
        return _timeline_poll(context)

    def invoke(self, context, event):
        props = context.scene.proscenium
        if not (0 <= self.index < len(props.prompt_blocks)):
            return {"CANCELLED"}

        fr = props.prompt_blocks[self.index]

        # Populate shared state that the draw callback reads
        inline_edit_state["active"] = True
        inline_edit_state["index"] = self.index
        inline_edit_state["text"] = fr.prompt
        inline_edit_state["cursor"] = len(fr.prompt)
        inline_edit_state["original"] = fr.prompt
        inline_edit_state["selection_start"] = None

        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    @staticmethod
    def _has_selection():
        s = inline_edit_state.get("selection_start")
        return s is not None and s != inline_edit_state["cursor"]

    @staticmethod
    def _get_selection_range():
        """Return (lo, hi) of current selection."""
        s = inline_edit_state["selection_start"]
        c = inline_edit_state["cursor"]
        return (min(s, c), max(s, c))

    @staticmethod
    def _delete_selection():
        """Delete selected text, update cursor, clear selection. Returns new (text, pos)."""
        lo, hi = PROSCENIUM_OT_timeline_strip_inline_edit._get_selection_range()
        text = inline_edit_state["text"]
        inline_edit_state["text"] = text[:lo] + text[hi:]
        inline_edit_state["cursor"] = lo
        inline_edit_state["selection_start"] = None
        return inline_edit_state["text"], lo

    def modal(self, context, event):
        if context.area is None:
            self._cancel(context)
            return {"CANCELLED"}

        # --- Confirm ---
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            self._commit(context)
            return {"FINISHED"}

        # --- Cancel ---
        if event.type == "ESC" and event.value == "PRESS":
            self._cancel(context)
            return {"CANCELLED"}

        # --- Click outside strip → confirm ---
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = hit_test_strips(
                context, event.mouse_region_x, event.mouse_region_y,
            )
            if hit["index"] != self.index:
                self._commit(context)
                return {"FINISHED"}
            return {"RUNNING_MODAL"}

        # Only act on key-press events from here on
        if event.value != "PRESS":
            return {"RUNNING_MODAL"}

        text = inline_edit_state["text"]
        pos = inline_edit_state["cursor"]
        handled = True

        if event.type == "BACK_SPACE":
            if self._has_selection():
                self._delete_selection()
            elif event.ctrl or event.oskey:
                # Delete word before cursor
                i = pos
                while i > 0 and text[i - 1] == " ":
                    i -= 1
                while i > 0 and text[i - 1] != " ":
                    i -= 1
                inline_edit_state["text"] = text[:i] + text[pos:]
                inline_edit_state["cursor"] = i
            elif pos > 0:
                inline_edit_state["text"] = text[:pos - 1] + text[pos:]
                inline_edit_state["cursor"] = pos - 1
            inline_edit_state["selection_start"] = None

        elif event.type == "DEL":
            if self._has_selection():
                self._delete_selection()
            elif pos < len(text):
                inline_edit_state["text"] = text[:pos] + text[pos + 1:]
            inline_edit_state["selection_start"] = None

        elif event.type == "LEFT_ARROW":
            if event.shift:
                # Extend selection
                if inline_edit_state["selection_start"] is None:
                    inline_edit_state["selection_start"] = pos
            else:
                # Clear selection; jump to selection start if active
                if self._has_selection():
                    lo, _ = self._get_selection_range()
                    inline_edit_state["cursor"] = lo
                    inline_edit_state["selection_start"] = None
                    context.area.tag_redraw()
                    return {"RUNNING_MODAL"}
                inline_edit_state["selection_start"] = None

            if event.ctrl or event.oskey:
                i = pos
                while i > 0 and text[i - 1] == " ":
                    i -= 1
                while i > 0 and text[i - 1] != " ":
                    i -= 1
                inline_edit_state["cursor"] = i
            else:
                inline_edit_state["cursor"] = max(0, pos - 1)

        elif event.type == "RIGHT_ARROW":
            if event.shift:
                if inline_edit_state["selection_start"] is None:
                    inline_edit_state["selection_start"] = pos
            else:
                if self._has_selection():
                    _, hi = self._get_selection_range()
                    inline_edit_state["cursor"] = hi
                    inline_edit_state["selection_start"] = None
                    context.area.tag_redraw()
                    return {"RUNNING_MODAL"}
                inline_edit_state["selection_start"] = None

            if event.ctrl or event.oskey:
                i = pos
                while i < len(text) and text[i] == " ":
                    i += 1
                while i < len(text) and text[i] != " ":
                    i += 1
                inline_edit_state["cursor"] = i
            else:
                inline_edit_state["cursor"] = min(len(text), pos + 1)

        elif event.type == "HOME":
            if event.shift:
                if inline_edit_state["selection_start"] is None:
                    inline_edit_state["selection_start"] = pos
            else:
                inline_edit_state["selection_start"] = None
            inline_edit_state["cursor"] = 0

        elif event.type == "END":
            if event.shift:
                if inline_edit_state["selection_start"] is None:
                    inline_edit_state["selection_start"] = pos
            else:
                inline_edit_state["selection_start"] = None
            inline_edit_state["cursor"] = len(text)

        elif event.type == "A" and (event.ctrl or event.oskey):
            # Select all
            inline_edit_state["selection_start"] = 0
            inline_edit_state["cursor"] = len(text)

        elif event.type == "V" and (event.ctrl or event.oskey):
            # Paste from clipboard (replace selection if active)
            if self._has_selection():
                text, pos = self._delete_selection()
            clipboard = context.window_manager.clipboard or ""
            clipboard = clipboard.replace("\n", " ").replace("\r", "")
            inline_edit_state["text"] = text[:pos] + clipboard + text[pos:]
            inline_edit_state["cursor"] = pos + len(clipboard)
            inline_edit_state["selection_start"] = None

        elif event.type == "C" and (event.ctrl or event.oskey):
            # Copy selected text (or full text if no selection)
            if self._has_selection():
                lo, hi = self._get_selection_range()
                context.window_manager.clipboard = text[lo:hi]
            else:
                context.window_manager.clipboard = text

        elif event.type == "X" and (event.ctrl or event.oskey):
            # Cut selected text
            if self._has_selection():
                lo, hi = self._get_selection_range()
                context.window_manager.clipboard = text[lo:hi]
                self._delete_selection()

        elif event.type == "TAB":
            # Tab confirms like Enter
            self._commit(context)
            return {"FINISHED"}

        elif event.unicode and event.unicode.isprintable():
            # Replace selection if active, then insert character
            if self._has_selection():
                text, pos = self._delete_selection()
            inline_edit_state["text"] = text[:pos] + event.unicode + text[pos:]
            inline_edit_state["cursor"] = pos + len(event.unicode)
            inline_edit_state["selection_start"] = None

        else:
            handled = False

        if handled:
            context.area.tag_redraw()

        return {"RUNNING_MODAL"}

    # -- helpers --

    def _commit(self, context):
        """Save edited text to the frame range and exit edit mode."""
        props = context.scene.proscenium
        idx = inline_edit_state["index"]
        if 0 <= idx < len(props.prompt_blocks):
            props.prompt_blocks[idx].prompt = inline_edit_state["text"]
        inline_edit_state["active"] = False
        context.area.tag_redraw()

    def _cancel(self, context):
        """Revert text and exit edit mode."""
        inline_edit_state["active"] = False
        context.area.tag_redraw()

    def cancel(self, context):
        self._cancel(context)


# ---------------------------------------------------------------------------
# Prompt editing popup (right-click menu → Edit Prompt)
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_edit_strip_prompt(bpy.types.Operator):
    """Edit the prompt text for a frame range strip."""

    bl_idname = "proscenium.edit_strip_prompt"
    bl_label = "Edit Strip Prompt"
    bl_options = {"REGISTER", "UNDO"}

    index: IntProperty(name="Strip Index", default=0)
    prompt: StringProperty(name="Prompt", default="")

    def invoke(self, context, event):
        props = context.scene.proscenium
        if 0 <= self.index < len(props.prompt_blocks):
            self.prompt = props.prompt_blocks[self.index].prompt
        # Mark that we went through invoke (dialog will be shown)
        self._from_dialog = True
        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        props = context.scene.proscenium

        if 0 <= self.index < len(props.prompt_blocks):
            fr = props.prompt_blocks[self.index]
            layout.label(
                text=f"Range {self.index + 1}:  frames {fr.frame_start} – {fr.frame_end}"
            )

        layout.prop(self, "prompt", text="Prompt")

    def execute(self, context):
        # When called from a popup menu, Blender calls execute() directly
        # instead of invoke(), so the dialog never opens. Detect this and
        # defer to invoke via a timer (menu must close first).
        if not getattr(self, "_from_dialog", False):
            idx = self.index

            def _open_dialog():
                try:
                    bpy.ops.proscenium.edit_strip_prompt(
                        "INVOKE_DEFAULT", index=idx
                    )
                except Exception:
                    pass
                return None  # don't repeat

            bpy.app.timers.register(_open_dialog, first_interval=0.1)
            return {"FINISHED"}

        # Normal path — dialog was shown, save the prompt
        self._from_dialog = False
        props = context.scene.proscenium
        if 0 <= self.index < len(props.prompt_blocks):
            props.prompt_blocks[self.index].prompt = self.prompt
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Add / remove / regenerate (hooked by header buttons and context menu)
# ---------------------------------------------------------------------------

class PROSCENIUM_OT_add_prompt_block(bpy.types.Operator):
    """Add a new prompt block in the first available gap."""

    bl_idname = "proscenium.add_prompt_block"
    bl_label = "Add Prompt Block"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.proscenium
        scene_end = context.scene.frame_end or 250
        gap = find_gap(props.prompt_blocks, min_length=10, scene_end=scene_end)
        if gap is None:
            self.report({'WARNING'}, "No room on the timeline for a new block")
            return {'CANCELLED'}

        block = props.prompt_blocks.add()
        block.prompt = ""
        block.frame_start = gap[0]
        block.frame_end = gap[1]
        block.enabled = True
        props.active_block_index = len(props.prompt_blocks) - 1

        # Persist to armature
        from .properties import save_blocks_to_armature
        save_blocks_to_armature(props.target_armature, props)
        return {'FINISHED'}


class PROSCENIUM_OT_remove_prompt_block(bpy.types.Operator):
    """Remove the active prompt block."""

    bl_idname = "proscenium.remove_prompt_block"
    bl_label = "Remove Prompt Block"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            hasattr(context.scene, "proscenium")
            and len(context.scene.proscenium.prompt_blocks) > 0
        )

    def execute(self, context):
        props = context.scene.proscenium
        if 0 <= props.active_block_index < len(props.prompt_blocks):
            props.prompt_blocks.remove(props.active_block_index)
            props.active_block_index = max(0, min(
                props.active_block_index, len(props.prompt_blocks) - 1,
            ))
            from .properties import save_blocks_to_armature
            save_blocks_to_armature(props.target_armature, props)
        return {'FINISHED'}


class PROSCENIUM_OT_regenerate_block(bpy.types.Operator):
    """Regenerate just the active block.  Placeholder — currently just fires the
    full-motion generate; per-block regen needs server-side support."""

    bl_idname = "proscenium.regenerate_block"
    bl_label = "Regenerate Block"

    def execute(self, context):
        bpy.ops.proscenium.generate('INVOKE_DEFAULT')
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Timeline header append (+ / - buttons only)
# ---------------------------------------------------------------------------

def draw_timeline_header(self, context):
    """Appended to DOPESHEET_HT_header — adds Proscenium strip controls."""
    if not hasattr(context, "space_data") or context.space_data is None:
        return
    if context.space_data.mode != "TIMELINE":
        return

    layout = self.layout
    layout.separator()
    layout.operator("proscenium.add_prompt_block", text="", icon="ADD")
    layout.operator("proscenium.remove_prompt_block", text="", icon="REMOVE")


# ---------------------------------------------------------------------------
# Keymap registration
# ---------------------------------------------------------------------------

def register_keymaps():
    """Register keyboard/mouse handlers in the Timeline (DopeSheet) editor."""
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon is None:
        return

    km = wm.keyconfigs.addon.keymaps.new(
        name="Dopesheet", space_type="DOPESHEET_EDITOR"
    )

    # Left-click — strip select/drag (also handles lane resize)
    kmi = km.keymap_items.new(
        "proscenium.timeline_strip_action",
        type="LEFTMOUSE",
        value="PRESS",
    )
    _addon_keymaps.append((km, kmi))

    # Double-click on empty — add strip (uses same LEFTMOUSE but the
    # operator internally tracks double-click timing)
    kmi = km.keymap_items.new(
        "proscenium.timeline_strip_add_click",
        type="LEFTMOUSE",
        value="PRESS",
    )
    _addon_keymaps.append((km, kmi))

    # Right-click — context menu
    kmi = km.keymap_items.new(
        "proscenium.timeline_strip_context_menu",
        type="RIGHTMOUSE",
        value="PRESS",
    )
    _addon_keymaps.append((km, kmi))

    # Delete key — remove active strip
    kmi = km.keymap_items.new(
        "proscenium.timeline_strip_delete",
        type="DEL",
        value="PRESS",
    )
    _addon_keymaps.append((km, kmi))

    # Backspace — remove active strip
    kmi = km.keymap_items.new(
        "proscenium.timeline_strip_delete",
        type="BACK_SPACE",
        value="PRESS",
    )
    _addon_keymaps.append((km, kmi))


def unregister_keymaps():
    for km, kmi in _addon_keymaps:
        km.keymap_items.remove(kmi)
    _addon_keymaps.clear()


# ---------------------------------------------------------------------------
# Class registration
# ---------------------------------------------------------------------------

_classes = (
    PROSCENIUM_OT_timeline_strip_action,
    PROSCENIUM_OT_timeline_strip_add_click,
    PROSCENIUM_OT_add_strip_between_keyframes,
    PROSCENIUM_OT_timeline_strip_delete,
    PROSCENIUM_OT_timeline_strip_context_menu,
    PROSCENIUM_OT_timeline_strip_toggle_enabled,
    PROSCENIUM_OT_timeline_strip_inline_edit,
    PROSCENIUM_OT_edit_strip_prompt,
    PROSCENIUM_OT_add_prompt_block,
    PROSCENIUM_OT_remove_prompt_block,
    PROSCENIUM_OT_regenerate_block,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.DOPESHEET_HT_header.append(draw_timeline_header)
    register_keymaps()


def unregister():
    unregister_keymaps()
    try:
        bpy.types.DOPESHEET_HT_header.remove(draw_timeline_header)
    except Exception:
        pass
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
