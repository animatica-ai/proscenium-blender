"""Timeline strip overlay for Animatica frame ranges.

Draws colored strips on the Blender Timeline editor representing
PromptBlock items. Each strip shows its prompt text and can
be interacted with via the modal operator in timeline_operators.py.
"""

import bpy
import gpu
import blf
from gpu_extras.batch import batch_for_shader


# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------

_strip_height = 56           # pixels tall per strip on first load (mutable — drag-resizable)
DEFAULT_STRIP_HEIGHT = 28    # font-scale reference: at this height the label renders at TEXT_SIZE.
                              # Default _strip_height is 2× this so the prompt reads comfortably
                              # without the user having to drag-resize on first launch.
MIN_STRIP_HEIGHT = 22        # below this the prompt label clips into the strip border
MAX_STRIP_HEIGHT = 120
RESIZE_HANDLE_HEIGHT = 5     # pixels — draggable zone at top of lane border

STRIP_Y_OFFSET = 6          # pixels from bottom of timeline region
EDGE_HANDLE_WIDTH = 6       # pixels — draggable edge zone width
TEXT_PADDING_X = 6           # horizontal text padding inside strip
TEXT_SIZE = 11               # blf font size (at default height)

# Track lane background
LANE_PADDING = 3            # extra padding around strip area
LANE_BG_COLOR = (0.12, 0.12, 0.12, 0.85)
LANE_BORDER_COLOR = (0.25, 0.25, 0.25, 0.6)
LABEL_COLOR = (0.55, 0.55, 0.55, 0.7)
LABEL_SIZE = 10

# Color palette (RGBA) — cycled per strip index
STRIP_COLORS = [
    (0.267, 0.541, 0.878, 0.75),   # Blue
    (0.878, 0.435, 0.267, 0.75),   # Orange
    (0.365, 0.737, 0.400, 0.75),   # Green
    (0.729, 0.333, 0.729, 0.75),   # Purple
    (0.878, 0.722, 0.267, 0.75),   # Yellow
    (0.267, 0.796, 0.796, 0.75),   # Teal
]

ACTIVE_BORDER_COLOR = (1.0, 1.0, 1.0, 0.9)
INACTIVE_BORDER_COLOR = (0.3, 0.3, 0.3, 0.5)
DISABLED_ALPHA_MULTIPLIER = 0.3
TEXT_COLOR = (1.0, 1.0, 1.0, 0.95)
TEXT_COLOR_DISABLED = (0.7, 0.7, 0.7, 0.4)

# Frame number styling
FRAME_NUM_SIZE = 9
FRAME_NUM_PADDING = 5          # padding from strip edge for frame numbers
FRAME_NUM_COLOR = (0.9, 0.9, 0.9, 0.65)
FRAME_NUM_COLOR_DISABLED = (0.6, 0.6, 0.6, 0.3)

# Inline-edit cursor
CURSOR_COLOR = (1.0, 1.0, 1.0, 0.85)

# Inline-edit state (written by timeline_operators, read by draw callback)
inline_edit_state = {
    "active": False,
    "index": -1,
    "text": "",
    "cursor": 0,
    "original": "",
    "selection_start": None,  # None = no selection, int = start of selection range
}

# Module-level draw-handler storage.  Mirrored onto bpy.app.driver_namespace
# so the handle survives module reloads — otherwise a reload (sys.modules pop
# + re-import) resets our local to None while Blender still holds the old
# handle, and register_draw_handler() would add a second one → duplicate draw.
_NS_KEY = "_proscenium_timeline_overlay_handle"
_draw_handle = None


# ---------------------------------------------------------------------------
# Strip height accessors (for drag-resize from timeline_operators)
# ---------------------------------------------------------------------------

def get_strip_height():
    """Return the current strip lane height (pixels)."""
    return _strip_height


def set_strip_height(h):
    """Set strip lane height, clamped to [MIN, MAX]."""
    global _strip_height
    _strip_height = max(MIN_STRIP_HEIGHT, min(MAX_STRIP_HEIGHT, int(h)))


def _scaled_font_size(base_size):
    """Scale font size proportionally to strip height."""
    scale = _strip_height / DEFAULT_STRIP_HEIGHT
    return max(7, int(base_size * scale + 0.5))


# ---------------------------------------------------------------------------
# Overlap helpers (single-track — strips must not overlap)
# ---------------------------------------------------------------------------

def get_sorted_blocks(prompt_blocks):
    """Return list of (index, frame_start, frame_end) sorted by frame_start."""
    items = [(i, fr.frame_start, fr.frame_end) for i, fr in enumerate(prompt_blocks)]
    items.sort(key=lambda x: x[1])
    return items


def find_neighbors(prompt_blocks, idx):
    """Find the left and right neighbor strips for strip *idx*.

    Returns (left_end, right_start) — the frame boundaries imposed by
    neighbors.  ``left_end`` is the frame_end of the nearest strip to
    the left (or 0 if none).  ``right_start`` is the frame_start of the
    nearest strip to the right (or None if none).
    """
    sorted_items = get_sorted_blocks(prompt_blocks)
    fr = prompt_blocks[idx]

    left_end = 0
    right_start = None

    for si, s_start, s_end in sorted_items:
        if si == idx:
            continue
        # Left neighbor: ends before or at our start
        if s_end <= fr.frame_start:
            left_end = max(left_end, s_end)
        # Right neighbor: starts at or after our end
        if s_start >= fr.frame_end:
            if right_start is None or s_start < right_start:
                right_start = s_start

    return left_end, right_start


def find_gap(prompt_blocks, min_length=10, scene_end=250):
    """Find the first gap on the timeline where a new strip can fit.

    Returns (gap_start, gap_end) or None if no room.
    """
    sorted_items = get_sorted_blocks(prompt_blocks)
    cursor = 1  # earliest possible frame

    for _idx, s_start, s_end in sorted_items:
        if s_start - cursor >= min_length:
            return (cursor, s_start)
        cursor = max(cursor, s_end)

    # Gap after all existing strips
    if scene_end - cursor >= min_length:
        return (cursor, min(cursor + 50, scene_end))
    # Force-fit even if tiny
    if cursor < scene_end:
        return (cursor, scene_end)
    return None


def blocks_overlap(prompt_blocks, exclude_idx=-1):
    """Return True if any two strips (excluding *exclude_idx*) overlap."""
    items = [(fr.frame_start, fr.frame_end)
             for i, fr in enumerate(prompt_blocks) if i != exclude_idx]
    items.sort()
    for i in range(len(items) - 1):
        if items[i][1] > items[i + 1][0]:
            return True
    return False


# ---------------------------------------------------------------------------
# Hit-testing (used by timeline_operators.py)
# ---------------------------------------------------------------------------

def hit_test_strips(context, mouse_x, mouse_y):
    """Determine which strip (if any) is under the mouse cursor.

    Returns dict: {'index': int|None, 'zone': str|None}
        zone: 'body', 'edge_start', 'edge_end', or None
    """
    scene = context.scene
    if not hasattr(scene, "proscenium"):
        return {"index": None, "zone": None}

    props = scene.proscenium
    if len(props.prompt_blocks) == 0:
        return {"index": None, "zone": None}

    view2d = context.region.view2d

    y_bottom = STRIP_Y_OFFSET
    y_top = y_bottom + _strip_height

    for i, fr in enumerate(props.prompt_blocks):
        x_start, _ = view2d.view_to_region(fr.frame_start, 0, clip=False)
        x_end, _ = view2d.view_to_region(fr.frame_end, 0, clip=False)

        # Y check
        if mouse_y < y_bottom or mouse_y > y_top:
            continue

        # X check (with edge tolerance)
        if mouse_x < x_start - EDGE_HANDLE_WIDTH or mouse_x > x_end + EDGE_HANDLE_WIDTH:
            continue

        # Determine zone
        if abs(mouse_x - x_start) <= EDGE_HANDLE_WIDTH:
            return {"index": i, "zone": "edge_start"}
        elif abs(mouse_x - x_end) <= EDGE_HANDLE_WIDTH:
            return {"index": i, "zone": "edge_end"}
        else:
            return {"index": i, "zone": "body"}

    return {"index": None, "zone": None}


def hit_test_lane_resize(context, mouse_x, mouse_y):
    """Return True if mouse is in the lane top-border resize zone."""
    lane_top = STRIP_Y_OFFSET + _strip_height + LANE_PADDING
    return abs(mouse_y - lane_top) <= RESIZE_HANDLE_HEIGHT


def pixel_to_frame(context, pixel_x):
    """Convert a region-pixel X coordinate to a frame number."""
    frame, _ = context.region.view2d.region_to_view(pixel_x, 0)
    return round(frame)


# ---------------------------------------------------------------------------
# Helpers for strip color resolution
# ---------------------------------------------------------------------------

def _is_unconditioned(fr) -> bool:
    """A block with no prompt becomes an MMCP UnconditionedSegment. We render
    those differently so the user can tell at a glance which spans the model
    is filling on its own vs. text-driven spans."""
    return not (fr.prompt or "").strip()


def _strip_color(fr, index):
    """Return RGBA tuple for a frame range strip."""
    # Custom colour (if any channel > 0)
    if hasattr(fr, "color") and any(c > 0 for c in fr.color):
        color = list(fr.color)
    else:
        color = list(STRIP_COLORS[index % len(STRIP_COLORS)])

    if _is_unconditioned(fr):
        # Mute toward neutral gray so the block reads as "no prompt here".
        gray = 0.45
        color[0] = gray * 0.6 + color[0] * 0.4
        color[1] = gray * 0.6 + color[1] * 0.4
        color[2] = gray * 0.6 + color[2] * 0.4
        color[3] *= 0.85

    if not fr.enabled:
        # Desaturate: mix RGB channels towards their mean (gray)
        gray = sum(color[:3]) / 3.0
        color[0] = gray * 0.7 + color[0] * 0.3
        color[1] = gray * 0.7 + color[1] * 0.3
        color[2] = gray * 0.7 + color[2] * 0.3
        color[3] *= DISABLED_ALPHA_MULTIPLIER

    return color


def _strip_border_color(fr, is_active):
    """Return border color for a strip."""
    if is_active:
        return ACTIVE_BORDER_COLOR
    return INACTIVE_BORDER_COLOR


# ---------------------------------------------------------------------------
# GPU draw callback
# ---------------------------------------------------------------------------

def draw_timeline_strips():
    """POST_PIXEL draw callback registered on SpaceDopeSheetEditor."""
    context = bpy.context

    # Guard: only draw in Timeline mode
    if not hasattr(context, "space_data") or context.space_data is None:
        return
    if context.space_data.type != "DOPESHEET_EDITOR":
        return
    if context.space_data.mode != "TIMELINE":
        return

    scene = context.scene
    if not hasattr(scene, "proscenium"):
        return

    props = scene.proscenium
    if len(props.prompt_blocks) == 0:
        return

    region = context.region
    # Metal (macOS) limits textures to 16384px. Skip drawing on ultra-wide regions
    # to avoid MTLTextureDescriptor validation crash.
    if region.width > 16384 or region.height > 16384:
        return

    view2d = region.view2d

    y_bottom = STRIP_Y_OFFSET
    y_top = y_bottom + _strip_height

    # Prepare GPU state
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")

    # --- Lane background (dark band spanning full width) ---
    lane_y0 = y_bottom - LANE_PADDING
    lane_y1 = y_top + LANE_PADDING
    lane_verts = (
        (0, lane_y0), (region.width, lane_y0),
        (region.width, lane_y1), (0, lane_y1),
    )
    lane_idx = ((0, 1, 2), (0, 2, 3))
    lane_batch = batch_for_shader(shader, "TRIS", {"pos": lane_verts}, indices=lane_idx)
    shader.uniform_float("color", LANE_BG_COLOR)
    lane_batch.draw(shader)

    # Top border line
    border_verts = ((0, lane_y1), (region.width, lane_y1))
    border_batch = batch_for_shader(shader, "LINES", {"pos": border_verts})
    shader.uniform_float("color", LANE_BORDER_COLOR)
    border_batch.draw(shader)

    # Resize grip indicator (subtle dots at top border centre)
    _draw_resize_handle(shader, region.width, lane_y1)

    # "PROSCENIUM" label on the left
    _draw_lane_label(lane_y0, lane_y1)

    for i, fr in enumerate(props.prompt_blocks):
        is_active = (i == props.active_block_index)
        is_editing = (inline_edit_state["active"]
                      and inline_edit_state["index"] == i)

        # Frame → pixel conversion
        x_start, _ = view2d.view_to_region(fr.frame_start, 0, clip=False)
        x_end, _ = view2d.view_to_region(fr.frame_end, 0, clip=False)

        # Skip if completely off-screen
        if x_end < 0 or x_start > region.width:
            continue

        # Clamp to region bounds for drawing
        x_start_draw = max(x_start, 0)
        x_end_draw = min(x_end, region.width)

        color = _strip_color(fr, i)

        # --- Filled rectangle ---
        verts = (
            (x_start_draw, y_bottom),
            (x_end_draw, y_bottom),
            (x_end_draw, y_top),
            (x_start_draw, y_top),
        )
        indices = ((0, 1, 2), (0, 2, 3))
        batch = batch_for_shader(shader, "TRIS", {"pos": verts}, indices=indices)
        shader.uniform_float("color", color)
        batch.draw(shader)

        # --- Border ---
        border_color = _strip_border_color(fr, is_active)
        border_batch = batch_for_shader(shader, "LINE_LOOP", {"pos": verts})
        shader.uniform_float("color", border_color)
        gpu.state.line_width_set(2.0 if is_active else 1.0)
        border_batch.draw(shader)
        gpu.state.line_width_set(1.0)

        # --- Edge handles (brighter vertical bars at start/end) ---
        handle_color = [min(1.0, c * 1.3) for c in color[:3]] + [0.9]
        hw = 2  # half-width
        for ex in (x_start, x_end):
            # Only draw if visible
            if ex < -hw or ex > region.width + hw:
                continue
            hverts = (
                (ex - hw, y_bottom), (ex + hw, y_bottom),
                (ex + hw, y_top), (ex - hw, y_top),
            )
            hindices = ((0, 1, 2), (0, 2, 3))
            hbatch = batch_for_shader(shader, "TRIS", {"pos": hverts}, indices=hindices)
            shader.uniform_float("color", handle_color)
            hbatch.draw(shader)

        # --- Disabled / unconditioned hash lines ---
        if not fr.enabled:
            _draw_disabled_hash(shader, x_start_draw, x_end_draw, y_bottom, y_top)
        elif _is_unconditioned(fr):
            _draw_unconditioned_hash(shader, x_start_draw, x_end_draw, y_bottom, y_top)

        # --- Text ---
        if is_editing:
            _draw_strip_text_editing(
                inline_edit_state["text"],
                inline_edit_state["cursor"],
                x_start_draw, x_end_draw, y_bottom, y_top,
                frame_start=fr.frame_start,
                frame_end=fr.frame_end,
                shader=shader,
            )
        else:
            label = (
                "unconditioned"
                if _is_unconditioned(fr)
                else fr.prompt
            )
            _draw_strip_text(
                label,
                x_start_draw, x_end_draw, y_bottom, y_top,
                disabled=not fr.enabled,
                frame_start=fr.frame_start,
                frame_end=fr.frame_end,
            )

    # Restore GPU state
    gpu.state.blend_set("NONE")


def _draw_lane_label(lane_y0, lane_y1):
    """Draw 'PROSCENIUM' label at the left edge of the lane."""
    font_id = 0
    blf.size(font_id, _scaled_font_size(LABEL_SIZE))
    tw, th = blf.dimensions(font_id, "PROSCENIUM")
    label_x = 4
    label_y = lane_y0 + (lane_y1 - lane_y0 - th) / 2
    blf.position(font_id, label_x, label_y, 0)
    blf.color(font_id, *LABEL_COLOR)
    blf.draw(font_id, "PROSCENIUM")


def _draw_resize_handle(shader, region_width, lane_y1):
    """Draw subtle resize grip dots at the top edge of the lane."""
    grip_color = (0.45, 0.45, 0.45, 0.5)
    cx = region_width / 2
    # Three pairs of small horizontal dashes
    lines = []
    for dx in (-10, 0, 10):
        x = cx + dx
        lines.extend([(x - 3, lane_y1 - 1), (x + 3, lane_y1 - 1)])
        lines.extend([(x - 3, lane_y1 + 1), (x + 3, lane_y1 + 1)])
    batch = batch_for_shader(shader, "LINES", {"pos": lines})
    shader.uniform_float("color", grip_color)
    batch.draw(shader)


def _draw_disabled_hash(shader, x_start, x_end, y_bottom, y_top):
    """Draw diagonal hash lines over a disabled strip for clear visual feedback."""
    hash_color = (0.5, 0.5, 0.5, 0.25)
    spacing = 12  # pixels between hash lines
    strip_w = x_end - x_start
    strip_h = y_top - y_bottom

    lines = []
    # Diagonal lines from bottom-left to top-right
    x = -strip_h  # start before the strip to cover the left edge
    while x < strip_w + strip_h:
        x0 = x_start + x
        y0 = y_bottom
        x1 = x_start + x + strip_h
        y1 = y_top

        # Clamp to strip bounds
        if x1 > x_end:
            dy = x1 - x_end
            x1 = x_end
            y1 = y_top - dy
        if x0 < x_start:
            dy = x_start - x0
            x0 = x_start
            y0 = y_bottom + dy

        if x0 < x_end and x1 > x_start and y0 < y_top and y1 > y_bottom:
            lines.extend([(x0, y0), (x1, y1)])

        x += spacing

    if lines:
        hash_batch = batch_for_shader(shader, "LINES", {"pos": lines})
        shader.uniform_float("color", hash_color)
        hash_batch.draw(shader)


def _draw_unconditioned_hash(shader, x_start, x_end, y_bottom, y_top):
    """Draw diagonal hash lines over an unconditioned (no-prompt) strip.

    Uses the *opposite* slope from the disabled hash so the two states are
    visually distinct (unconditioned slants top-left → bottom-right; disabled
    slants bottom-left → top-right). The colour is a faint blue to suggest
    'model is filling this in' rather than 'this is off'.
    """
    hash_color = (0.40, 0.55, 0.85, 0.28)
    spacing    = 14
    strip_w    = x_end - x_start
    strip_h    = y_top - y_bottom

    lines = []
    x = -strip_h
    while x < strip_w + strip_h:
        # Reverse-slope: from top-left to bottom-right
        x0 = x_start + x
        y0 = y_top
        x1 = x_start + x + strip_h
        y1 = y_bottom

        if x1 > x_end:
            dy = x1 - x_end
            x1 = x_end
            y1 = y_bottom + dy
        if x0 < x_start:
            dy = x_start - x0
            x0 = x_start
            y0 = y_top - dy

        if x0 < x_end and x1 > x_start and y0 > y_bottom and y1 < y_top:
            lines.extend([(x0, y0), (x1, y1)])
        x += spacing

    if lines:
        batch = batch_for_shader(shader, "LINES", {"pos": lines})
        shader.uniform_float("color", hash_color)
        batch.draw(shader)


def _draw_strip_text(text, x_start, x_end, y_bottom, y_top, disabled=False,
                     frame_start=None, frame_end=None):
    """Render frame numbers at edges and truncated prompt text in the centre."""
    font_id = 0
    strip_width = x_end - x_start

    if strip_width < 16:
        return  # too narrow for anything

    num_color = FRAME_NUM_COLOR_DISABLED if disabled else FRAME_NUM_COLOR

    # -- Draw frame numbers at edges --
    left_reserved = 0.0
    right_reserved = 0.0

    if frame_start is not None:
        blf.size(font_id, _scaled_font_size(FRAME_NUM_SIZE))
        start_str = str(frame_start)
        sw, sh = blf.dimensions(font_id, start_str)
        need = sw + FRAME_NUM_PADDING * 2
        if need < strip_width * 0.45:
            ny = y_bottom + (_strip_height - sh) / 2
            blf.position(font_id, x_start + FRAME_NUM_PADDING, ny, 0)
            blf.color(font_id, *num_color)
            blf.draw(font_id, start_str)
            left_reserved = need

    if frame_end is not None:
        blf.size(font_id, _scaled_font_size(FRAME_NUM_SIZE))
        end_str = str(frame_end)
        ew, eh = blf.dimensions(font_id, end_str)
        need = ew + FRAME_NUM_PADDING * 2
        if need < strip_width * 0.45:
            ny = y_bottom + (_strip_height - eh) / 2
            blf.position(font_id, x_end - ew - FRAME_NUM_PADDING, ny, 0)
            blf.color(font_id, *num_color)
            blf.draw(font_id, end_str)
            right_reserved = need

    # -- Draw prompt text centred in the remaining space --
    prompt_x_start = x_start + left_reserved
    prompt_x_end = x_end - right_reserved
    available_width = prompt_x_end - prompt_x_start - (TEXT_PADDING_X * 2)

    if available_width < 20:
        return  # no room for prompt

    blf.size(font_id, _scaled_font_size(TEXT_SIZE))

    display_text = text
    tw, th = blf.dimensions(font_id, display_text)
    while tw > available_width and len(display_text) > 4:
        display_text = display_text[:-4] + "..."
        tw, th = blf.dimensions(font_id, display_text)

    # Centre the prompt between the reserved edges
    text_x = prompt_x_start + TEXT_PADDING_X + (available_width - tw) / 2
    text_y = y_bottom + (_strip_height - th) / 2

    blf.position(font_id, text_x, text_y, 0)
    color = TEXT_COLOR_DISABLED if disabled else TEXT_COLOR
    blf.color(font_id, *color)
    blf.draw(font_id, display_text)


def _draw_strip_text_editing(text, cursor_pos, x_start, x_end, y_bottom, y_top,
                             frame_start=None, frame_end=None, shader=None):
    """Render editable text with a cursor and optional selection highlight."""
    font_id = 0
    strip_width = x_end - x_start

    if strip_width < 16:
        return

    # -- Frame numbers at edges (same as normal drawing) --
    left_reserved = 0.0
    right_reserved = 0.0

    if frame_start is not None:
        blf.size(font_id, _scaled_font_size(FRAME_NUM_SIZE))
        start_str = str(frame_start)
        sw, sh = blf.dimensions(font_id, start_str)
        need = sw + FRAME_NUM_PADDING * 2
        if need < strip_width * 0.45:
            ny = y_bottom + (_strip_height - sh) / 2
            blf.position(font_id, x_start + FRAME_NUM_PADDING, ny, 0)
            blf.color(font_id, *FRAME_NUM_COLOR)
            blf.draw(font_id, start_str)
            left_reserved = need

    if frame_end is not None:
        blf.size(font_id, _scaled_font_size(FRAME_NUM_SIZE))
        end_str = str(frame_end)
        ew, eh = blf.dimensions(font_id, end_str)
        need = ew + FRAME_NUM_PADDING * 2
        if need < strip_width * 0.45:
            ny = y_bottom + (_strip_height - eh) / 2
            blf.position(font_id, x_end - ew - FRAME_NUM_PADDING, ny, 0)
            blf.color(font_id, *FRAME_NUM_COLOR)
            blf.draw(font_id, end_str)
            right_reserved = need

    # -- Editable text area --
    area_left = x_start + left_reserved + TEXT_PADDING_X
    area_right = x_end - right_reserved - TEXT_PADDING_X
    area_width = area_right - area_left

    if area_width < 10:
        return

    blf.size(font_id, _scaled_font_size(TEXT_SIZE))
    _, th = blf.dimensions(font_id, "Ay")  # stable reference height
    text_y = y_bottom + (_strip_height - th) / 2

    # Build the visible portion of text around the cursor.
    # We trim characters from the left when text overflows so
    # that the cursor always stays in view.
    display_text = text
    cursor_in_display = cursor_pos  # cursor offset within display_text
    left_trim = 0

    w_full, _ = blf.dimensions(font_id, display_text)
    if w_full > area_width:
        # Text overflows — find a window around the cursor.
        # 1) Trim from the left so cursor is visible
        w_before, _ = blf.dimensions(font_id, text[:cursor_pos])
        if w_before > area_width - 20:
            # Walk forward trimming chars until cursor fits
            target = w_before - area_width + 40
            trimmed_w = 0.0
            while left_trim < cursor_pos:
                cw, _ = blf.dimensions(font_id, text[left_trim])
                trimmed_w += cw
                left_trim += 1
                if trimmed_w >= target:
                    break

        display_text = text[left_trim:]
        cursor_in_display = cursor_pos - left_trim

        # 2) Truncate from the right if still too wide
        dw, _ = blf.dimensions(font_id, display_text)
        while dw > area_width and len(display_text) > 1:
            display_text = display_text[:-1]
            dw, _ = blf.dimensions(font_id, display_text)

    # -- Selection highlight (drawn before text so text is on top) --
    sel_start = inline_edit_state.get("selection_start")
    if shader is not None and sel_start is not None and sel_start != cursor_pos:
        sel_a = min(sel_start, cursor_pos) - left_trim
        sel_b = max(sel_start, cursor_pos) - left_trim
        sel_a = max(0, min(sel_a, len(display_text)))
        sel_b = max(0, min(sel_b, len(display_text)))
        if sel_a < sel_b:
            w_sel_a, _ = blf.dimensions(font_id, display_text[:sel_a])
            w_sel_b, _ = blf.dimensions(font_id, display_text[:sel_b])
            sx0 = max(area_left, area_left + w_sel_a)
            sx1 = min(area_right, area_left + w_sel_b)
            if sx1 > sx0:
                sel_verts = (
                    (sx0, y_bottom + 2), (sx1, y_bottom + 2),
                    (sx1, y_top - 2), (sx0, y_top - 2),
                )
                sel_idx = ((0, 1, 2), (0, 2, 3))
                sel_batch = batch_for_shader(
                    shader, "TRIS", {"pos": sel_verts}, indices=sel_idx
                )
                shader.uniform_float("color", (0.3, 0.5, 0.8, 0.45))
                sel_batch.draw(shader)

    # Draw the visible text left-aligned in the area
    blf.position(font_id, area_left, text_y, 0)
    blf.color(font_id, *TEXT_COLOR)
    blf.draw(font_id, display_text)

    # -- Cursor (vertical line) --
    if shader is not None:
        before_cursor = display_text[:cursor_in_display]
        w_cur, _ = blf.dimensions(font_id, before_cursor if before_cursor else "")
        cursor_x = area_left + w_cur
        if area_left - 1 <= cursor_x <= area_right + 1:
            c_verts = ((cursor_x, y_bottom + 3), (cursor_x, y_top - 3))
            c_batch = batch_for_shader(shader, "LINES", {"pos": c_verts})
            shader.uniform_float("color", CURSOR_COLOR)
            gpu.state.line_width_set(1.5)
            c_batch.draw(shader)
            gpu.state.line_width_set(1.0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_draw_handler():
    """Install the POST_PIXEL draw handler, but only once — even across
    module reloads.  The handle lives on ``bpy.app.driver_namespace`` so
    subsequent imports find and reuse it."""
    global _draw_handle
    ns = bpy.app.driver_namespace
    existing = ns.get(_NS_KEY)
    if existing is not None:
        # Previous module load already installed one — reuse.
        _draw_handle = existing
        return
    _draw_handle = bpy.types.SpaceDopeSheetEditor.draw_handler_add(
        draw_timeline_strips, (), "WINDOW", "POST_PIXEL",
    )
    ns[_NS_KEY] = _draw_handle


def unregister_draw_handler():
    global _draw_handle
    ns = bpy.app.driver_namespace
    handle = _draw_handle or ns.get(_NS_KEY)
    if handle is not None:
        try:
            bpy.types.SpaceDopeSheetEditor.draw_handler_remove(handle, "WINDOW")
        except (ValueError, RuntimeError):
            pass  # Already removed by another unregister
    ns.pop(_NS_KEY, None)
    _draw_handle = None
