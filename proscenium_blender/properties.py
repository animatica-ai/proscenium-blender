"""Blender property definitions for Proscenium addon state."""

import json

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, PropertyGroup


# ---------------------------------------------------------------------------
# Per-armature prompt-block persistence
# ---------------------------------------------------------------------------

_BLOCKS_KEY = "proscenium_prompt_blocks"
_ACTIVE_KEY = "proscenium_active_block_index"


def _serialize_blocks(blocks) -> str:
    return json.dumps([
        {
            "prompt": b.prompt,
            "frame_start": b.frame_start,
            "frame_end": b.frame_end,
            "enabled": b.enabled,
            "color": list(b.color),
        }
        for b in blocks
    ])


def save_blocks_to_armature(arm_obj, settings):
    """Pickle prompt_blocks + active_block_index onto the armature's ID props."""
    if arm_obj is None:
        return
    try:
        _ = arm_obj.name
    except ReferenceError:
        return
    arm_obj[_BLOCKS_KEY] = _serialize_blocks(settings.prompt_blocks)
    arm_obj[_ACTIVE_KEY] = int(settings.active_block_index)


def load_blocks_from_armature(arm_obj, settings):
    """Replace settings.prompt_blocks with the serialized list on *arm_obj*.
    If the armature has no stored blocks, creates a single default block
    spanning the scene frame range."""
    settings.prompt_blocks.clear()
    if arm_obj is None:
        settings.active_block_index = 0
        return

    raw = arm_obj.get(_BLOCKS_KEY)
    if raw:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = []
        for item in data:
            b = settings.prompt_blocks.add()
            b.prompt = item.get("prompt", "")
            b.frame_start = int(item.get("frame_start", 1))
            b.frame_end = int(item.get("frame_end", 250))
            b.enabled = bool(item.get("enabled", True))
            color = item.get("color") or [0, 0, 0, 0]
            b.color = color[:4] + [0] * (4 - len(color))
        settings.active_block_index = int(arm_obj.get(_ACTIVE_KEY, 0))
        return

    # No stored data — seed with one default block covering the scene range.
    scene = bpy.context.scene
    b = settings.prompt_blocks.add()
    b.prompt = ""
    b.frame_start = scene.frame_start
    b.frame_end = scene.frame_end
    b.enabled = True
    settings.active_block_index = 0


def _preview_path_snap_update(self, context):
    """When the user flips the "Snap to Path" toggle on, immediately sync
    the current curve's control points into root-bone location keyframes.

    Without this the depsgraph handler only fires on the NEXT curve edit,
    which makes the checkbox feel dead when you first turn it on.
    Turning the toggle off is a no-op: keyframes stay where they are.
    """
    if not self.preview_path_snap:
        return
    arm = self.target_armature
    if arm is None or arm.type != 'ARMATURE':
        return
    # Local import — properties.py would otherwise drag path_follow into
    # registration order and we'd hit a circular load when the addon
    # loads properties first.
    from . import path_follow
    curve = path_follow._find_root_path_curve(context.scene)
    if curve is not None:
        path_follow.sync_path_to_armature(arm, context.scene, curve)


def _target_armature_update(self, context):
    """When the user switches target armature, swap the blocks list to match
    that armature's saved state (or seed defaults)."""
    settings = context.scene.proscenium
    new_arm = settings.target_armature
    old_arm = settings.previous_target_armature

    if new_arm == old_arm:
        return

    if old_arm is not None:
        try:
            save_blocks_to_armature(old_arm, settings)
        except ReferenceError:
            pass

    settings.prompt_blocks.clear()
    if new_arm is not None:
        load_blocks_from_armature(new_arm, settings)

    settings.previous_target_armature = new_arm

    # Redraw dopesheet so the strip overlay updates
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "DOPESHEET_EDITOR":
                area.tag_redraw()


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class PromptBlock(PropertyGroup):
    """One frame-range window with a text prompt, drawn on the timeline."""

    prompt: StringProperty(
        name="Prompt",
        description="Text prompt driving generation for this time window",
        default="",
    )
    frame_start: IntProperty(
        name="Start Frame",
        description="First frame of this range (inclusive)",
        default=1, min=1,
    )
    frame_end: IntProperty(
        name="End Frame",
        description="Last frame of this range (inclusive)",
        default=250, min=1,
    )
    enabled: BoolProperty(
        name="Enabled",
        description="Include this block when generating",
        default=True,
    )
    color: FloatVectorProperty(
        name="Color",
        description="Display color for this strip (0,0,0,0 = auto palette)",
        subtype="COLOR",
        size=4,
        min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0, 0.0),
    )


# Animatica Cloud's MMCP endpoint. Surfaced as a non-editable label in the
# prefs UI; users opt in to a self-hosted override via the `self_hosted`
# checkbox. Resolved at request time by mmcp_client.get_server_url().
CLOUD_API_URL = "https://api.animatica.ai"


class ProsceniumAddonPreferences(AddonPreferences):
    """Addon-level preferences — edit in ``Edit > Preferences > Add-ons > Proscenium``."""

    bl_idname = __package__  # "proscenium_blender"

    self_hosted: BoolProperty(
        name="Self-hosted",
        default=False,
        description=(
            "Tick when running an MMCP server on your own machine or LAN "
            "(e.g. motionmcp-kimodo on localhost). Untick to use Animatica "
            "Cloud at api.animatica.ai (default; requires sign-in)."
        ),
    )
    server_url: StringProperty(
        name="Server URL",
        default="http://localhost:8000",
        description="Base URL of your self-hosted MMCP server",
    )

    # --- Animatica Cloud session (populated by /auth/login) ----------------
    # Auth is NOT part of the MMCP protocol — it lives at the cloud's proxy
    # in front of /generate. Self-hosted servers ignore the Authorization
    # header entirely.
    access_token: StringProperty(
        name="Access Token",
        default="",
        description="Animatica session token; valid for ~1 hour, then auto-refreshed",
        subtype='PASSWORD',
    )
    refresh_token: StringProperty(
        name="Refresh Token",
        default="",
        description="Long-lived refresh token used to renew the session",
        subtype='PASSWORD',
    )
    email: StringProperty(
        name="Email",
        default="",
        description="Email of the signed-in Animatica user",
    )
    tier: StringProperty(
        name="Tier",
        default="",
        description="Animatica plan tier (free / pro / team / admin)",
    )

    def draw(self, context):
        from . import mmcp_client

        layout = self.layout

        # --- Server selection -------------------------------------------------
        col = layout.column(align=True)
        col.label(text="Server", icon='WORLD_DATA')

        # Cloud URL: always visible, never editable.
        row = col.row(align=True)
        row.enabled = False
        row.label(text=f"Animatica Cloud — {CLOUD_API_URL}/")

        col.prop(self, "self_hosted")
        if self.self_hosted:
            col.prop(self, "server_url", text="Override URL")

        # --- Connection status + connect/reconnect ----------------------------
        layout.separator()
        caps = mmcp_client.cached_capabilities()
        box = layout.box()
        if caps is None:
            err = mmcp_client.last_connection_error()
            if err:
                box.label(text="Connection failed", icon='ERROR')
                for line in err.split("\n")[:3]:
                    box.label(text=line)
            else:
                box.label(text="Not connected", icon='UNLINKED')
            box.operator("proscenium.connect", icon='URL', text="Connect")
        else:
            n_models = len(caps.get("models", []))
            row = box.row()
            row.label(text=f"Connected — {n_models} model(s)", icon='LINKED')
            row.operator("proscenium.connect", icon='FILE_REFRESH', text="Reconnect")

            proto = caps.get("protocol_version", "?")
            box.label(text=f"MMCP {proto} · {caps.get('coordinate_system', '?')} · {caps.get('units', '?')}")

            for m in caps.get("models", []):
                mbox = box.box()
                mbox.label(text=m.get("id", "?"), icon='OUTLINER_OB_ARMATURE')
                joints = len(m.get("canonical_skeleton", {}).get("joints", []))
                fps = m.get("fps", "?")
                retarget = "yes" if m.get("supports_retargeting") else "no"
                mbox.label(text=f"{joints} joints @ {fps} fps · retargeting: {retarget}")

                segs = ", ".join(m.get("supported_segments") or []) or "—"
                mbox.label(text=f"segments: {segs}")
                cons = ", ".join(m.get("supported_constraints") or []) or "—"
                mbox.label(text=f"constraints: {cons}")

                limits = m.get("limits") or {}
                max_dur = limits.get("max_duration_seconds")
                rec_dur = m.get("recommended_max_duration_seconds")
                if max_dur is not None or rec_dur is not None:
                    parts = []
                    if rec_dur is not None:
                        parts.append(f"recommended ≤ {rec_dur:g}s")
                    if max_dur is not None:
                        parts.append(f"max {max_dur:g}s")
                    mbox.label(text=" · ".join(parts))

        layout.separator()

        # --- Auth section -----------------------------------------------------
        if self.self_hosted:
            box = layout.box()
            box.label(text="Self-hosted: sign-in not required", icon='INFO')
            return

        if self.access_token:
            box = layout.box()
            row = box.row()
            row.label(text=f"Signed in: {self.email}", icon='CHECKMARK')
            if self.tier:
                row.label(text=f"({self.tier})")
            row = box.row()
            row.operator("proscenium.signout", icon='X', text="Sign out")
        else:
            box = layout.box()
            box.label(text="Animatica Cloud — sign in", icon='USER')
            box.operator("proscenium.signin", icon='IMPORT', text="Sign in")


def _model_id_items(self, context):
    """Dynamic EnumProperty items, populated by the Connect operator."""
    from . import mmcp_client
    return mmcp_client.cached_model_items()


class ProsceniumSettings(PropertyGroup):
    """Scene-level addon state."""

    # -- MMCP server connection --
    model_id: EnumProperty(
        name="Model",
        description="Motion-generation model exposed by the connected MMCP server",
        items=_model_id_items,
    )

    # -- Target armature --
    target_armature: PointerProperty(
        name="Target Armature",
        type=bpy.types.Object,
        description="Armature with keyframed animation to generate from",
        poll=lambda self, obj: obj.type == 'ARMATURE',
        update=_target_armature_update,
    )

    previous_target_armature: PointerProperty(
        name="Previous Target Armature",
        type=bpy.types.Object,
        options={"HIDDEN", "SKIP_SAVE"},
    )

    # -- Prompt blocks (one time-window with one prompt each) --
    prompt_blocks: CollectionProperty(
        type=PromptBlock,
        name="Prompt Blocks",
        description="Per-window prompts drawn as strips on the timeline",
    )
    active_block_index: IntProperty(
        name="Active Block",
        default=0, min=0,
    )

    # -- Generation settings --
    seed: IntProperty(name="Seed", default=42, min=0, max=999999)

    quality_preset: EnumProperty(
        name="Quality",
        items=[
            ("STANDARD", "Standard", "50 denoising steps"),
            ("HALF", "Half", "25 denoising steps"),
            ("QUARTER", "Quarter", "12 denoising steps"),
            ("CUSTOM", "Custom", "Custom step count"),
        ],
        default="STANDARD",
    )
    custom_steps: IntProperty(
        name="Steps", default=50, min=1, max=200,
    )

    cfg_enabled: BoolProperty(name="CFG Enabled", default=True)
    cfg_text: FloatProperty(
        name="Text Weight", default=2.0, min=0.0, max=5.0, step=10,
    )
    cfg_constraint: FloatProperty(
        name="Constraint Weight", default=2.0, min=0.0, max=5.0, step=10,
    )

    post_processing: BoolProperty(
        name="Motion Cleanup",
        description=(
            "Run server-side motion correction after generation: tightens "
            "keyframe pins, fixes foot sliding, enforces end-effector "
            "constraints. Requires the server to have the motion_correction "
            "package installed. Safe to enable; adds a second or two per "
            "sample"
        ),
        default=True,
    )
    preview_path_snap: BoolProperty(
        name="Snap to Path",
        description=(
            "Bake each root_path curve control point into one keyframe on "
            "the target armature's root bone. Turning this on syncs "
            "immediately from the current curve; leaving it on keeps the "
            "keyframes in sync as you edit the curve in the viewport"
        ),
        default=True,
        update=_preview_path_snap_update,
    )
    root_margin: FloatProperty(
        name="Root Margin", default=0.04, min=0.0, max=0.5,
    )
    num_transition_frames: IntProperty(
        name="Transition Frames", default=5, min=0, max=30,
    )

    default_prompt: StringProperty(
        name="Prompt",
        default="a person moves naturally",
        description="Text prompt for motion generation",
    )

    last_pose_prompt: StringProperty(
        name="Last pose prompt",
        default="a person stands in a neutral pose",
        description=(
            "Most-recent prompt used in the Generate Pose dialog. "
            "Pre-fills the dialog the next time it opens so the user "
            "can iterate on a phrasing without retyping"
        ),
    )

    # -- Runtime state (not saved) --
    is_generating: BoolProperty(name="Generating", default=False)
    generation_progress: FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    cancel_requested: BoolProperty(
        name="Cancel Requested",
        default=False,
        description="Flipped by the Cancel button; the running modal op picks it up and exits",
    )

    # -- Quota / upgrade state. Set when the cloud returns 429
    #    quota_exceeded; cleared on the next successful generation or
    #    when the user dismisses the banner. ``upgrade_url`` is sent in
    #    the error envelope so the plugin doesn't hardcode a billing URL.
    quota_exceeded_message: StringProperty(
        name="Quota Message",
        default="",
        description="Human-readable quota error message from the cloud",
    )
    quota_upgrade_url: StringProperty(
        name="Upgrade URL",
        default="",
        description="URL to open in the user's browser to upgrade the plan",
    )

    # -- Preview state: name of the user's source action while a
    #    Proscenium_Generated action is being previewed.  Empty = idle.
    source_action_name: StringProperty(
        name="Source Action",
        default="",
        description="Original action name preserved while previewing a generated motion",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    ProsceniumAddonPreferences,
    PromptBlock,
    ProsceniumSettings,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.proscenium = PointerProperty(type=ProsceniumSettings)


def unregister():
    del bpy.types.Scene.proscenium
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
