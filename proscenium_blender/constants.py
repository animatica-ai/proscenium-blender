"""Shared constants for the Proscenium addon (MMCP client side).

Most lookup tables that the legacy addon needed (auto-detection, retarget
hints, slot mappings) are gone — the MMCP server publishes the canonical
skeleton via ``GET /capabilities`` and the addon imports it as a Blender
armature instead of guessing at the user's rig.
"""

# Default frame counts for new prompt blocks if the scene range is unset.
DEFAULT_FRAMES = 64

# Bezier curve display defaults for root-path constraints.
ROOT_PATH_CONTROL_POINTS = 4

# Empty display sizes (metres) for effector-target empties.
EFFECTOR_EMPTY_SIZE = 0.08

# Per-joint colours for effector empties — pure aesthetics, no semantics.
EFFECTOR_COLORS = {
    "Hips":      (1.0, 0.63, 0.34, 1.0),
    "LeftFoot":  (0.30, 0.70, 1.00, 1.0),
    "RightFoot": (0.30, 0.70, 1.00, 1.0),
    "LeftHand":  (0.30, 1.00, 0.50, 1.0),
    "RightHand": (0.30, 1.00, 0.50, 1.0),
    "Head":      (1.00, 0.85, 0.30, 1.0),
}

# Custom-property keys stamped on Blender objects to mark them as MMCP
# constraints. Picked up by the constraints scene-walker.
PROP_IS_ROOT_PATH    = "proscenium_is_root_path"
PROP_TARGET_JOINT    = "proscenium_target_joint"
PROP_MATCH_DIRECTION = "proscenium_match_direction"
PROP_SAMPLE_DENSITY  = "proscenium_sample_density"
