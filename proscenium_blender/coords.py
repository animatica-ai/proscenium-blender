"""Coordinate conversion between MMCP wire format (right-handed Y-up) and
Blender (right-handed Z-up).

The whole protocol is in MMCP frame; the addon converts at the boundary —
once outbound (when building requests) and once inbound (when baking
responses or importing the canonical skeleton).

Conversions (verified against docs/concepts.html#coordinates):

    MMCP → Blender (inbound):
        position    (x, y, z)        -> (x, -z,  y)
        quaternion  (qx, qy, qz, qw) -> (qx, -qz, qy, qw)

    Blender → MMCP (outbound):
        position    (x, y, z)        -> (x,  z, -y)
        quaternion  (qx, qy, qz, qw) -> (qx,  qz, -qy, qw)
"""

from __future__ import annotations

from typing import Sequence, Tuple


Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]   # always (x, y, z, w) — MMCP order


# ---------------------------------------------------------------------------
# MMCP → Blender (inbound)
# ---------------------------------------------------------------------------

def mmcp_pos_to_blender(p: Sequence[float]) -> Vec3:
    x, y, z = p[0], p[1], p[2]
    return (x, -z, y)


def mmcp_quat_to_blender(q: Sequence[float]) -> Quat:
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return (qx, -qz, qy, qw)


# ---------------------------------------------------------------------------
# Blender → MMCP (outbound)
# ---------------------------------------------------------------------------

def blender_pos_to_mmcp(p: Sequence[float]) -> Vec3:
    x, y, z = p[0], p[1], p[2]
    return (x, z, -y)


def blender_quat_to_mmcp(q: Sequence[float]) -> Quat:
    """Blender quaternions on pose bones are stored as (w, x, y, z); callers
    must reorder to (x, y, z, w) *before* calling this. See request_builder.py
    for the conversion helper.
    """
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    return (qx, qz, -qy, qw)
