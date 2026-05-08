---
name: proscenium
description: Drives the Proscenium Blender addon — AI motion generation via the MMCP protocol — through the Blender MCP. Use when the user asks to generate character motion or a single pose from a text prompt, animate a Blender armature with Proscenium, invoke any `proscenium.*` operator (connect, generate, generate_pose, accept, reject, import_canonical_skeleton, add_root_path, add_effector_target), or work with MMCP / Animatica / motionmcp endpoints from Blender. Also use when the user wants Claude to drive the Proscenium sidebar in Blender remotely or is troubleshooting Proscenium connection, generation, quota, or NLA-bake behavior.
---

# Proscenium for Blender

Proscenium is a Blender 4.x addon that generates skeletal motion from text prompts by talking MMCP HTTP to a server (self-hosted [motionmcp-kimodo](https://github.com/animatica-ai/motionmcp-kimodo) or Animatica Cloud). The addon is ML-free — it assembles a JSON request describing segments + constraints, POSTs `/generate`, then bakes the returned glTF onto a target armature.

Everything in this skill is driven from Python through the Blender MCP's `execute_blender_code` tool. Operators live in the `proscenium.*` namespace and scene-level state lives at `bpy.context.scene.proscenium`.

## Core invariant — generation is asynchronous

`proscenium.generate` and `proscenium.generate_pose` are **modal operators**. They spawn a worker thread that runs the HTTP request, return `{'RUNNING_MODAL'}` immediately, and the bake happens later when the worker finishes and the modal timer fires. Calling the operator does not mean the motion is on the armature yet.

Two consequences:

1. **Pick the right invocation form for the operator.**
   - `proscenium.generate` → `'INVOKE_DEFAULT'`. Goes through Blender's invoke flow so the modal handler is wired to a real window/event context. Has no props dialog.
   - `proscenium.generate_pose` → `'EXEC_DEFAULT'` with kwargs (`prompt`, `seed`, `preserve_height`). Its `invoke()` opens a `wm.invoke_props_dialog`, which **blocks waiting on the user** when called from `execute_blender_code` and there's no way to dismiss it programmatically. `EXEC_DEFAULT` skips the dialog and starts the modal directly.

2. **Poll across separate `execute_blender_code` calls.** The modal handler advances on Blender's event-timer tick — those ticks fire *between* MCP calls, not during one. A `time.sleep` loop inside a single call will hang Blender without progress. Issue the kickoff in one call, then poll `bpy.context.scene.proscenium.is_generating` in subsequent calls until it flips to `False`.

```python
# Call 1 — kick off motion
bpy.ops.proscenium.generate('INVOKE_DEFAULT')
# or pose
bpy.ops.proscenium.generate_pose(
    'EXEC_DEFAULT', prompt="a person sits on a chair", seed=42, preserve_height=False,
)
```

```python
# Call 2..N — poll
s = bpy.context.scene.proscenium
print({"generating": s.is_generating, "previewing": s.is_previewing,
       "source_action": s.source_action_name,
       "quota_msg": s.quota_exceeded_message})
```

**Different end states for the two operators:**
- `proscenium.generate` flips `is_generating: False, is_previewing: True` on success — bake lives in `Proscenium_Motion: Preview`, awaiting Accept/Reject.
- `proscenium.generate_pose` flips `is_generating: False` (and `is_previewing` stays `False`). It writes rotation keyframes for every pose bone + a location keyframe for the root directly into the armature's active action. There is no Accept/Reject — the keys are committed (Ctrl+Z to undo).

First generation against Animatica Cloud after idle can take 60s+ (cold-start spinning up the GPU container). Self-hosted motionmcp-kimodo is typically a few seconds. Set user expectations accordingly.

## Standard workflow

The canonical pipeline for a directed motion is **block out → prompt → generate**:

```
1. Verify the addon is enabled
2. Connect              → proscenium.connect
3. Pick model           → settings.model_id = "..."
4. Set target armature  → proscenium.import_canonical_skeleton  (or assign existing)
5. Block out the motion with keyframes:
   - Set the scene frame range for the full motion (scn.frame_start/frame_end)
   - At each waypoint frame, run proscenium.generate_pose with a prompt
     describing that pose ("stands neutrally", "sits on a chair", "raises
     arms", ...). Each call commits a full-body rotation key + root key.
   - Author any extra hand-edits on top (e.g. rotate Hips 180° around vertical
     for facing direction, then re-keyframe).
6. Set prompt blocks    → settings.prompt_blocks (one per timeline segment)
7. (optional) Spatial constraints → proscenium.add_root_path,
                                    proscenium.add_effector_target
8. Generate + poll      → proscenium.generate (INVOKE_DEFAULT), then poll
                          is_generating
9. Accept or Reject     → proscenium.accept / proscenium.reject (ask the user)
```

The pose-anchor step is the lever for *direction*: a `pose_keyframe` constraint at frame N pins the body's pose at N, and the model fills in motion between anchors guided by the prompt blocks. Skipping this step (free-form generation) gives the model total freedom over both pose and trajectory — fine for a generic walk, weak for "walk *here*, then sit *facing back*".

### 1. Verify the addon

```python
import bpy
addon = bpy.context.preferences.addons.get("proscenium_blender")
# If None, the user must enable it in Edit > Preferences > Add-ons
```

If absent, stop and tell the user to enable it — don't try to install the addon programmatically.

### 2. Connect

`proscenium.connect` fetches `GET /capabilities` from the configured server. Server URL lives on the addon prefs: `addon.preferences.server_url` for self-hosted, with `addon.preferences.self_hosted` toggling cloud vs. self-hosted mode. Cloud is the default.

```python
bpy.ops.proscenium.connect()
from proscenium_blender import mmcp_client
caps = mmcp_client.cached_capabilities()  # None if connect failed
print("models:", [m["id"] for m in (caps or {}).get("models", [])])
print("error:", mmcp_client.last_connection_error())
```

If the user is on cloud and not signed in, `proscenium.signin` opens a browser flow. Only invoke this with explicit user confirmation — it's an external auth action.

### 3. Pick a model

```python
s = bpy.context.scene.proscenium
s.model_id = "kimodo-v1"   # one of the IDs from caps["models"]
```

### 4. Target armature

If the user has no rig, import the canonical skeleton. Joint names match what the server expects, so generation works without retargeting (which most v1 servers don't support).

```python
bpy.ops.proscenium.import_canonical_skeleton(with_body=True)
# This also assigns the new armature to settings.target_armature
```

Otherwise point at an existing armature:

```python
s.target_armature = bpy.data.objects["Armature"]
```

### 5. Block out the motion with keyframes (recommended)

For directed motion ("walk *here*, then sit *facing back*"), block out waypoint poses with `generate_pose` *before* writing prompt blocks or running the main generator. Each `generate_pose` call commits a full-body rotation key + a root-bone location/height key at the current frame, which the main generator picks up as a `pose_keyframe` constraint and treats as a hard anchor.

```python
# Standing anchor at the start.
scn.frame_set(1)
bpy.ops.proscenium.generate_pose(
    'EXEC_DEFAULT',
    prompt="a person stands in a neutral pose with arms relaxed at sides",
    seed=42,
    preserve_height=False,
)
# ...poll is_generating across calls until False...

# Seated anchor at the end.
scn.frame_set(105)
bpy.ops.proscenium.generate_pose(
    'EXEC_DEFAULT',
    prompt="a person sits on a chair facing forward",
    seed=42,
    preserve_height=False,
)
# ...poll again...
```

After blocking, hand-edit any pose channel you want to override — for instance, post-multiplying the Hips rotation by `Quaternion((0, 1, 0), pi)` and re-keying flips the seated body 180° around vertical (= bone-local Y for Hips = world Z) so it faces the opposite direction without the model having to interpret an orientation prompt.

`fill_mode` is what controls how aggressively the model honors your blocked-in pose. The sampler picks it from which bones you keyframed:

- All keyed bones in the EE-chain set (`Hips`, `LeftFoot`, `LeftToeBase`, `RightFoot`, `RightToeBase`, `LeftHand`, `LeftHandMiddleEnd`, `RightHand`, `RightHandMiddleEnd`) → `fill_mode="generate"` — the server pins those joints, fills the rest freely.
- Any keyed bone outside that set (spine, neck, arms mid-chain, etc.) → `fill_mode="rest"` — the server pins **every** joint via FullBodyConstraintSet. Strong anchor, no freedom.

`generate_pose` keyframes all 30 canonical joints, so it inherently puts you in `"rest"` mode. That's usually what you want for waypoint anchors. If you only want a *partial* anchor (e.g. just rotate Hips to control facing), keyframe only EE-chain bones — manually keying just `Hips` keeps the model free to interpret the body pose from the prompt.

### 6. Prompt blocks

Each block is one frame range with one text prompt. Multi-block requests get split into one action per block on Accept.

```python
s = bpy.context.scene.proscenium
s.prompt_blocks.clear()
b = s.prompt_blocks.add()
b.prompt = "a person walks forward casually"
b.frame_start = 1
b.frame_end = 60
b.enabled = True
```

For multiple blocks, add several entries with non-overlapping frame ranges. The plugin half-expands gaps so NLA strips abut on Accept.

### 7. Constraints (optional)

Spatial constraints get sampled from scene objects at request time. Coordinate conversion (Blender Z-up ↔ MMCP Y-up) is handled internally — work in Blender's frame.

```python
# Bezier curve on the floor; the character follows it.
bpy.ops.proscenium.add_root_path()

# Pin a canonical end-effector joint to an animated empty.
# Each location keyframe on the empty becomes one constraint frame.
bpy.ops.proscenium.add_effector_target(joint="RightHand")
```

Only canonical end-effector joints are pinnable (`RightHand`, `LeftHand`, `RightFoot`, `LeftFoot`). Pinning interior joints over-constrains the IK solver; the operator filters those out.

**Watch out for `preview_path_snap`.** When you add a `root_path` curve and the scene-level `preview_path_snap` setting is `True` (default), a depsgraph handler in `path_follow.py` aggressively rewrites the root pose-bone's horizontal location fcurves on every curve update — distributing N control points evenly across `[scene.frame_start, scene.frame_end]` and **clearing every existing keyframe on those axes**. If you authored Hips X/Z location keys (e.g. for a hand-paced walk) before adding the curve, they get clobbered the moment the curve appears. To keep your keys, set `bpy.context.scene.proscenium.preview_path_snap = False` *before* `proscenium.add_root_path()` (or never add a curve and let the source-action keyframes alone drive root motion through the auto-added start anchor).

The handler also creates / renames the armature's action to `Proscenium_Path` the first time it runs, so don't be surprised when an action you called `MyAction` shows up as `Proscenium_Path` after a curve is added.

### 8. Generate and wait

```python
# Kick off motion
bpy.ops.proscenium.generate('INVOKE_DEFAULT')
```

Poll in subsequent calls until `is_generating` is `False`. Read `quota_exceeded_message` too — a non-empty value means the cloud returned 429 and no bake happened.

For pose-only generation (used in step 5 above), use `EXEC_DEFAULT` with explicit kwargs — see the Core Invariant section. Polling is on `is_generating` only; pose bakes don't enter `is_previewing` state.

### 9. Accept or Reject

Ask the user before committing — these are irreversible from the plugin's UI perspective.

```python
bpy.ops.proscenium.accept()   # pushes to NLA; multi-block scenes split into per-block actions
bpy.ops.proscenium.reject()   # restores the source action
```

To abort an in-flight generation, call `bpy.ops.proscenium.cancel()`. The HTTP request continues server-side but the addon discards the result.

## Operators

| Operator                                | Purpose                                                       |
|-----------------------------------------|---------------------------------------------------------------|
| `proscenium.connect`                    | Fetch `/capabilities`, populate model picker                  |
| `proscenium.import_canonical_skeleton`  | Build a Blender armature from the model's canonical skeleton  |
| `proscenium.generate`                   | Modal: build request, POST `/generate`, bake glTF             |
| `proscenium.generate_pose`              | Modal: single-frame pose at the current frame                 |
| `proscenium.accept`                     | Keep the bake, push to NLA                                    |
| `proscenium.reject`                     | Restore the source action                                     |
| `proscenium.cancel`                     | Abort the in-flight generation                                |
| `proscenium.add_root_path`              | Bezier curve sampled into a `root_path` constraint            |
| `proscenium.add_effector_target`        | Empty pinned to an end-effector joint                         |
| `proscenium.remove_constraint_object`   | Delete a named constraint object                              |
| `proscenium.focus_constraint_object`    | Select + view-frame a constraint object                       |
| `proscenium.signin` / `proscenium.signout` | Animatica Cloud auth (cloud-only; browser flow)            |

Invocation form per operator:
- `proscenium.generate` → `'INVOKE_DEFAULT'` (modal, no props dialog).
- `proscenium.generate_pose` → `'EXEC_DEFAULT'` with kwargs (modal, but its `invoke()` opens a blocking props dialog — `EXEC_DEFAULT` skips it).
- All others → bare call is fine.

## Scene state — `bpy.context.scene.proscenium`

| Field                      | Type   | Notes                                                          |
|----------------------------|--------|----------------------------------------------------------------|
| `model_id`                 | enum   | One of the connected server's model IDs                        |
| `target_armature`          | Object | Armature being animated                                        |
| `prompt_blocks`            | coll.  | `{prompt, frame_start, frame_end, enabled, color}` per entry   |
| `seed`                     | int    | 0–999999                                                       |
| `quality_preset`           | enum   | `STANDARD` / `HALF` / `QUARTER` / `CUSTOM`                     |
| `custom_steps`             | int    | Used when `quality_preset == "CUSTOM"`                         |
| `cfg_enabled`              | bool   | Toggle classifier-free guidance                                |
| `cfg_text`                 | float  | Text guidance weight (0–5)                                     |
| `cfg_constraint`           | float  | Constraint guidance weight (0–5)                               |
| `inplace`                  | bool   | Live toggle: pin root xz so character animates in place        |
| `post_processing`          | bool   | Server-side foot-skate / pin tightening                        |
| `is_generating`            | bool   | True while the modal worker is running — poll this             |
| `is_previewing`            | bool   | True after a successful bake, awaiting Accept/Reject           |
| `source_action_name`       | str    | The pre-generation action name (restored on Reject)            |
| `quota_exceeded_message`   | str    | Non-empty after a cloud 429; surface to user, don't auto-retry |
| `quota_upgrade_url`        | str    | Upgrade URL from the cloud's error envelope                    |

## Things to watch for

- **Don't sleep inside one `execute_blender_code` call to wait for generation.** Blender's modal handler advances between MCP calls, not during one. Long blocks won't observe `is_generating` flipping. Always poll across calls.

- **`proscenium.generate` returns `CANCELLED` immediately** if `target_armature` is unset, no model is picked, or another generation is in flight. Always check `is_generating` before invoking.

- **Reconnect after server-URL changes.** `mmcp_client` caches capabilities; without `proscenium.connect` the model dropdown and request builder still see the old schema.

- **Quota errors are non-retriable.** A non-empty `quota_exceeded_message` means the cloud rejected the request. Show it to the user along with `quota_upgrade_url` — don't loop.

- **Regenerate from preview is fine.** If `is_previewing == True` and you call `proscenium.generate` again, the plugin restores the source action first so the new request reflects the user's authored keys, not the previous preview. No need to reject manually.

- **Coordinate conversion is internal.** The plugin's `coords` module handles MMCP Y-up ↔ Blender Z-up for both joint rotations and constraints sampled from scene objects. Don't try to flip axes or remap quaternions in user code.

- **Cancel ≠ undo.** `proscenium.cancel` stops the modal from waiting; the server-side request keeps running. The bake is just discarded. There's no client-side rollback because nothing was applied yet.

- **Effector pins only target canonical end-effectors.** `RightHand`, `LeftHand`, `RightFoot`, `LeftFoot`. The operator's enum filters by what's present on the target armature, so an empty list means the rig isn't using canonical joint names.

- **Animatica auth is not part of MMCP.** The `Authorization` header lives at the cloud's proxy in front of `/generate`; self-hosted servers ignore it. `signin` / `signout` are no-ops in self-hosted mode.

## Known plugin issues (workarounds)

These are bugs in the plugin as of v0.3.0 that you should be aware of when driving it programmatically. They are not "watch-outs" — they are real defects with practical workarounds.

- **Depsgraph staleness on per-frame pose evaluation in MCP / non-interactive contexts.** Several plugin code paths sample armature state across frames by `scene.frame_set(f); view_layer.update(); read arm.matrix_world @ pose_bone.matrix`. In a normal interactive Blender session this works because the viewport drives depsgraph re-evaluation between calls. When driven through `mcp__blender__execute_blender_code` (no viewport tick between bone-space reads), the evaluation can stay frozen on whatever frame the depsgraph last committed to — every per-frame read returns the *same* world position regardless of which frame was set. This silently breaks two important code paths:
  1. `bake_single_pose` in `height_only` mode (`gltf_to_blender.py`): `current_world = arm.matrix_world @ bone.head` is read *before* the modal handler advances the playhead to `_target_frame` (the `frame_set` happens at the very end of the modal callback). A `generate_pose` at frame N may commit XY values that came from whatever frame Blender was last evaluating.
  2. `sample_pose_keyframes` in `constraints_ui.py`: each pose anchor's `root_position` field is sampled from `arm.matrix_world @ sample_root.matrix.translation` after `scene.frame_set(f); view_layer.update()`. In a stale-depsgraph context, all anchors get the **same** `root_position` even when the user keyframed real per-frame trajectory on the root bone — the server then sees N pose constraints sharing one root, and the generated motion has a flat (or arbitrary) trajectory between anchors.
  
  **Workarounds:**
  - For trajectories: don't rely on per-frame pose-bone reads to convey root motion. Add an explicit `root_path` Bezier curve (manually authored object with `proscenium_is_root_path=True` and `proscenium_match_direction`/`proscenium_sample_density` ID props) — its sampler reads Bezier geometry directly, no depsgraph dependency. Pair this with `preview_path_snap=False` so the path-follow handler doesn't clobber your pose-bone keyframes.
  - For Hips height arcs (jumps, crouches): the curve only carries XY trajectory; vertical comes from pose anchors' joint rotations and from the model interpolating the prompt. Authoring a Hips Z keyframe sequence won't show up in the request because of point 2 above. If you really need a strong vertical anchor, pin the foot/hand effectors with explicit `proscenium.add_effector_target` empties (those *do* sample correctly because they read empty world position, not pose-bone evaluation).
  - For `generate_pose` XY: verify Hips world XY after each call; re-keyframe the location channel if it drifted.

- **`path_follow`'s depsgraph handler clobbers user-authored Hips horizontal keyframes** when `preview_path_snap` is `True` (default). It clears the entire X and Z (bone-local) location fcurves and rewrites them with one keyframe per curve control point, evenly spaced across the scene range — irrespective of any keys you authored. **Workaround:** set `scn.proscenium.preview_path_snap = False` before `proscenium.add_root_path()`, or skip the curve entirely and rely on the `_start_anchor` + pose-keyframe constraints to position the character.

- **`bake_single_pose` `height_only` writes all three location axes, not just height.** The mode is meant to "preserve current XY, override Z", but the implementation writes the full 3-component bone-local location every time. Combined with the bone.head staleness above, this is how a `generate_pose` call can silently move the root horizontally. **Workaround:** same as above — verify Hips world position after `generate_pose` and re-keyframe if needed.

- **`is_previewing` is not cleared on file load when the target armature is missing.** A `.blend` saved mid-preview reloads with `is_previewing=True` even if the armature it was bound to has been deleted, which leaves the UI in a state where Accept/Reject is offered for nothing. **Workaround:** at session start, if `target_armature is None and is_previewing` is True, manually flip both `is_previewing` and `source_action_name` off — there's nothing real to revert to.

- **`generate_pose`'s description claims "Non-destructive — undo with Ctrl+Z".** With a `root_path` curve in the scene this is not true: any keyframes you authored at frames other than the bake target can be lost via the path-follow handler firing during the bake. Treat `generate_pose` as additive only when no curve is present.

## Reference

- Repo: this addon (`proscenium_blender/` package)
- MMCP wire format: https://animatica.ai/mmcp
- Self-host server: https://github.com/animatica-ai/motionmcp-kimodo
- See `proscenium_blender/operators.py` for the canonical operator definitions and `proscenium_blender/properties.py` for the full `ProsceniumSettings` schema.
