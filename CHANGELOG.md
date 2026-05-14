# Changelog

All notable changes to the Proscenium for Blender addon are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] — 2026-05-14

### Added

- **Generate Pose: apply to all bones or selected bones.** New *Apply pose
  to* option on the dialog (*All bones* / *Selected bones*). Selected
  scope uses pose-bone selection; on Mixamo-style control rigs, IK /
  control handles expand to the driving deform joints. Scripting:
  `pose_apply_scope='SELECTED'` on `proscenium.generate_pose`.
- **`blender_compat` helpers** for pose-bone selection across Blender
  versions (`PoseBone.select` on Blender 5 vs. legacy `Bone.select`).
- **Need help?** Button at the top of the Proscenium sidebar opens the
  [Animatica Discord](https://discord.gg/A8CrURBewz) in your browser
  (`proscenium.open_discord_help`).

### Changed

- **Canonical import with SOMA77 body.** The reference body mesh is
  shaded smooth, the armature uses **In Front** in the viewport so bones
  read through the surface, and after a successful body import the view
  switches to **Pose** mode on the new rig (best-effort; ignored without a
  3D View context).

### Fixed

- **Target armature after delete.** Deleting the rig (including multi-object
  delete) clears the picker, preview / source-action bookkeeping, timeline
  prompt strips, and dangling pointers. Centralized
  `reset_target_armature_state`; depsgraph validation treats unlinked
  armatures as gone (`users_collection`); panel and timer fallbacks when
  RNA updates do not fire.
- **Generate / Regenerate after a deleted rig.** Stale preview flags no
  longer send merges onto the wrong action when you pick a new character.
- **Timeline strip delete.** Deletes the strip under the cursor when
  possible, persists removals onto the target armature’s stored blocks, and
  clears strips when there is no live target armature.
- **Regenerate (Generate again) with a stashed source action.** Before
  building the motion request, generated sample keys are stripped from the
  preview action and surviving keys are merged onto the source action, then
  the source is made active again — keys added or edited during preview are
  no longer dropped when you click **Generate** a second time.
- **Reject after motion preview.** Same strip-and-merge path: removes
  generated motion samples from the preview while preserving authored
  keyframes, merges them onto the saved source action when present, and
  restores that action (avoids T-pose gaps on channels that only had
  generated keys).
- **Generate Pose keyframe tags.** Keys written at the pose frame are tagged
  as authored so the Dopesheet does not treat them like inherited
  **GENERATED** tags from a motion-bake preview.

## [0.3.1] — 2026-05-08

### Added

- **In-place motion (preview).** Scene setting pins the root bone’s
  horizontal translation with a `Limit Location` constraint during
  preview so the character plays vertically in place. F-curves are left
  intact; **Push to NLA** zeros root X/Z keys on the committed actions and
  removes the constraint so the NLA data is genuinely travel-free.
- **Agent skill (`skills/proscenium/SKILL.md`).** Cursor-oriented guide for
  driving Proscenium via Blender MCP: async operators, polling, prompt
  blocks, constraints, and known caveats.

### Changed

- **Root-path MMCP sampling.** Root-path curves share a world-space
  polyline helper; heading is derived from the tangent in MMCP XZ with
  `atan2(tx, tz)` so facing matches walk direction along the path.
  Single-frame generation windows are supported instead of returning no
  constraint. The auto **start anchor** can include `heading_radians`
  when the curve has **Follow direction** enabled, so frame 0 is not sent
  as translation-only with an arbitrary facing.

### Fixed

- **Snap to path vs. preview bake.** While `is_generating` or
  `is_previewing`, path snap no longer rewrites the root’s horizontal
  location curves from sparse Bezier control points, so dense glTF root
  translation from `/generate` is not replaced (which looked like snapping
  to the guide curve and foot sliding).

## [0.3.0] — 2026-05-01

### Added

- **Push to NLA workflow.** The Generate panel's preview action splits into
  one action per prompt block on commit, then assembles them on a single
  shared `Proscenium: Motion` NLA track. Previously-named "Accept" is now
  "Push to NLA" — the operator's `bl_idname` (`proscenium.accept`) is
  preserved for keymap / external compatibility.
- **Per-block action names from prompts.** Generated motion actions are now
  named after the user's prompt — `Proscenium_Motion: a person jumps` —
  with truncation to fit Blender's 63-char action-name limit.
- **Generation window from authored content.** The new
  `request_builder.compute_frame_range` derives the request's frame range
  from the union of enabled prompt blocks and source-action keyframes
  instead of the scene's `frame_start..frame_end`. Short edits no longer
  pay the cost of a full timeline.
- **Pose-generator: Preserve height option.** New checkbox in the Generate
  Pose dialog (default off → height matches the generated pose). When on,
  only the rig's local rotations apply, leaving the world XY *and* world Z
  untouched. When off, the model's height is applied while world XY stays
  pinned.
- **Pose-generator: persistent prompt.** The dialog pre-fills with the most
  recent prompt the user submitted (`scene.proscenium.last_pose_prompt`),
  so iterating on phrasings doesn't require retyping.
- **Effector pin: end-effector restriction.** The pin-joint dropdown now
  only offers the four canonical end-effectors (`LeftHand`, `RightHand`,
  `LeftFoot`, `RightFoot`); pinning interior chain joints would
  over-constrain the IK solver.

### Changed

- **Action naming.** `Proscenium_Generated` → `Proscenium_Motion: <prompt>`
  for full-motion bakes. `Proscenium_Poses` → `Proscenium_Pose` for the
  pose generator's output. The internal `_GENERATED_ACTION_PREFIXES` tuple
  catches both legacy and new names so back-compat with older scenes is
  preserved.
- **Anchor-frame tagging.** All source-action fcurve keyframes (rotation,
  location, and other channels) are now collected into the `KEYFRAME` tag
  set, not just rotation-bearing pose keyframes. Hand-authored Hips paths
  and other location-only keys keep their dopesheet styling after a bake.

### Fixed

- `AttributeError: 'NoneType' object has no attribute 'action'` in the
  Generate / Reject operators when an armature had no `animation_data`
  block yet (common after orphan-purge or a fresh canonical-skeleton
  import). Both call sites now `animation_data_create()` before assigning.
- Effector-pin and root-path samplers ship the wrong (previous-frame)
  world position when the empty / armature is parented or driven. Fixed
  by forcing a `view_layer.update()` after every `scene.frame_set` in
  `sample_effector_target` and `_root_keyframe_points`, matching the
  defensive flush already present in `sample_pose_keyframes`.
- The Generate Pose dialog appearing empty on first open after install
  (missing `BoolProperty` import; non-Property type annotations on
  `_thread`/`_result`/etc. tripping Blender's annotation resolver under
  `from __future__ import annotations`).
- NLA strips created via `track.strips.new` defaulting to `influence=0`
  in Blender 5.x — strips were silently producing zero contribution.
- Per-block bakes that switch the active action mid-bake leaving Blender
  5.x's NLA evaluator in a stale state where strips referencing the
  touched-then-detached actions silently produced zero animation. The
  split path now writes fcurves directly via the layered Action API
  (`action.layers.new` → `strip.channelbag(slot, ensure=True)
  .fcurves.new`), so the active action is never switched during writes.

### Internal

- `bake_gltf_to_actions_per_block` (`gltf_to_blender.py`): layered-Action
  bake that writes N actions in one pass with per-block frame filtering;
  retained for future surgical-regen use even though the live "preview
  then split on Push to NLA" path now goes through `_split_action_into_blocks`.
- `_block_ranges_for_split`, `_push_actions_to_nla`,
  `_clear_proscenium_nla_tracks`, `_split_action_into_blocks` (`operators.py`).
- `_is_orphan(action)` helper accounting for `use_fake_user` so the Reject
  cleanup loop correctly identifies per-block actions whose only reference
  is the fake user.

## [0.2.0]

- Bundled SOMA77 body mesh, skinned to the imported canonical armature.

## [0.1.0]

- Initial public release.

[0.3.2]: https://github.com/animatica-ai/proscenium-blender/releases/tag/v0.3.2
[0.3.1]: https://github.com/animatica-ai/proscenium-blender/releases/tag/v0.3.1
[0.3.0]: https://github.com/animatica-ai/proscenium-blender/releases/tag/v0.3.0
[0.2.0]: https://github.com/animatica-ai/proscenium-blender/releases/tag/v0.2.0
[0.1.0]: https://github.com/animatica-ai/proscenium-blender/releases/tag/v0.1.0
