# Proscenium for Blender

The **officially supported MMCP client** for [Blender](https://www.blender.org)
4.x. Sketch poses on an armature, draw a path on the floor, pin a hand to
an empty — click Generate, get a glTF animation baked straight onto your
rig.

The addon doesn't ship a model. It speaks plain MMCP HTTP to whatever
[MMCP-compatible server][impls] you point it at — a self-hosted
[motionmcp-kimodo][kimodo] on your workstation, or the hosted **Animatica Cloud**
product.

[impls]: https://animatica.ai/mmcp/docs/get-started/implementations
[kimodo]: https://github.com/animatica-ai/motionmcp-kimodo
[mmcp]:   https://animatica.ai/mmcp

## Install

### Option 1 — install the zip (recommended for users)

```bash
git clone https://github.com/animatica-ai/proscenium-blender
cd proscenium-blender
make zip
# → dist/proscenium-blender-X.Y.Z.zip
```

In Blender:

1. **Edit → Preferences → Add-ons → Install…**
2. Pick the `dist/proscenium-blender-X.Y.Z.zip`.
3. Tick **Proscenium — AI Motion Generation** in the addon list.

### Option 2 — symlink the source tree (for development)

```bash
make install
```

Symlinks `proscenium_blender/` into your Blender 4.x addons directory
(`~/Library/Application Support/Blender/4.2/scripts/addons` on macOS;
override `BLENDER_ADDONS_DIR` for other paths). Editing files here lands
live in Blender on the next addon reload.

`make uninstall` removes the symlink.

## Configure

Once installed and enabled:

1. **Edit → Preferences → Add-ons → Proscenium → Preferences**
2. Set **Server** to your MMCP endpoint:
   - Self-host: `http://localhost:8000` (default)
   - LAN GPU box: `http://your-box.lan:8000`
   - Cloud: whatever URL the hosted product gives you

That's it — the addon discovers the available models via `/capabilities`.

## Use

1. Open the **N** sidebar in the 3D viewport, pick the **Proscenium** tab.
2. **Connection** panel → click **Connect**. Pick a model.
3. **Import &lt;model&gt; skeleton** — adds a Blender armature that mirrors
   the model's canonical skeleton.
4. **Main** panel → set the target armature.
5. Add prompt blocks on the timeline, draw root paths, pin effectors —
   whatever direction you want to give.
6. Click **Generate Motion**. The addon POSTs to `/generate`, parses the
   returned glTF, bakes joint rotations + the root translation onto your
   armature as a fresh action.
7. **Accept** keeps the result; **Reject** reverts.

For a single keyframed pose at the current frame (non-destructive,
undo-able), use **Generate Pose @ Frame N** instead.

## Architecture

| File | Role |
|---|---|
| `mmcp_client.py` | stdlib HTTP client, capabilities cache, `MmcpError` |
| `canonical_skeleton.py` | Import operator: build a Blender armature from the model's `canonical_skeleton` |
| `coords.py` | MMCP (right-handed Y-up) ↔ Blender (right-handed Z-up) axis swap + quaternion remap |
| `request_builder.py` | Assembles a complete `GenerateRequest` (segments + constraints + options) from current Blender state |
| `constraints_ui.py` | Operators + scene-walker + samplers for `root_path` curves and `effector_target` empties |
| `gltf_to_blender.py` | Decodes the `data:`-URI buffers in the glTF response and writes pose-bone keyframes |
| `mixamo_bake.py` | Optional bake step for control-rig-style armatures (e.g. Mixamo Control Rig) |
| `operators.py` | `proscenium.connect`, `proscenium.generate`, `proscenium.generate_pose`, `proscenium.accept` / `reject` / `cancel` |
| `panels.py` | Sidebar panels: Connection, Main, Constraints, Settings |
| `path_follow.py` | Path-curve constraint helpers for the Proscenium root-path workflow |
| `timeline_operators.py` + `timeline_overlay.py` | Prompt-block strips on the dopesheet (drag, resize, inline edit, hatched fill for unconditioned spans) |
| `properties.py` | `ProsceniumSettings`, `PromptBlock`, addon preferences |

The addon is ML-free: all generation, retargeting, and post-processing
happens on the server.

## Protocol & spec

The MMCP wire format is documented at **[animatica.ai/mmcp][mmcp]**. The addon
implements the client side — capabilities discovery, request assembly,
glTF response decoding. If you'd rather build your own client, the
[Quickstart (client)][qsclient] walks through it.

[qsclient]: https://animatica.ai/mmcp/docs/get-started/quickstart-client

## Compatibility

- **Blender 4.x** (tested on 4.2 LTS through 4.4).
- **MMCP v1.0** wire format. Designed to keep working against any v1.x
  server thanks to additive minor versions; see
  [Versioning](https://animatica.ai/mmcp/docs/protocol/versioning).
- The addon ships no Python deps beyond Blender's bundled stdlib —
  `urllib`, `json`, `base64`. No `pip install` step inside Blender.

## Limitations (current)

- **Server-side retargeting required** when your rig doesn't match the
  model's canonical skeleton. Self-hosted [motionmcp-kimodo][kimodo]
  doesn't support retargeting yet — use **Import canonical skeleton** in
  the addon, or point at the cloud.
- **One sample per request.** Multi-sample picker is future work.
- **No async polling.** The addon waits synchronously on `/generate`.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

Blender itself is GPL-2.0+, and the Blender Foundation considers addons
that import `bpy` to be derivative works of Blender. GPL-3.0+ keeps
distribution clean across the Blender Extensions Platform and the wider
addon ecosystem. The MMCP protocol and the [`motionmcp`](https://github.com/animatica-ai/motionmcp)
SDK that this addon talks to are Apache 2.0 — neither imports `bpy`, so
the GPL boundary stops at the addon.
