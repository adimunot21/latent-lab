# 10 — Decoder-free visualization: lookup tables and PCA

JEPA's "no decoder" principle creates a UX problem the moment you want to
*show* anyone what the model imagines: an imagined latent ẑ is 128 numbers
with no pixels attached. This lesson covers the two devices that solve it —
both built at export time by `training/src/latentlab/export/latent_maps.py`,
both shipped to the browser, both validated before being trusted.

## Device 1: the latent↔state lookup table

Idea: if you can't decode a latent, *look it up*. Precompute pairs
(state, latent) densely over the state space; at runtime, map any latent to
the state of its nearest stored neighbor.

```python
def grid_free_states(env, resolution=64):
    coords = (np.arange(resolution) + 0.5) / resolution
    states = [(x, y) for y in coords for x in coords if env.is_free(x, y)]
    return np.asarray(states, dtype=np.float32)
```

A 64×64 grid of the unit square, keeping only collision-free cells → **2880
states**. Render each, encode each (batched), store both arrays. Decoding is
brute-force nearest neighbor — 2880 squared-distance computations in R¹²⁸ —
which sounds crude and is: crude, exact-enough, and fast-enough. (The
browser's planner decodes whole elite trajectories this way inside a worker,
lesson 12, with an early-exit distance loop.)

Why this works *here* and wouldn't in general: Two Rooms' state space is 2-D
and compact, so 2880 samples blanket it. The technique's cost is exponential
in intrinsic state dimension — for the descoped PushT stretch goal (pose ×
block pose ≈ 6-D) this exact trick is already marginal. Know your manifold.

**Validation before trust** (house rule by now): NN-decode the latent
trajectory of a real held-out episode and compare against its true states.
Measured: **0.0063 world units** mean error — sub-pixel (one frame pixel =
1/64 ≈ 0.0156). The script also writes a PNG overlaying true and decoded
paths; they visually coincide.

## Device 2: the fixed PCA projection

For the latent-cloud panel we need 128-D → 2-D. PCA via SVD on the centered
grid latents:

```python
mean = z.mean(dim=0)
_, s, v = torch.linalg.svd(z - mean, full_matrices=False)
components = v[:2].T.contiguous()            # (D, 2)
explained = (s**2) / (s**2).sum()
```

Top-2 components explain **71.5%** of variance — a 2-D-manifold-shaped
number (lesson 02 again). The scatter, colored by room, is genuinely
beautiful: **two sheets joined at a narrow neck** — the doorway bottleneck,
visible as topology in latent space. The encoder learned the rooms are
almost-separate regions connected by a thin passage, because that's what the
dynamics data says.

### The decision that makes the collapse demo work

The projection is **fit once, on the healthy encoder, and never refit.**
When the browser's checkpoint switcher (lesson 12) swaps in the collapsed
model, its latents are projected through the *healthy* basis, in the *same
fixed view box*. Why insist:

- Refit PCA per checkpoint and the projection auto-normalizes scale — the
  collapsed model's microscopic residual cloud (std 0.001, but isotropic-ish,
  lesson 08) would be blown up to fill the panel and look disconcertingly
  *healthy*. The demo's punchline would be normalized away.
- Fixed basis + fixed viewport = what the user sees is the honest amplitude
  story: spread 11.53 → 0.0012, a ~10,000× visual implosion to a dot.

This is lesson 08's "scale-invariant metrics lie" principle applied to
*visualization*: a per-checkpoint PCA is a scale-invariant view. The fix is
the same — hold the measuring stick fixed.

## Packaging for the browser

`export/manifest.py` ships both devices: the PCA (small: 128×2 + 128 floats)
inline in `manifest.json`; the lookup table (2880×130 floats ≈ 1.5 MB) as
two raw little-endian float32 `.bin` files — binary data has no business
being JSON-stringified. The browser memory-maps them into Float32Arrays
directly (`new Float32Array(arrayBuffer)`), zero parsing.

## Try it

1. Rebuild the maps (`uv run python -m latentlab.export.latent_maps
   --checkpoint checkpoints/two_rooms_v1/healthy_v1/final.pt --out
   /tmp/maps`) and open both sanity PNGs. Find the doorway neck in the PCA
   scatter.
2. Build maps for `collapse_v1/final.pt` into a different dir. Compare
   `pca_explained` and the scatter against the healthy ones — then explain
   why the *shipped* manifest still uses the healthy PCA for all
   checkpoints.
3. Resolution study: rebuild at `--resolution 16` (≈180 states) and re-run
   the lookup sanity check. How does NN-decode error scale with grid spacing,
   and why is it roughly half a cell?
