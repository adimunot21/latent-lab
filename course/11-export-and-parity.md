# 11 — Export & parity: getting weights out of PyTorch, provably intact

The browser can't run PyTorch. **ONNX** is the interchange: a serialized
computation graph (operators + weights) that any conforming runtime —
onnxruntime on CPU, onnxruntime-web on WebGPU/WASM — can execute. This
lesson covers the export (`export/to_onnx.py`), the *proof* it's correct
(`export/parity_test.py`), and the quantization judgment call.

## Exporting with a dynamic batch dimension

```python
torch.onnx.export(
    encoder,
    (torch.zeros(2, 1, frame_size, frame_size),),  # batch=2 so nothing folds to 1
    str(encoder_path),
    input_names=["frame"],
    output_names=["latent"],
    dynamic_axes={"frame": {0: "batch"}, "latent": {0: "batch"}},
    opset_version=17,
)
```

Three deliberate choices:

- **`dynamic_axes` on batch.** The exporter *traces* the model with an
  example input; without this annotation, the traced batch size is baked in
  as a constant. Our consumers need wildly different batches: the browser
  encodes 1 frame per keystroke but 64 per cloud-refresh, and the CEM
  planner pushes **256 candidates** through the predictor per step. One
  graph serves all of it.
- **The dummy input uses batch=2, not 1.** Tracing at batch 1 invites silent
  disasters: any op that treats size-1 specially (squeeze, broadcast) can
  fold "batch" into a constant that only works at 1. Batch 2 makes such
  folding immediately visible. A CI test drives the exported graph at batch
  300 having traced at 2 — keeping the promise honest forever.
- **opset 17**: new enough for GroupNorm/SiLU to export cleanly, old enough
  for onnxruntime-web's WebGPU and WASM backends to fully support.

Sizes: encoder 3.66 MB, predictor 0.53 MB (fp32) — a browser-friendly total
even ×4 checkpoints.

## Parity: trust nothing, measure everything

An export can *succeed* and still be wrong — subtly different op semantics,
layout mismatches, precision differences. So the exported graph is treated
as a claim to be verified: identical inputs through PyTorch and onnxruntime,
compare outputs.

```python
for batch in (1, 7, 64):
    frames = rng.standard_normal((batch, 1, frame_size, frame_size)).astype(np.float32)
    torch_out = encoder(torch.from_numpy(frames)).numpy()
    ort_out = session.run(None, {"frame": frames})[0]
    worst = max(worst, float(np.abs(torch_out - ort_out).max()))
```

Gate: fp32 max-abs-diff < 1e-4. Measured on the real checkpoints: **~1e-6** —
two orders of margin (the residual is legitimate float32 op-reordering
noise). Batch sizes 1/7/64 exercise the dynamic axis; 7 is there precisely
because it's neither the traced size nor a power of two.

**The CI trick worth stealing:** checkpoints aren't in git, but CI must run
the parity gate. Resolution — parity is a property of the *export path*, not
of particular weights, so `tests/test_onnx_parity.py` exports
**randomly-initialized** models and gates on those. Real-checkpoint parity
runs locally via the CLI before any upload.

## Quantization: a measured verdict, not a vibe

`quantize_dynamic` (weight-only int8) shrinks the encoder 3.66→0.94 MB and
the predictor 0.53→0.14 MB. Worth it? Measure, don't guess:

| artifact | int8 max-abs error | verdict |
|---|---|---|
| encoder | 0.034 | fine — latent scale is ~1, distances 5–36 |
| predictor | **0.24** | **not fine as default** |

The predictor number needs context to interpret, and context is what the
earlier lessons provide: per-step latent deltas are ~4.6 (lesson 09's
geometry), so 0.24 is ~5% error *per step* — and planning chains 12 steps,
rollouts 8, compounding it. Meanwhile the fp32 predictor costs all of
0.53 MB. So the shipped decision, recorded in `manifest.json` itself:

```
"quantization_note": "int8 predictor max-abs error ~0.24 vs per-step latent
deltas ~4.6; compounds over rollouts. Prefer fp32 predictor (0.53 MB); int8
encoder ok."
```

(Also interesting: the *collapsed* model's int8 predictor error is 10× lower
— quantization error scales with weight richness. A dead network is easy to
compress. There's a metaphor in there.)

## The bundle: one folder, fully self-describing

`export/manifest.py` assembles what the browser consumes: per-checkpoint
model files, the lookup `.bin`s (lesson 10), and `manifest.json` carrying
norm stats, env config, the fixed PCA, checkpoint labels for the UI — and
**sha256 + byte size for every file**. `export/push_to_hub.py` uploads it to
Hugging Face with a model card and prints the resulting **commit hash**; the
browser pins that hash. Why the pin and the hashes matter — the full trust
chain — is lesson 13's territory.

One chicken-and-egg detail people trip on: the manifest *inside* the upload
has `"revision": null`. A commit cannot contain its own hash. The pin lives
on the consumer side (`web/src/config.ts`), where it belongs.

## Try it

1. Run the CI parity test verbosely:
   `uv run pytest tests/test_onnx_parity.py -v -s`. Note the printed int8
   errors on random weights vs the trained-checkpoint numbers above.
2. Prove the batch=2 dummy matters: in a scratch copy of `to_onnx.py`, export
   with a batch-1 dummy and *without* `dynamic_axes`, then try batch 300
   through onnxruntime. Read the error you get — this is the failure the
   test exists to prevent.
3. Compute what int8-predictor error does to a plan: assuming ~0.24 error
   per predictor call is adversarially aligned, what's the worst-case latent
   displacement after 12 steps, and how does it compare to the latent
   distance equivalent of the 0.08 success radius (≈ 4.6)? Now argue the
   manifest note in one sentence.
