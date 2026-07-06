# The latent-lab course

A ground-up course on JEPA world models and latent planning, taught through
this repository. It assumes you can read Python and have trained *a* neural
network before — everything else (world models, self-supervised learning,
collapse, characteristic functions, CEM, ONNX, WebGPU) is built from scratch
as it's needed.

By the end you will understand every design decision and every interesting
line of code in this project — not "how to run it" but *why it is the way it
is*, including the two places where the first attempt failed and what the
failure taught us.

## How to use it

- Lessons are ordered; later ones assume earlier ones.
- Code excerpts are taken verbatim from the repo with a `file:line` pointer —
  keep the file open beside the lesson.
- Every lesson ends with **Try it** exercises. Do them. Most take minutes and
  several are designed to make something break informatively.
- Two lessons (04, 09) center on real failures preserved in the git history
  and docs. Those are the highest-value parts of the course.

## Map

| # | Lesson | You'll be able to explain… |
|---|--------|---------------------------|
| 01 | [The big picture](01-big-picture.md) | what a world model is, why plan in latent space, what the demo shows |
| 02 | [Representations](02-representations.md) | why neither raw pixels nor ground-truth states are the right currency |
| 03 | [JEPA & collapse](03-jepa-and-collapse.md) | why predicting embeddings (not pixels) works, and why it collapses without help |
| 04 | [SIGReg](04-sigreg.md) | the characteristic-function test that prevents collapse, and its 30-line implementation |
| 05 | [The environment](05-environment.md) | dynamics designed to be ported bit-for-bit to another language |
| 06 | [Data](06-data.md) | dataset design, the validation gate, and why coverage beats size |
| 07 | [Training](07-training.md) | the loop, AMP, checkpointing, and live collapse diagnostics |
| 08 | [Measuring representations](08-measuring-representations.md) | linear probes, effective rank — and which metrics lie under collapse |
| 09 | [Planning](09-planning.md) | CEM + MPC, and the dense-cost lesson (20% → 97%) |
| 10 | [Decoder-free visualization](10-decoder-free-viz.md) | how to show imagined latents without a decoder: lookup + PCA |
| 11 | [Export & parity](11-export-and-parity.md) | ONNX dynamic axes, quantization tradeoffs, parity as a CI gate |
| 12 | [The browser](12-browser.md) | onnxruntime-web, the worker planner, the app, and the collapse switcher |
| 13 | [The engineering spine](13-engineering-spine.md) | parity fixtures, CI design, deploy, and the artifact trust chain |

## Prerequisites & setup

You'll want the repo running locally to do the exercises:

```bash
cd training && uv sync --extra gpu --extra dev    # or --extra cpu
uv run python -m latentlab.data.generate --out data/two_rooms_v1
uv run python -m latentlab.data.inspect --data data/two_rooms_v1 --png-out /tmp/inspect
```

The web-side lessons (12–13) additionally need Node 22
(`conda activate latent-lab-node` on the original dev machine) and `npm ci`
in `web/`.
