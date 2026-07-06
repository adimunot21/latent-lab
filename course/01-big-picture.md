# 01 — The big picture

## What problem is this project about?

An agent that acts in the world benefits enormously from being able to answer
one question: **"if I do X, what happens next?"** A model that answers it is
called a **world model**. Given a world model, you don't need to be *told*
how to reach a goal — you can *search*: imagine many possible action
sequences, predict where each leads, and execute the one whose predicted
outcome is closest to what you want. That search-with-a-model recipe is
**planning**, and it is qualitatively different from the reflexive
stimulus→action mapping of a plain policy network.

The catch is the phrase "what happens next". Next *what*? If your agent sees
pixels, the naive answer is "the next image", and that turns your world model
into a video generator — expensive, and mostly busy predicting things that
don't matter (textures, lighting, static backgrounds). The alternative this
project explores:

> Predict the next **representation** — a small learned vector that captures
> what matters about the scene — and never render pixels at all.

That is the **JEPA** (Joint-Embedding Predictive Architecture) idea, and this
repo is a complete, working, inspectable implementation of it:

```
frame_t  ──encoder──►  z_t ─────────┐
                                    ├──predictor──►  ẑ_{t+1}   ≈   z_{t+1}
action_t ───────────────────────────┘                              ▲
                                                                   │
frame_{t+1} ──encoder──────────────────────────────────────────────┘
```

Both the encoder and predictor are trained together so that the predictor's
guess `ẑ_{t+1}` matches the encoder's actual embedding of the next frame.
No decoder, no pixel loss, anywhere.

## The playground

Everything is demonstrated on **Two Rooms**: a dot-agent in a unit square
split by a wall with a doorway, observed as 64×64 grayscale images, actions
being small (dx, dy) nudges. It's deliberately tiny — the whole point is that
every phenomenon (representation formation, collapse, planning) stays cheap
enough to train in minutes and small enough to run in a web browser.

The finished product is https://adimunot21.github.io/latent-lab/ where you can:

1. **Drive** the agent and watch its embedding move through the latent cloud.
2. **Click a goal** — a planner searches action sequences *inside the world
   model's imagination* and steers the agent, discovering the doorway route
   without any hand-coded knowledge of walls.
3. **Swap in a deliberately broken model** — trained identically but without
   the anti-collapse regularizer — and watch the latent cloud implode to a
   single point. That failure mode, called **representation collapse**, is
   the central villain of this course.

## Why should collapse get this much attention?

Because it's the defining difficulty of the whole model family. The training
objective "make ẑ_{t+1} match z_{t+1}" has a degenerate global optimum:
*encode everything to the same constant vector*. Then prediction is trivially
perfect, the loss is essentially zero — and the representation contains
nothing. Worse, as lesson 08 shows with real numbers from this repo, several
standard quality metrics **still look fine** on a collapsed model. The loss
curve of our collapsed run looks 10,000× *better* than the healthy one.

Different JEPA-family systems fight collapse differently (momentum encoders,
stop-gradients, variance penalties). This project deliberately strips all of
those away and relies on exactly one mechanism — a distributional regularizer
called SIGReg (lesson 04) — so the cause-and-effect is unmistakable: one
knob, `lambda_reg`, separates the healthy model from the collapsed one, and
the demo ships both.

## The full pipeline you'll understand by lesson 13

```
Python (training/)                          TypeScript (web/)
──────────────────                          ─────────────────
Two Rooms env                               Two Rooms env (bit-exact port)
   │ scripted policies                            ▲ parity fixtures gate both CIs
   ▼                                              │
60k-transition dataset ──validation gate──►       │
   ▼                                              │
encoder+predictor training (MSE + SIGReg)         │
   ▼                                              │
probes / collapse metrics / rollout eval          │
   ▼                                              │
CEM planner, 97% success                    CEM planner in a Web Worker
   ▼                                              ▲
lookup table + PCA                                │ fetch + sha256 verify
   ▼                                              │
ONNX export ──parity gate──► HF Hub (pinned) ─────┘
```

## Try it

1. Open the live site. Click a goal in the opposite room. Watch the blue
   candidate paths during planning: early iterations fan out; later ones
   funnel through the doorway. You are watching cross-entropy-method search
   converge (lesson 09).
2. Switch the model dropdown to *No regularizer (final)* and click the same
   goal. Describe what the planner does now and form a hypothesis for why —
   you'll verify it in lesson 08.
3. Read `PROJECTPLAN.md` top to bottom. The checked boxes with italic
   annotations are the project's actual measured history and will anchor
   everything in this course.
