# LPW Scope Charter — read before any change

Latent Physics World is a **high-speed, high-accuracy physics simulator
built as the substrate for training world models** (physical foundation
models). Engine reference class: MuJoCo / Genesis. We build the world;
users train on it — nothing in this repo trains a user's model.

The core is classical, verifiable rigid-body contact physics. For physics
the rigid core does not cover — an exhaustive whitelist: **deformables,
fluids, aerodynamics** (extending it is an owner decision) — the simulator
may use FEA solvers and **learned simulation** (MeshGraphNets-style
surrogates). Learning is a sanctioned *simulation method*, never anything
that outputs behavior.

## Three questions every change must pass

1. Does it make simulation more **accurate**, more **parallel/faster**,
   cover **whitelisted new physics** (via FEA or learned solvers), or make
   simulatable **worlds richer**? Worlds-richer means procedural/classical
   generation and imported assets; generative-model content (LLM scene
   composition, diffusion assets, neural sensors) needs explicit owner
   sign-off. If none of the four — reject.
2. Does it produce, contain, **execute, download, or depend on** a
   policy / skill / agent / behavior — in code, tests, CI, or examples?
   If yes — reject, even as an "example" or "validation tool" (PPO
   precedent: 9f3f072). Task solvability is demonstrated by scripted
   kinematic replay, never by running a policy artifact.
3. If it involves learning, ALL of the following or reject:
   - the learned artifact is a **drop-in solver** behind the engine
     stepper's API — same inputs/outputs as the classical solver it
     substitutes, consumable ONLY inside the step — or a bounded
     calibration residual over physical parameters. An artifact usable
     outside the stepper is a user model, not a solver;
   - validated against a classical reference, with owner-approved numeric
     gates (rollout error, conservation drift, boundary-force error)
     committed as tests BEFORE the solver merges;
   - trained on data from fixed scripted or randomized action
     distributions — no learned, optimized, or feedback-adaptive data
     collection ("adaptive data scheduling" is an exploration policy).

## Hard lines around the rigid core

- The rigid contact core stays classical in EVERY mode — no learned
  approximation of rigid contact, including opt-in "fast modes".
- "Faster" is never a justification for a learned solver; only
  whitelisted physics coverage is.
- Calibration residuals adjust physical parameters (friction, restitution,
  inertia, latency) or bounded corrections; a residual that dominates the
  classical forces is a replacement in disguise — banned.
- Benchmark verifier quantities come from classical solver paths, or are
  cross-checked against one at verification time — never asserted on a
  network's output alone.

## Module map (roles are exhaustive; extending a role = owner decision)

- `backend/` — engine adapter: budgets, snapshots, CUDA-graph stepping
- `assets/` — procedural worlds, articulated furniture, GLB/USD import,
  convex decomposition, SDF voxelization
- `perception/` — batched LiDAR / depth / segmentation
- `envs/` — batched task containers + physically checkable benchmark
  verifiers (poses, distances, joint travel; never learned). Task rewards
  exist as part of the benchmark interface users consume; training loops,
  success-rate leaderboards, and reward engineering for training are not
  this module's role.
- `broadphase/` — large-scene BVH broadphase (engine performance)
- `domain_rand/` — domain randomization + sim-to-real calibration (sysid)
- `latentphysics/neural/` — **DOES NOT EXIST YET.** It is the only place
  solver learning will ever live. Creating it — and the matching
  scope-guard carve-out — is a standalone change requiring the owner's
  explicit sign-off in that PR, never bundled into a feature PR. Until
  then the guard stays maximally strict.

## Out of scope — do not add

- RL algorithms, policy/value networks, agents, skill or curriculum
  learning, LLM-in-the-loop task generation — anything that outputs
  behavior rather than physics
- Training pipelines for user models (world models included): we ship the
  simulator they train on, not the training
- Any README claim not backed by a committed test or a reproducible
  recorded run (committed script + seed + artifact)

## Discipline

- Every gallery clip is a real run from this repo, with a hard assert in
  the script that produced it, and a label linking to the source.
- Every roadmap item is a simulator capability with a physics KPI;
  learned-solver items carry accuracy-vs-reference KPIs like everything
  else.
- `tests/test_scope_guard.py` bans learning-code signatures across the
  whole package and examples TODAY, with no carve-outs. Keep it green;
  any loosening is a standalone owner-approved change.
- This charter and an equivalent scope guard apply to every LPW
  repository, including the engine fork (`latentphysics-engine`), which
  carries its own copy of this file.
- Repo is English-only.
