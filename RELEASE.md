# Release 0.5.0

Version 0.5.0 is a major refactoring of Tantale. The core architecture, the
experiment API, the objective types, recording and checkpointing have been
rewritten, and every crate documentation (core, algorithms, macros, facade)
has been rewritten accordingly. The previous published version is 0.3.2
(0.4.x was not published).

This release is **not backward compatible**. The detailed list of changes is
in [CHANGELOG.md](CHANGELOG.md).

## Highlights

- A single `Sampler` trait replaces the optimizer and sampler hierarchy.
  Batching (`Batched`) and multi-fidelity pruning (`Pruner`, `Prunable`) are
  optional capabilities layered on top of it.
- Problems (`Pb`, `SimplePb`, `FidPb`) bundle a searchspace and an objective.
  `objective!` now generates `get_problem` directly.
- Two experiment types, `Experiment` and `ParExperiment`, replace the
  `mono`, `threaded` and `distributed` builders. Parallelism is selected with a
  communicator (`ThreadCommunicator` or `MPICommunicator`).
- Objectives are `OneStep` (single evaluation) or `MultiStep` (multi-fidelity,
  with a `Step` and a `FuncState`).
- Recording uses the `Schema` and `Recordable` traits (with derive macros).
  Checkpointing and recording each have one trait.
- Samplers expose their checkpointable state with `HasState` and `State`.
- New algorithms and helpers: `Premade`, `NoPruner`, `BrackettedAsha`, `BrackettedMoAsha`.
- New macros: `Schema`, `Recordable`, `basic_bounds`, `fidelity_bounds`.
- Fully rewritten tutorials, including threaded and MPI quick starts, a
  multi-fidelity explanation, sampler implementation guides, Python
  integration, and Burn and PyTorch tutorials.
- Solved major errors on SHA, ASHA, MO-ASHA, and Hyperband. 
- Due to ambiguities within references :
  - Hyperband is now strictly based on the Hyperband-SHA combination from the article. 
  - ASHA has a separated  Hyperband, i.e. bracketted version (BrackettedAsha).
  - MO-ASHA has a separated  Hyperband, i.e. bracketted version (BrackettedMoAsha).

## Migration guide

| Before (0.3.x) | Now (0.5.0) |
|----------------|-------------|
| `Optimizer`, `SingleOptimizer`, `BatchOptimizer`, `SingleSampler`, `BatchSampler` | `Sampler`, plus `Batched` for batched samplers |
| `OptState`, `OptInfo` | `State`, `HasState`, `SolInfo`, `Sampled` |
| `Objective`, `Stepped` | `OneStep`, `MultiStep` |
| `mono`, `threaded`, `distributed` (and `_with_pool`, `_load`) | `Experiment::new`, `ParExperiment::new` with a communicator, `.with_pool(PoolMode::...)` |
| `MonoExperiment`, `ThrExperiment`, `MPIExperiment` | `Experiment`, `ParExperiment` |
| `MPIProcess` | `MPICommunicator::new(&problem)` |
| separate searchspace and objective passed to the experiment | `Pb::new(searchspace, objective)` or `get_problem()` from `objective!` |
| `CSVWritable` | `Schema` and `Recordable` |
| `SeqRecorder`, `BatchRecorder`, `DistRecorder`, `NoSaver` | `Recorder` (`CSVRecorder`) |
| `MonoCheckpointer`, `ThrCheckpointer`, `DistCheckpointer` | `Checkpointer` (`MessagePack`) |
| `SaverConfig`, `DistSaverConfig`, `.init(&proc)` | `FolderConfig::new(path)` |
| `DomainSampler` | `Distribution` |
| `Step::Error` | removed |
| Python `csv_header` and `csv_write` | Python `header` and `record` |

## Things to know when upgrading

- `CSVRecorder::new` and `MessagePack::new` return `Option`, and they fail if the
  output folder already exists. Pass the values to the experiment as they are.
- Multi-fidelity wrappers (`Sha`, `Asha`, `Hyperband`, `MoAsha`) validate their
  budgets and return a `Result`.
- In `ParExperiment`, only the master process gets the accumulator from `run()`.
- A Python outcome class must provide `header()` and `record()`, the step
  attribute must hold a `PyStep`, and a state class needs `save(path)` and a
  static `load(path)`.
- `init_python!` takes `OneStep` or `MultiStep` and returns the function only.
  File paths are relative to the crate root.

## Feature flags

| Feature | Effect |
|---------|--------|
| `mpi` | MPI parallel execution (`MPICommunicator`) |
| `py` | Python integration (`tantale::python`, `pyhpo!`, `init_python!`) |
| `bayes` | Bayesian algorithms |