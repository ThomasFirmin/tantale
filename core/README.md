# Tantale core

`tantale_core` is the foundational crate of the [Tantale](https://github.com/your-org/tantale) AutoML library.
It defines the modeling layer (domains, variables, search spaces, solutions), the optimization layer (objectives, optimizers, stop criteria), and the execution layer (experiments, recorders, checkpointing).

> **Note:** This is the first release of the core library. It is stable and usable but designed to evolve. Most users will interact with the top-level `tantale` crate, which re-exports this crate as `tantale::core`.

---

## Architecture overview

```
tantale_core
|--- domain        — Domain definitions (Real, Int, Nat, Cat, Bool, Unit, Mixed…) and Codomain (objective outputs)
|--- sampler       — Sampling utilities and probability distributions (Uniform, Bernoulli…)
|--- variable      — Var: relates objective and optimizer domains via Onto mappings
|--- searchspace   — Searchspace: composition of Vars into a typed search space (Sp, SpPar)
|--- solution      — Solution shapes, identifiers, fidelity, computed wrappers (Uncomputed, Computed, Batch…)
|--- objective     — Objective functions, multi-step evaluation (Stepped), outcomes
|--- optimizer     — Optimizer traits (BatchOptimizer, SequentialOptimizer) and state metadata
|--- stop          — Stop criteria (Calls, Evaluated…)
|--- experiment    — Execution pipelines: mono, threaded, distributed (MPI)
|--- recorder      — Result logging utilities (CSVRecorder, NoSaver…)
|--- checkpointer  — Experiment checkpointing (MessagePack)
```

---

## Main building blocks

| Category | Items |
|---|---|
| **Modeling** | `Domain`, `Var`, `Searchspace`, `Solution` |
| **Objective** | `Objective`, `Outcome`, `Stepped`, `Step` |
| **Optimization** | `Optimizer`, `BatchOptimizer`, `SequentialOptimizer` |
| **Execution** | `Runable`, `MonoExperiment`, `ThrExperiment`, `MPIExperiment` |
| **Support** | `Recorder`, `Checkpointer`, `Stop`, `FolderConfig` |

---

## Feature flags

| Flag | Description |
|---|---|
| `mpi` | Enables MPI-based distributed execution via `MPIExperiment` and related helpers. Requires a local MPI installation (e.g. OpenMPI). |

---

## Dependencies

| Crate | Role |
|---|---|
| `num` | Traits for numeric types |
| `rand` | RNG traits used by domains and samplers to generate valid values |
| `rayon` | Parallel evaluation in synchronous batched optimization and `SpPar` utilities |
| `csv` | CSV-backed `CSVRecorder` |
| `serde` | Serialization / deserialization for all core types (required for checkpointing) |
| `rmp-serde` | MessagePack serialization used by `MessagePack` for compact checkpointing |
| `mpi` *(features = mpi)*| Message parsing interface, for distributed computing |
| `bincode` *(optional)* | Binary serialization for MPI message passing |
| `bitvec` *(optional)* | Bit-level storage used by MPI idle-worker tracking |
| `num_cpus` | Detects available CPU cores to size thread pools in multi-threaded execution |
| `indexmap` | Used for managing pools of funcstate, in order to implement FIFO pruning of old states | 

---

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
tantale_core = { version = "0.1.1" }
# or, for MPI-distributed execution:
tantale_core = { version = "0.1.1", features = ["mpi"] }
```

Prefer the top-level `tantale` crate:

```toml
[dependencies]
tantale = { version = "0.1.1" }
```

---

## Quick start

### 1 — Define the objective function

The `objective!` macro builds a `Searchspace` and wraps a user-defined function into an `Objective`.
It must be called inside a dedicated module or file so that the generated code does not conflict with the surrounding scope.

Each variable is declared with the syntax `[! name | ObjectiveDomain | OptimizerDomain !]`.
When the optimizer domain is left empty (`!]`), the optimizer searches directly over the objective domain.

```rust
mod searchspace {
    use tantale::core::{Bool, Cat, Int, Nat, Real, Unit, Bernoulli, Uniform};
    use tantale::macros::{objective, Outcome, CSVWritable};
    use serde::{Deserialize, Serialize};

    #[derive(Outcome, Debug, CSVWritable, Serialize, Deserialize)]
    pub struct OutExample {
        pub obj: f64,
        pub info: f64,
    }

    objective!(
        pub fn example() -> OutExample {
            let _a = [! a | Int(0,100,    Uniform)                    | !];
            let _b = [! b | Nat(0,100,    Uniform)                    | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5))                      | !];
            let e  = [! e | Real(1000.0, 2000.0, Uniform)             | !];

            OutExample {
                obj:  e,
                info: [! f | Unit(Uniform) | !],
            }
        }
    );
}
```

The macro generates:
- `example::get_searchspace()` — returns the `Searchspace` object.
- `example::get_function()` — returns the function wrapped in an `Objective`.
- `example::ObjType` / `example::OptType` — type aliases for the objective and optimizer solution domains.

### 2 — Run a batch optimization experiment

```rust
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::{Runable, mono}, stop::Calls,
    HasY, Solution, SolutionShape,
};
use tantale::algos::{random_search, BatchRandomSearch};
use searchspace::{get_searchspace, get_function, OutExample};
use std::sync::Arc;

let sp  = get_searchspace();
let obj = get_function();
let opt = BatchRandomSearch::new(7);
let cod = random_search::codomain(|o: &OutExample| o.obj);

let stop   = Calls::new(50);
let config = Arc::new(FolderConfig::new("run_batch"));
let rec    = CSVRecorder::new(config.clone(), true, true, true, true);
let check  = MessagePack::new(config);

let exp         = mono((sp, cod), obj, opt, stop, (rec, check));
let accumulator = exp.run();
let best        = accumulator.get().unwrap().get_sobj();
println!("Best: f({:?}) = {}", best.get_x(), best.y().value);
```

---

## Execution contexts

| Function | Type | Description |
|---|---|---|
| `mono` | `MonoExperiment` | Single-threaded, sequential or batched |
| `threaded` | `ThrExperiment` | Multi-threaded on a single machine |
| `distributed` | `MPIExperiment` | MPI-distributed across multiple machines (`mpi` feature) |

### Parallelization philosophy

- **Synchronous** (`BatchOptimizer`): batches of solutions are evaluated in parallel; optimization steps are sequential by batch.
- **Asynchronous** (`SequentialOptimizer`): solutions are generated on demand as threads/processes become free.

---

## Optimization concepts

### Searchspace and Domains

A `Searchspace` is a product of typed `Var`s. Each variable has:
- An **objective domain** — the domain in which the objective function is evaluated (e.g. `Real`, `Int`, `Cat`).
- An **optimizer domain** — the domain in which the optimizer searches. When empty, it matches the objective domain.
  The mapping between the two is handled by `Onto`.

Mixed-type search spaces are represented by `Mixed`.

### Codomain

A `Codomain` extracts the metrics to optimize from an `Outcome`. It defines the type of optimization problem:

| Type | Description |
|---|---|
| `SingleCodomain` | Single real-valued objective |
| `MultiCodomain` | Multi-objective (Pareto) |
| `CostCodomain` | Cost-aware (objective + cost) |
| `Constrained` | Constrained optimization |

### Stopping criteria

| Criterion | Description |
|---|---|
| `Calls` | Stop after a fixed number of function evaluations |
| `Evaluated` | Stop after a fixed number of fully evaluated solutions (multi-fidelity) |

---

## License

**CECILL-C**

CeCILL-C functions as a “weak” copyleft comparable to the Mozilla Public License. Only modified parts fall within its scope, while linked modules may adopt their own license. This decoupling facilitates the use of the library in proprietary products.

See [cecill.info](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html) for the full text.