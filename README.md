# Tantale

**Tantale** is a Rust library for **Automated Machine Learning (AutoML)**, focusing on:
- **Hyperparameter Optimization (HPO)** — find the best hyperparameters for a model.
- **Neural Architecture Search (NAS)** *(planned)* — find the best architecture for a neural network.

> **Why Rust?**
> AutoML is computationally expensive, and all types are statically known from the search space definition.
> AutoML involves long run and costly experiments. Thus, Rust's provide higher rank security with its strong
> type system which catches errors at compile time, eliminates whole classes of runtime bugs, 
> and delivers the performance needed when running thousands of evaluations or benchmarking algorithms without an expensive objective.

---

## Crate structure

Tantale is a workspace of three crates, re-exported from this top-level crate:

| Sub-crate | Re-exported as | Role |
|---|---|---|
| `tantale_core` | `tantale::core` | Core abstractions: domains, search spaces, solutions, objectives, optimizers, experiments |
| `tantale_macros` | `tantale::macros` | Procedural macros: `objective!`, `hpo!`, `Outcome`... |
| `tantale_algos` | `tantale::algos` | Concrete algorithms: Random Search, SHA, ASHA, Hyperband |
| `tantale_python` | `tantale::python` | Python bindings with [pyo3](https://pyo3.rs) |

---

## Feature flags

| Flag | Description |
|---|---|
| `mpi` | Enables MPI-based distributed execution across multiple machines. Requires a local MPI installation (e.g. OpenMPI). Propagates to `tantale_core` and `tantale_algos`. |
| `py` | Enables Python bindings with [pyo3](https://pyo3.rs) allowing to optimize Python functions |

---

## Installation

```console
foo@bar:~$ cargo add tantale
```

Minimum supported Rust version: **1.91.1** (2024 edition).

---

## Algorithms

| Algorithm | Module | Type | Multi-fidelity | Multi-objective | Description |
|---|---|---|---|---|---|
| Random Search | `tantale::algos::random_search` | Sequential | No |  No | Sequential Random sampling |
| BatchRandomSearch | `tantale::algos::random_search` | Batched | No |  No | Random sampling in fixed-size batches |
| GridSearch | `tantale::algos::grid_search` | Sequential | No |  No | Deterministic sequential sampling of a grid of solutions|
| SHA | `tantale::algos::sha` | Batched | Yes |  No |[Successive Halving](https://arxiv.org/abs/1502.07943): bracket-based multi-fidelity pruning |
| ASHA | `tantale::algos::asha` | Sequential | Yes |  No |[Asynchronous SHA](https://arxiv.org/abs/1810.05934): on-demand asynchronous pruning |
| Hyperband | `tantale::algos::hyperband` | Batched / Sequential | Yes |  No |[Hyperband](https://arxiv.org/abs/1603.06212): ensemble of SHA/ASHA brackets |
| MO-ASHA | `tantale::algos::moasha` | Sequential | Yes | Yes | [MO-ASHA](https://arxiv.org/pdf/2106.12639): Multi-objective ASHA |

---

## Quick start

### 1 — Define the objective function

The `objective!` macro extracts the search space from the function body and produces:
- `example::get_searchspace()` — the typed `Searchspace`.
- `example::get_function()` — the user function wrapped in an `Objective` or `Stepped`.

It must be called inside a dedicated module or file.

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
            let _a = [! a | Int(0, 100, Uniform)                      | !];
            let _b = [! b | Nat(0, 100, Uniform)                      | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform)  | !];
            let _d = [! d | Bool(Bernoulli(0.5))                       | !];
            let e  = [! e | Real(1000.0, 2000.0, Uniform)              | !];

            OutExample {
                obj:  e,
                info: [! f | Unit(Uniform) | !],
            }
        }
    );
}
```

Each variable is declared with `[! name | ObjectiveDomain | OptimizerDomain !]`.
Leaving the optimizer domain empty (`| !]`) means the optimizer searches directly over the objective domain.

### 2 — Assemble and run the experiment

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
let opt = BatchRandomSearch::new(7);                              // batch size = 7
let cod = random_search::codomain(|o: &OutExample| o.obj);       // maximize o.obj

let stop   = Calls::new(50);                                     // stop after 50 evaluations
let config = Arc::new(FolderConfig::new("run_batch"));
let rec    = CSVRecorder::new(config.clone(), true, true, true, true);
let check  = MessagePack::new(config);

let exp         = mono((sp, cod), obj, opt, stop, (rec, check)); // mono-threaded
let accumulator = exp.run();
let best        = accumulator.get().unwrap().get_sobj();
println!("Best: f({:?}) = {}", best.get_x(), best.y().value);
```

## Experiment components

An experiment is composed of up to 7 components:

| Component | Trait | Role |
|---|---|---|
| Search space | `Searchspace` | Defines the input domain |
| Codomain | `Codomain` | Extracts metrics (objective, constraints, cost) from an `Outcome` |
| Function | `FuncWrapper` | The function to optimize (`Objective` or `Stepped`) |
| Optimizer | `Optimizer` | Generates candidate solutions (`BatchOptimizer` or `SequentialOptimizer`) |
| Stop | `Stop` | Defines when to terminate (`Calls`, `Evaluated`, …) |
| Recorder *(optional)* | `Recorder` | Logs solutions to disk (CSV, …) |
| Checkpointer *(optional)* | `Checkpointer` | Saves and restores experiment state (MessagePack) |

### Execution contexts

| Function | Experiment | Parallelism |
|---|---|---|
| `mono` | `MonoExperiment` | Single-threaded |
| `threaded` | `ThrExperiment` | Multi-threaded (one machine) |
| `distributed` | `MPIExperiment` | MPI-distributed (multiple machines) |

### Parallelization philosophy

- **Synchronous** (`BatchOptimizer`): a full batch is evaluated in parallel before the next optimization step.
- **Asynchronous** (`SequentialOptimizer`): new solutions are generated on demand as soon as a thread/process becomes free.

---

---

## Macros

| Macro | Kind | Description |
|---|---|---|
| `objective!` | Declarative | Defines a search space and wraps a function into `Objective` / `Stepped` |
| `hpo!` | Declarative | Concise search space definition (without a function body) |
| `pyhpo!` | Declarative | Same as `hpo!` but for Python bindings (`py` feature) |
| `Outcome` | Derive | Implements the `Outcome` trait for a result struct |
| `CSVWritable` | Derive | Enables CSV logging of an `Outcome` via `CSVRecorder` |
| `OptState` | Derive | Implements checkpointing for an optimizer's internal state |
| `OptInfo` | Derive | Attaches metadata to solutions produced by an optimizer |
| `SolInfo` | Derive | Attaches per-solution metadata |

---

---

## License

**CECILL-C**

CeCILL-C functions as a “weak” copyleft comparable to the Mozilla Public License. Only modified parts fall within its scope, while linked modules may adopt their own license. This decoupling facilitates the use of the library in proprietary products.

See [cecill.info](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html) for the full text.
