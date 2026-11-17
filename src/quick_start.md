# Quick start

Tantale optimizes a problem consisting of a search space and an objective.
An objective returns an outcome, while the outcome's codomain identifies the
metrics and directions to optimize. Additional outcome fields can hold
information that is recorded but not optimized.

The `objective!` macro defines the objective and its search space together.
Each variable uses the form
`[! name | ObjectiveDomain | OptimizerDomain !]`. Leaving the optimizer domain
empty tells the optimizer to search the objective domain directly. A different
optimizer domain can be used to map the candidates onto the objective domain.

## Define a problem

```rust
mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Real, Uniform};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct Evaluation {
        #[maximize]
        pub score: f64,
        pub extra: f64,
    }

    objective!(
        pub fn tune() -> Evaluation {
            let x = [! x | Real(0.0, 1.0, Uniform) | !];
            let y = [! y | Real(0.0, 1.0, Uniform) | !];

            Evaluation {
                score: 1.0 - (x - 0.25).abs() - (y - 0.75).abs(),
                extra: x + y,
            }
        }
    );
}
```

The outcome derives have distinct roles. `Outcome` derives the optimization
codomain from field attributes such as `#[maximize]` and `#[minimize]`.
`Schema` and `Recordable` describe recorded columns and row values. Serde
derives support serialization used by the experiment infrastructure.

Use multiple `#[maximize]` or `#[minimize]` fields for multi-objective
problems. `#[constraint]` and `#[cost]` mark constraint and evaluation-cost
fields. For a multi-fidelity outcome, `#[step]` marks the `Step` field. The
`hpo!` macro can define a search space without declaring an objective function
in the same invocation.

The macro generates `get_searchspace`, `get_function`, and `get_problem`
helpers, along with types used by the generated function. Keep each invocation
in its own module to avoid collisions between generated helper names.

For multi-fidelity objectives, return `(Outcome, State)` and use the
`[! FIDELITY !]` and `[! STATE !]` placeholders inside the objective body. The
outcome's `#[step]` field communicates whether an evaluation is partial,
complete, pending, or discarded.

## Run an experiment

```rust
# mod searchspace {
#     use serde::{Deserialize, Serialize};
#     use tantale::core::{Real, Uniform};
#     use tantale::macros::{Outcome, Recordable, Schema, objective};
#
#     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
#     pub struct Evaluation {
#         #[maximize]
#         pub score: f64,
#         pub extra: f64,
#     }
#
#     objective!(
#         pub fn tune() -> Evaluation {
#             let x = [! x | Real(0.0, 1.0, Uniform) | !];
#             let y = [! y | Real(0.0, 1.0, Uniform) | !];
#             Evaluation {
#                 score: 1.0 - (x - 0.25).abs() - (y - 0.75).abs(),
#                 extra: x + y,
#             }
#         }
#     );
# }
use searchspace::get_problem;
use tantale::algos::RandomSearch;
use tantale::core::{
    CSVRecorder, Experiment, FolderConfig, HasX, HasY, MessagePack, SolutionShape,
    experiment::Runable, stop::Evaluated,
};

fn main() {
    let problem = get_problem();
    let sampler = RandomSearch::new();
    let stop = Evaluated::new(50);

    let config = FolderConfig::new("tmp_quick_start");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);

    let experiment = Experiment::new(
        problem,
        sampler,
        stop,
        recorder,
        checkpointer,
    );
    let accumulator = experiment.run();
    if let Some(best) = accumulator.get().map(|acc| acc.get_sobj()) {
        println!("Best candidate: {:?}, score: {}", best.ref_x(), best.y().value);
    }
    # let _ = std::fs::remove_dir_all("tmp_quick_start");
}
```

`Experiment` is the single-process runner. Its type parameters are inferred
from the problem, sampler, stop rule, recorder, and checkpointer. `Evaluated`
counts complete evaluations; `Evaluated` counts fully evaluated
solutions only.

`CSVRecorder::new` and `MessagePack::new` return `Option`. Pass `Some(value)`
to require the component, or `None` to intentionally disable it. Handle
construction failure explicitly when recording or checkpointing is required.
`FolderConfig::new` computes paths. However `FolderConfig` does not itself create output
directories, this is done by the `init` functions of the recorder and checkpointer.

## Parallel execution

Use `ParExperiment` for parallel work. The following two excerpts reuse the
`searchspace` module defined above; each is an alternative `main`, not an
additional block in the same program.

### Threaded

```rust
# mod searchspace {
#     use serde::{Deserialize, Serialize};
#     use tantale::core::{Real, Uniform};
#     use tantale::macros::{Outcome, Recordable, Schema, objective};
#
#     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
#     pub struct Evaluation {
#         #[maximize]
#         pub score: f64,
#         pub extra: f64,
#     }
#
#     objective!(
#         pub fn tune() -> Evaluation {
#             let x = [! x | Real(0.0, 1.0, Uniform) | !];
#             let y = [! y | Real(0.0, 1.0, Uniform) | !];
#             Evaluation {
#                 score: 1.0 - (x - 0.25).abs() - (y - 0.75).abs(),
#                 extra: x + y,
#             }
#         }
#     );
# }

use tantale::algos::RandomSearch;
use tantale::core::{
    CSVRecorder, Evaluated, FolderConfig, HasX, HasY, MessagePack, ParExperiment,
    SolutionShape, ThreadCommunicator,
};

fn main() {
    let problem = searchspace::get_problem();
    let config = FolderConfig::new("tmp_threaded_run");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);
    let experiment = ParExperiment::new(
        problem, RandomSearch::new(), Evaluated::new(50),
        recorder, checkpointer, ThreadCommunicator::new(),
    );
    if let Some(accumulator) = experiment.run() {
        if let Some(best) = accumulator.get() {
            println!("{:?}: {}", best.get_sobj().ref_x(), best.y().value);
        }
    }
    # let _ = std::fs::remove_dir_all("tmp_threaded_run");
}
```

`ThreadCommunicator::new()` uses the detected CPU count, with rank 0 acting as
the master and remaining ranks as workers. A machine needs at least one worker
in addition to the master. Run the workspace's complete example with:

```console
cargo run -p tantale_examples --example quick_start_threaded
```

### MPI

Enable the `mpi` feature and install an MPI implementation. Run the same
program on every MPI rank; the communicator assigns the master and workers.
Only the master receives `Some(accumulator)`. By default we suppose a Parallel File System (PFS),
you can change this asumption with `.with_pfs(false)`.
The file system is mainly used by worker for function state management,
and by the master for recording and checkpointing.

```rust,ignore
use tantale::algos::RandomSearch;
use tantale::core::{
    CSVRecorder, Evaluated, FolderConfig, HasX, HasY, MessagePack, MPICommunicator,
    ParExperiment, SolutionShape,
};

fn main() {
    let problem = searchspace::get_problem();
    let communicator = MPICommunicator::new(&problem);
    let config = FolderConfig::new("tmp_mpi_run");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);
    let experiment = ParExperiment::new(
        problem, RandomSearch::new(), Evaluated::new(50),
        recorder, checkpointer, communicator,
    );
    if let Some(accumulator) = experiment.run() {
        if let Some(best) = accumulator.get() {
            println!("{:?}: {}", best.get_sobj().ref_x(), best.y().value);
        }
    }
    # let _ = std::fs::remove_dir_all("tmp_mpi_run");
}
```

For the workspace example, build once and launch at least two processes:

```console
cargo build -p tantale_examples --example mpi_quick_start
mpiexec -n 4 target/debug/examples/mpi_quick_start
```

The workspace example checks `OMPI_COMM_WORLD_SIZE`, so this particular
command assumes Open MPI; it skips execution if that variable is absent.
Other MPI implementations may use different environment variables and
launcher options. For a separate project, build with `--features mpi`, then
launch the resulting binary with `mpiexec -n 4` (or your cluster's MPI
launcher). Choose a fresh output directory for each run. Both excerpts above
have been compiled and run in an isolated project, but are not RustDoc tests.

## Further examples

- [Random search and batching](rs_example.md)
- [Finite grid search](gs_example.md)
- [Multi-fidelity, multi-objective MO-ASHA](mo_example.md)
- [Python objective integration](python_integration.md)
