# Grid search

[`GridSearch`](crate::algos::GridSearch) enumerates values from finite grid domains. It is appropriate
when the candidate set is small enough to evaluate exhaustively or when a
fixed discrete set of values is required. Grid values are declared with
`Grid<Domain([...])>` in an [`objective!`](crate::macros::objective) invocation.

## Define finite domains

```rust
mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Int, Real, Uniform};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct Evaluation {
        #[maximize]
        pub score: f64,
    }

    objective!(
        pub fn tune() -> Evaluation {
            let _a = [! a | Grid<Int([-1_i64, 0, 1], Uniform)> | !];
            let e = [! e | Grid<Real([0.0, 0.5, 1.0], Uniform)> | !];
            Evaluation { score: e }
        }
    );
}
```

## Run the search

```rust
# mod searchspace {
#     use serde::{Deserialize, Serialize};
#     use tantale::core::{Int, Real, Uniform};
#     use tantale::macros::{Outcome, Recordable, Schema, objective};
#
#     #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
#     pub struct Evaluation {
#         #[maximize]
#         pub score: f64,
#     }
#
#     objective!(
#         pub fn tune() -> Evaluation {
#             let _a = [! a | Grid<Int([-1_i64, 0, 1], Uniform)> | !];
#             let e = [! e | Grid<Real([0.0, 0.5, 1.0], Uniform)> | !];
#             Evaluation { score: e }
#         }
#     );
# }
use searchspace::get_problem;
use tantale::algos::GridSearch;
use tantale::core::{
    CSVRecorder, Experiment, FolderConfig, HasX, HasY, MessagePack, Problem,
    SolutionShape, experiment::Runable, stop::Calls,
};

fn main() {
    let problem = get_problem();
    let sampler = GridSearch::new(problem.get_searchspace());
    let stop = Calls::new(9);

    let config = FolderConfig::new("tmp_grid_search_run");
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
    # let _ = std::fs::remove_dir_all("tmp_grid_search_run");
}
```

The two three-value dimensions form nine candidate combinations, so the stop
limit is set to nine calls. [`GridSearch::new`](crate::algos::GridSearch::new) takes the problem's search space.
The same domain must be finite and grid-backed for exhaustive enumeration.

The corresponding runnable workspace example is
`examples/examples/gs_example.rs`.
