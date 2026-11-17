# Random search

Random search samples candidates from the declared search space without using
previous objective values to choose the next candidate. It is a useful
baseline and can be used with single-step or multi-fidelity objectives.

`RandomSearch::new()` samples one candidate at a time. For a sampler that
provides batches, use `BatchRandomSearch::new(batch_size)`. The following
example uses the batched sampler, following the runnable workspace example.

## Simple

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

use searchspace::get_problem;
use tantale::algos::RandomSearch;
use tantale::core::{
    CSVRecorder, Experiment, FolderConfig, HasX, HasY, MessagePack, SolutionShape,
    experiment::Runable, stop::Evaluated,
};

// fn main() {
    let problem = get_problem();
    let sampler = RandomSearch::new();
    let stop = Evaluated::new(50);

    let config = FolderConfig::new("tmp_random_search");
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
    # let _ = std::fs::remove_dir_all("tmp_random_search");
// }
```

## Batched version

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
use tantale::algos::BatchRandomSearch;
use tantale::core::{
    CSVRecorder, Experiment, FolderConfig, HasX, HasY, MessagePack, SolutionShape,
    experiment::Runable, stop::Evaluated,
};

// fn main() {
    let problem = get_problem();
    let sampler = BatchRandomSearch::new(7);
    let stop = Evaluated::new(50);

    let config = FolderConfig::new("tmp_batch_random_search");
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
    # let _ = std::fs::remove_dir_all("tmp_batch_random_search");
// }
```

`BatchRandomSearch` implements the `Batched` capability; a batch size of zero
does not generate candidates. `Evaluated::new(50)` stops after 50 fully evaluated solutions.
For multi-fidelity work, a partial evaluation counts as a call; use
`Evaluated::new(limit)` to stop by the number of completed solutions instead.

The complete definition of `searchspace::get_problem` is shown in the
[Quick start](quick_start.md). The source example is
`examples/examples/rs_example.rs`.
