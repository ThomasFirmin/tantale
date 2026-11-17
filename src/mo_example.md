# Multi-fidelity, multi-objective optimization with MO-ASHA

In a single-step optimization, each candidate is evaluated once at full
cost and returns an outcome. In multi-fidelity optimization, the same
candidate can first be evaluated cheaply, then continued at a larger budget.
For example, the fidelity can be the **total number of training epochs**:
a candidate evaluated at 1 epoch may be promoted to 2, then 4. A candidate
that is not promoted may never reach the full budget.

MO-ASHA uses an inner sampler for new candidates and a multi-objective
selector to decide promotions. With minimum 1, maximum 4, and scaling 2, its
budget levels are 1, 2, and 4. An outcome with two objective fields lets the
accumulator maintain a Pareto set rather than a single best value.

## Fidelity, step, and state are different

- **Fidelity** is the `f64` budget supplied by the sampler to this call. Its
  units are chosen by the objective: epochs in the example below.
- **Step** is the status returned by the objective. `Step::Partially(n)`
  reports a positive integer progress marker, [`Step::Evaluated`](crate::core::Step::Evaluated) marks
  completion, [`Step::Pending`](crate::core::Step::Pending) means not yet evaluated, and [`Step::Discard`](crate::core::Step::Discard)
  discards the evaluation. The integer marker is not automatically the
  floating-point fidelity.
- **Function state** is per-candidate data returned by one evaluation and
  passed back on a later promotion. A training objective would save model and
  optimizer state and the completed epoch count; it must implement [`FuncState`](crate::core::FuncState)
  for checkpointing. For example, a function state could be the weights and biases
  of a neural network.

MO-ASHA expects a partial step below the maximum budget and an evaluated step
at the maximum. [`Calls`](crate::core::Calls) counts partial and completed evaluations, whereas
[`Evaluated`](crate::core::Evaluated) counts completed ones. Parallel runs may already have work in
flight when a stop condition is reached.

## A complete example

This small objective simulates training progress; replace the score
calculation and state with a real model and its training state. It follows the
structure of `examples/examples/mo_example.rs`.

```rust,no_run
mod searchspace {
    use std::{io, path::PathBuf};
    use serde::{Deserialize, Serialize};
    use tantale::core::{Bernoulli, Bool, Cat, FuncState, Int, Nat, Real, Step, Uniform};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

    #[derive(Serialize, Deserialize)]
    pub struct TrainingState {
        pub epochs: isize,
    }
    impl FuncState for TrainingState {
        fn save(&self, path: PathBuf) -> io::Result<()> {
            std::fs::write(path.join("epochs"), self.epochs.to_string())
        }
        fn load(path: PathBuf) -> io::Result<Self> {
            let value = std::fs::read_to_string(path.join("epochs"))?;
            let epochs = value.parse().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Self { epochs })
        }
    }

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct Evaluation {
        #[maximize]
        pub score: f64,
        #[minimize]
        pub cost: f64,
        #[step]
        pub step: Step,
    }

    objective!(
        pub fn train() -> (Evaluation, TrainingState) {
            let fidelity = [! FIDELITY !];
            let mut state = [! STATE !].unwrap_or(TrainingState { epochs: 0 });
            let _a = [! a | Int(-100, 100, Uniform) | !];
            let _b = [! b | Nat(0, 100, Uniform) | !];
            let _c = [! c | Cat(["relu", "tanh"], Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];
            let rate = [! rate | Real(0.0, 1.0, Uniform) | !];

            // A real trainer would resume from state.epochs to this budget.
            state.epochs = fidelity as isize;
            let step = if fidelity >= 4.0 {
                Step::Evaluated
            } else {
                Step::Partially(state.epochs)
            };
            (Evaluation { score: rate, cost: rate * rate, step }, state)
        }
    );
}

use tantale::algos::{MoAsha, RandomSearch, mo::NSGA2Selector};
use tantale::core::{
    CSVRecorder, Calls, FolderConfig, HasX, HasY, MessagePack, ParExperiment,
    SolutionShape, ThreadCommunicator,
};

// fn main() {
    let problem = searchspace::get_problem();
    let sampler = MoAsha::new(RandomSearch::new(), NSGA2Selector, 1.0, 4.0, 2.0)
        .expect("invalid budgets");
    let config = FolderConfig::new("tmp_moasha_run");
    let recorder = CSVRecorder::new(config.clone());
    let checkpointer = MessagePack::new(config);
    let experiment = ParExperiment::new(
        problem, sampler, Calls::new(25), recorder, checkpointer,
        ThreadCommunicator::new(),
    );
    if let Some(accumulator) = experiment.run() {
        let pareto_front = accumulator.get();
        for solution in pareto_front {
            println!("f( {:?} ) = {:?}", solution.get_sobj().ref_x(), solution.y().value);
        }
    }
    # let _ = std::fs::remove_dir_all("tmp_moasha_run");
// }
```

The returned state belongs to the candidate fed to the function `train`, not to the sampler. `STATE` is
`None` for a new candidate (or if something failed), and `Some(state)` on continuation. This example
saves only an epoch counter, so it does not demonstrate actual model
training. For the workspace's runnable mock objective and threaded
experiment, see `examples/examples/mo_example.rs`.
