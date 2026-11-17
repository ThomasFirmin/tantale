# Tantale

Tantale is a Rust framework for hyperparameter optimization (HPO). It separates
the problem definition, candidate-generation algorithm, stopping rule, and
experiment runner so that these components can be combined independently.

The facade crate re-exports the workspace crates:

- `tantale::core` provides problems, domains, outcomes, experiment runners,
  recorders, and stopping criteria.
- `tantale::algos` provides samplers and pruning algorithms, including random
  search, grid search, successive halving, ASHA, MO-ASHA, and Hyperband.
- `tantale::macros` provides the `objective!`, `hpo!`, and `Outcome` macros.
- `tantale::python` is available with the `py` feature.

## Install

Add Tantale to a Rust 2024 project:

```console
cargo add tantale
```

Tantale requires Rust 1.91.1 or later.

Enable only the optional integrations your project uses:

```toml
[dependencies]
tantale = { version = "0.5.0", features = ["py"] }
```

Available facade features are `mpi`, `py` and `bayes`.
MPI runs require an MPI implementation in the build and runtime environment.

## Quick start

This complete example defines a mixed search space, maximizes a scalar
objective, runs random search, records CSV output, and checkpoints the
experiment using MessagePack.

```rust
mod searchspace {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Real, Uniform};
    use tantale::macros::{Outcome, Recordable, Schema, objective};

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct Evaluation {
        #[maximize]
        pub score: f64,
        pub note: f64,
    }

    objective!(
        pub fn tune() -> Evaluation {
            let x = [! x | Real(0.0, 1.0, Uniform) | !];
            let y = [! y | Real(0.0, 1.0, Uniform) | !];

            Evaluation {
                score: 1.0 - (x - 0.25).abs() - (y - 0.75).abs(),
                note: x + y,
            }
        }
    );
}

use searchspace::get_problem;
use tantale::algos::RandomSearch;
use tantale::core::{
    CSVRecorder, Experiment, FolderConfig, HasX, HasY, MessagePack, SolutionShape,
    experiment::Runable, stop::Calls,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = get_problem();
    let sampler = RandomSearch::new();
    let stop = Calls::new(50);

    let run_dir = format!("tantale-run-{}", std::process::id());
    let config = FolderConfig::new(&run_dir);
    let recorder = CSVRecorder::new(config.clone())
        .ok_or_else(|| std::io::Error::other("could not create CSV recorder"))?;
    let checkpointer = MessagePack::new(config)
        .ok_or_else(|| std::io::Error::other("could not create MessagePack checkpointer"))?;

    let experiment = Experiment::new(
        problem,
        sampler,
        stop,
        Some(recorder),
        Some(checkpointer),
    );
    let accumulator = experiment.run();
    if let Some(best) = accumulator.get().map(|acc| acc.get_sobj()) {
        println!("Best candidate: {:?}, score: {}", best.ref_x(), best.y().value);
    }

    Ok(())
}
```

`objective!` generates a module containing the search-space and problem
construction helpers. The `#[maximize]` field defines the optimized metric;
other outcome fields are retained as recorded data. `Calls` counts objective
calls, including partial evaluations. Use `Evaluated` when the stopping limit
should count only fully evaluated solutions.

The recorder and checkpointer constructors can fail and return `Option`.
Passing `None` to `Experiment::new` disables that component; the example
converts failure into an error instead of silently disabling recording.

## Run and validate examples

The workspace examples are maintained as executable Rust programs. From the
repository root, check them with:

```console
cargo check -p tantale_examples --examples
```

Run an example with:

```console
cargo run -p tantale_examples --example quick_start
```

The detailed tutorials and current architecture are documented in the
[Quick start](src/quick_start.md), [random search](src/rs_example.md),
[grid search](src/gs_example.md), [MO-ASHA](src/mo_example.md), and
[custom sampler](src/create_single_optimizer.md) guides.
