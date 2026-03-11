**Tantale** is a Rust library for **Automated Machine Learning (AutoML)**, focusing on:
- **Hyperparameter Optimization (HPO)** — find the best hyperparameters for a model.
- **Neural Architecture Search (NAS)** *(planned)* — find the best architecture for a neural network.

## Quick start

In Tantale, we always consider by default a **maximization** problem.

### 1 — Define the objective function

The `objective!` macro extracts the search space from the function body and produces:
- `example::get_searchspace()` — the typed [`Searchspace`](tantale::core::Searchspace).
- `example::get_function()` — the user function wrapped in an [`Objective`](tantale::core::Objective) or [`Stepped`](tantale::core::Stepped).

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
Leaving the optimizer domain empty (`!]`) means the optimizer searches directly over the objective domain.

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