# Random search on a mock function

In this tutorial we quickly implement a [`RandomSearch`](crate::algos::RandomSearch) for a mock function.
See [Quick Start](crate::docs::QuickStart) for in-depth explanation.

## Defining the searchspace

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
            let _a = [! a | Int(0,100, Uniform) | !]; // Defines the one domain of the searchspace. _a will receive a i64
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];
            let e = [! e | Real(1000.0,2000.0, Uniform) | !];

            // ... more variables and computation ...

            OutExample{
                obj: e, // In practice put your accuracy, MSE... here
                info: [! f | Unit(Uniform) | !],
            }
        }
    );
    // The macro expands to helpers like:
    // let sp = example::get_searchspace();
    // let obj = example::get_function();
}
```

## Define the optimization loop

```rust
# mod searchspace {
#     use tantale::core::{Bool, Cat, Int, Nat, Real, Unit, Bernoulli, Uniform};
#     use tantale::macros::{objective, Outcome, CSVWritable};
#     use serde::{Deserialize, Serialize};
# 
#     #[derive(Outcome, Debug, CSVWritable, Serialize, Deserialize)]
#     pub struct OutExample {
#         pub obj: f64,
#         pub info: f64,
#     }
# 
#     objective!(
#         pub fn example() -> OutExample {
#             let _a = [! a | Int(0,100, Uniform) | !]; // Defines the one domain of the searchspace. _a will receive a f64
#             let _b = [! b | Nat(0,100, Uniform) | !];
#             let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
#             let _d = [! d | Bool(Bernoulli(0.5)) | !];
#             let e = [! e | Real(1000.0,2000.0, Uniform) | !];
# 
#             // ... more variables and computation ...
# 
#             OutExample{
#                 obj: e,
#                 info: [! f | Unit(Uniform) | !],
#             }
#         }
#     );
#     // The macro expands to helpers like:
#     // let sp = example::get_searchspace();
#     // let obj = example::get_function();
# }
# 
# struct Cleaner {
#     path: String,
# }
# 
# impl Drop for Cleaner {
#     fn drop(&mut self) {
#         let _ = std::fs::remove_dir_all(&self.path);
#     }
# }

use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack,
    experiment::{Runable, mono}, stop::Calls,
    HasY, Solution, SolutionShape,
};
use tantale::algos::{random_search, BatchRandomSearch};
use searchspace::{get_searchspace, get_function, OutExample};

use std::sync::Arc;

# drop(Cleaner {
#     path: String::from("rs_example"),
# });
# let _clean = Cleaner {
#     path: String::from("rs_example"),
# };// In practice put your accuracy, MSE... here

let sp = get_searchspace();
let obj = get_function();
let opt = BatchRandomSearch::new(7);
let cod= random_search::codomain(|o: &OutExample| o.obj);

let stop = Calls::new(50);
let config = Arc::new(FolderConfig::new("rs_example"));
let rec = CSVRecorder::new(config.clone(), true, true, true, true);
let check = MessagePack::new(config);

let exp = mono((sp, cod), obj, opt, stop, (rec, check));
let accumulator = exp.run();
let best = accumulator.get().unwrap().get_sobj();
println!("Best solution found: f({:?}) = {}",best.get_x(), best.y().value);
```