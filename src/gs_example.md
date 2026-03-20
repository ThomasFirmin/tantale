# Grid search on a mock function

In this tutorial we quickly implement a [`GridSearch`](algos::RandomSearch) for a mock function.
See [Quick Start](crate::QuickStart) for in-depth explanation.

## Defining the searchspace

```rust
mod searchspace {
    use tantale::core::{Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
    use tantale::macros::{objective, Outcome, CSVWritable};
    use serde::{Deserialize, Serialize};

    #[derive(Outcome, Debug, CSVWritable, Serialize, Deserialize)]
    pub struct OutExample {
        pub obj: f64,
    }

    objective!(
        pub fn example() -> OutExample {
            let _a = [! a | Grid<Int([-2_i64, -1, 0 ,1, 2] , Uniform)> | !];
            let _b = [! b | Grid<Nat([0_u64, 1, 2, 3, 4] , Uniform)> | !];
            let _c = [! c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | !];
            let _d = [! d | Grid<Bool(Bernoulli(0.5))> | !];
            let e = [! e | Grid<Real([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], Uniform)> | !];

            // ... more variables and computation ...

            OutExample{
                obj: e, // In practice put your accuracy, MSE... here
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
#     use tantale::core::{Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
#     use tantale::macros::{objective, Outcome, CSVWritable};
#     use serde::{Deserialize, Serialize};
# 
#     #[derive(Outcome, Debug, CSVWritable, Serialize, Deserialize)]
#     pub struct OutExample {
#         pub obj: f64,
#     }
# 
#     objective!(
#         pub fn example() -> OutExample {
#             let _a = [! a | Grid<Int([-2_i64, -1, 0 ,1, 2] , Uniform)> | !];
#             let _b = [! b | Grid<Nat([0_u64, 1, 2, 3, 4] , Uniform)> | !];
#             let _c = [! c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | !];
#             let _d = [! d | Grid<Bool(Bernoulli(0.5))> | !];
#             let e = [! e | Grid<Real([1000.0, 2000.0, 3000.0, 4000.0, 5000.0], Uniform)> | !];
# 
#             // ... more variables and computation ...
# 
#             OutExample{
#                 obj: e,
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
use tantale::algos::{grid_search, GridSearch};
use searchspace::{get_searchspace, get_function, OutExample};

use std::sync::Arc;

# drop(Cleaner {
#     path: String::from("gs_example"),
# });
# let _clean = Cleaner {
#     path: String::from("gs_example"),
# };

let sp = get_searchspace();
let obj = get_function();
let opt = GridSearch::new(&sp);
let cod= grid_search::codomain(|o: &OutExample| o.obj);

let stop = Calls::new(50);
let config = Arc::new(FolderConfig::new("gs_example"));
let rec = CSVRecorder::new(config.clone(), true, true, true, true);
let check = MessagePack::new(config);

let exp = mono((sp, cod), obj, opt, stop, (rec, check));
let accumulator = exp.run();
let best = accumulator.get().unwrap().get_sobj();
println!("Best solution found: f({:?}) = {}",best.get_x(), best.y().value);
```