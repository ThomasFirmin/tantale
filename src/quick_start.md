# Quick start

We consider a **maximization** problem defined by:
$$ x^\star \in \arg\max_{x \in \mathcal{X}} f(x) \enspace,$$
where $x^\star$ is the optimal [`Solution`](core::Solution), $\mathcal{X}$ is the [`Searchspace`](core::Searchspace), 
and $f$ is the [`Objective`](core::Objective).

The objective function $f$ can be written as:
$$ f: \mathcal{X} \to \mathcal{O} \enspace,$$
where $\mathcal{O}$ is the [`Outcome`](core::Outcome) such that $\mathcal{Y} \subseteq \mathcal{O}$. With $\mathcal{Y}$ the [`Codomain`](core::Codomain) of the optimization problem.
We differentiate the [`Outcome`](core::Outcome) containing extra information/meta-data, and the [`Codomain`](core::Codomain) containing only the metrics to optimize, constraints, costs...

To reach this optimal solution $x^\star$, or at least a good solution, we need an [`Optimizer`](core::Optimizer) generating candidate solutions in the search space $\mathcal{X}$.

## Define the objective function within a separate module

The [`objective!`] macro allows to define an objective function and its search space together.
The searchspace is defined by `[ name | ObjectiveDomain | OptimizerDomain ]`.

See [`hpo!`] macro for more details on the syntax of the search space definition.

Then, the macro automatically extract the search space from the function body and generate the necessary types and helper functions for optimization.
It must be called within a separate module/file. Here for example, we define an objective function `example` in a module `searchspace`:

The procedural macro generate the following items:
- `example::get_searchspace()`: returns a `Searchspace` object containing all variables defined in the function body.
- `example::get_function()`: returns the user-defined function wrapped in an [`Objective`](core::Objective) or [`Stepped`](core::Stepped) object.
Here `example` is wrapped in an [`Objective`](core::Objective).
- `example::ObjType`: the type of the solution used in the objective function.
Here the [`Searchspace`](core::Searchspace) is made of 4 variables of different types 
([`Real`](core::Real), [`Int`](core::Int), [`Cat`](core::Cat), [`Bool`](core::Bool) ,[`Unit`](core::Unit)).
So the domain of the objective function is of mixed types, and `example::ObjType` is an alias for [`Mixed`](core::Mixed).
- `example::OptType`: the type of the search space used for optimization.
Here the right-hand side of the variable definitions are empty (replaced by a [`NoDomain`](core::NoDomain) under the hood).
This indicate that the optimizer will search over the domain of the objective function (left-hand side of the variable definitions).
So the domain of the optimizer is the same as the domain of the objective function, and `example::OptType` is an alias for `example::ObjType` (which is [`Mixed`](core::Mixed)).

### Mapping between objective and optimizer domains

The previous `Obj`-`Opt` dual definition allows to have the hand on the objective and optimizer domains without any hidden relaxation.
See [`Onto`](core::Onto) for more details on the mapping between optimizer and objective domains.

### Example

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
            let _a = [! a | Int(0,100, Uniform) | !]; // Defines the one domain of the searchspace. _a will receive a f64
            let _b = [! b | Nat(0,100, Uniform) | !];
            let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
            let _d = [! d | Bool(Bernoulli(0.5)) | !];
            let e = [! e | Real(1000.0,2000.0, Uniform) | !];

            // ... more variables and computation ...

            OutExample{
                obj: e,
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

An optimization experiment is made of 7 components:
- A [`FuncWrapper`](core::FuncWrapper): [`Objective`](core::Objective) or [`Stepped`](core::Stepped).
It contains the function to optimize.
- A [`Searchspace`](core::Searchspace): it contains the variables [`Var`](core::Var) and their domains.
- A [`Codomain`](core::Codomain): it extracts the different metrics from an [`Outcome`](core::Outcome), which will be optimized.
* A [`Codomain`](core::Codomain) defines [`Single`](core::Single)-objective, [`Constrained`](core::Constrained), [`Multi`](core::Multi)-objective, [`Cost`](core::Cost)-aware optimization problems.
- An [`Optimizer`](core::Optimizer): defines a single step of the optimization process.
We distinguish between 2 types of optimizers:
* [`BatchOptimizer`](core::BatchOptimizer): generates a [`Batch`](core::Batch) of solutions at each optimization step.
* [`SequentialOptimizer`](core::SequentialOptimizer): generates a single solution at each optimization step.
- A [`Stop`](core::Stop): defines the stopping criterion of the optimization process.
- Optional [`Recorder`](core::Recorder): defines how to log the optimization process.
- Optional [`Checkpointer`](core::Checkpointer): defines how to checkpoint the optimization process.

These 7 components are then assembled together in an optimization loop using different execution strategies:
- Mono-threaded - [`mono`](core::mono): for mono-threaded execution.
- Multi-threaded - [`threaded`](core::threaded): for multi-threaded execution on a single machine. 
- Distributed - [`distributed`](core::distributed): for MPI-distributed execution on multiple machines.

### Parallelization 
The parallelization philosophy is defined by the optimizer and by the user via [`mono`](core::mono), [`threaded`](core::threaded) or [`distributed`](core::distributed):
* Synchronous: For [`BatchOptimizer`](core::BatchOptimizer) where [`Batch`](core::Batch)es of solutions are evaluated in parallel, but the optimization steps are executed sequentially.
* Asynchronous: For [`SequentialOptimizer`](core::SequentialOptimizer) where the optimizer generates on-demand new solutions as soon as one thread is free.
* Hybrid **(not yet implemented)**: For [`HybridOptimizer`](core::HybridOptimizer) where the optimizer can generate [`Batch`](core::Batch)es of solutions on demand and of variable sizes.

### Example

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
#     path: String::from("run_batch"),
# });
# let _clean = Cleaner {
#     path: String::from("run_batch"),
# };

let sp = get_searchspace();
let obj = get_function();
let opt = BatchRandomSearch::new(7);
let cod= random_search::codomain(|o: &OutExample| o.obj);

let stop = Calls::new(50);
let config = Arc::new(FolderConfig::new("run_batch"));
let rec = CSVRecorder::new(config.clone(), true, true, true, true);
let check = MessagePack::new(config);

let exp = mono((sp, cod), obj, opt, stop, (rec, check));
let accumulator = exp.run();
let best = accumulator.get().unwrap().get_sobj();
println!("Best solution found: f({:?}) ={}",best.get_x(), best.y().value);
```


### Multi-threaded example

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
# 
# use tantale::core::{
#     CSVRecorder, FolderConfig, MessagePack,
#     experiment::{Runable, threaded}, stop::Calls,
#     HasY, Solution, SolutionShape,
# };
# use tantale::algos::{random_search, BatchRandomSearch};
# use searchspace::{get_searchspace, get_function, OutExample};
# 
# use std::sync::Arc;
# 
# drop(Cleaner {
#     path: String::from("run_batch_thr"),
# });
# let _clean = Cleaner {
#     path: String::from("run_batch_thr"),
# };
# 
# let sp = get_searchspace();
# let obj = get_function();
# let opt = BatchRandomSearch::new(7);
# let cod= random_search::codomain(|o: &OutExample| o.obj);
# 
# let stop = Calls::new(50);
# let config = Arc::new(FolderConfig::new("run_batch_thr"));
# let rec = CSVRecorder::new(config.clone(), true, true, true, true);
# let check = MessagePack::new(config);
#
let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
let accumulator = exp.run();
let best = accumulator.get().unwrap().get_sobj();
println!("Best solution found: f({:?}) ={}",best.get_x(), best.y().value);
```


### MPI-distributed example

To run the MPI distributed example, you'll need first to install a distribution of [MPI](https://www.open-mpi.org/), make sure [rust mpi](https://docs.rs/mpi/latest/mpi/) works. The installation might be tricky, you might face issues with FFI and [libffi](https://github.com/libffi-rs/libffi-rs).

Then you can add `tantale` with the feature `mpi` to your project:
```console
foo@bar:~$ cargo add tantale --feature mpi
```

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
# 
use tantale::core::{
    CSVRecorder, DistSaverConfig, FolderConfig, MessagePack,
    experiment::{distributed, mpi::utils::MPIProcess}, stop::Calls,
    HasY, Solution, SolutionShape,
};
use tantale::algos::{random_search, BatchRandomSearch};
use searchspace::{get_searchspace, get_function, OutExample};

if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
    eprintln!("Skipping MPI test (not under mpirun)");
    return;
}

let proc = MPIProcess::new();
# 
# if proc.rank == 0 {
#     drop(Cleaner {
#         path: String::from("run_batch_mpi"),
#     });
#     let _clean = Cleaner {
#         path: String::from("run_batch_mpi"),
#     };
# }

let sp = get_searchspace();
let obj = get_function();
let opt = BatchRandomSearch::new(7);
let cod = random_search::codomain(|o: &OutExample| o.obj);
let stop = Calls::new(50);
let config = FolderConfig::new("run_batch_mpi").init(&proc); // <======= /!\ Be careful to the .init(&proc) here /!\
let rec = CSVRecorder::new(config.clone(), true, true, true, true);
let check = MessagePack::new(config);

let exp = distributed(&proc, (sp, cod), obj, opt, stop, (rec, check));
let accumulator = exp.run();
if let Some(acc) = accumulator{ // <==== Because workers return None !
    let best = acc.get().unwrap().get_sobj();
    println!("Best solution found: f({:?}) ={}",best.get_x(), best.y().value);
}
```

Then you have to run the binaries obtained with `cargo build` with `mpirun` or `mpiexec`.
The following example uses 4 distributed processes:

```console
foo@bar:~$ mpirun -n 4 <PATH_TO_BINARIES>/my_mpi_example
```