# Changelog (Version 0.1.1)

ac7cb24 feat!(PoolMode): Now all experiments handles a PoolMode for multi-fidelity optimization, allowing to switch from in-memory to disk management of func states.

## Breaking change

### Accumulator

Renamed `BestComputed` and `ParetoComputed` with explicit names; `BestAccumulator` and `ParetoAccumulator`.

### Removed `Cat`

The `Cat` domain has been replaced by a type alias of a `GridDom<String>`; `pub type Cat = GridDom<String>`.
The usage of a `Cat` domain remains the same.

### Mixed

Added new domains (see below), to the `Mixed` enum domain.

### Checkpointer

- `load_func_state` now returns `Option<(SolId, FnState)>` instead of `Option<FnState>`.
- Added `FuncStateCheckpointer` to `WorkerCheckpointer`
- `DistCheckpointer` must have the same `FuncStateCheckpointer` as it's worker. 
- Added `new_func_state_checkpointer` function to `WorkerCheckpointer` to create a `FuncStateCheckpointer` from the worker side.
- Update `DistCheckpointer` and `WorkerCheckpointer` of `MessagePack` checkpointer.

### Funcstate

- Removed the `FuncState` macro as the `FuncState` trait now has `save` and `load` methods for user-customed saving and loading method via a path to a folder.
- Added `save` and `load` methods to the `FuncState` trait allowing the user to define custom `save` and `load` methods to checkpoint the state of a  `Stepped` function.

## New features

### Multi-objective Asha

Added the `MoAsha` algorithm, the multi-objective version of the `Asha` algorithm, from [Schmucker et al. (2021)](https://arxiv.org/pdf/2106.12639).
This is a multi-fidelity, `SequentialOptimizer` (asynchronous), and multi-objective optimization algorithm.

### GridSearch

Added the usual `GridSearch` algorithm. This is a `SequentialAlgorithm`, where solutions are iteratively selected, on-demand, from the `Grid`, made of the Cartesian product of inner `GridDom<T>`. `BatchOptimizer` is not implemented, due to the exponential growth of the batch size, due to combinatorial explosion of the grid.

### Non-dominated sorting Trait

Added the `NonDominatedSorting` trait,  and implemented it for `[T]`, where `T` has the `Dominate` trait.
The trait has two methods:
- `non_dominated_sort(&mut self) -> Vec<Vec<&T>>`: Modifying the order of the given `[T]` , and returning the fronts made of references to `T`
- `non_dominated_argsort(&self) -> Vec<Vec<usize>>`: Equivalent to the previous one. But instead uses the indices of the given `[T]` without modifying it.
  It returns the fronts containing indices of elements within the givent `[T]`.

### Non dominating binary search

Added efficient `front_binary_search` and `arg_front_binary_search` from [Zhang et al. (2014)](http://www.soft-computing.de/ENS.pdf).
These functions are used in `NonDominatedSorting` to find the front index in which a `Dominate` belongs to.

### Crowding distance

Added the `crowding_distance<T: Dominate>(values: &[&T]) -> Vec<f64>` function from [Deb et al. (2002)](https://sci2s.ugr.es/sites/default/files/files/Teaching/).
It computes the distances between solutions within a frond given by `NonDominatedSorting`

### Candidate selector

Added the `CandidateSelector` trait used to select some candidates among a slice of `[T]`, with `T` an `Dominate`.
The trait has two methods:
- `select_candidates<'a, T:Dominate>(&self, values: &'a mut [T],size: usize) -> Vec<&'a T>`: Select candidate solutions by their references.
- `arg_select_candidates<T:Dominate>(&self, values: &[T],size: usize) -> Vec<usize>`: Select candidate solutions by their index iwthin the initial slice.

### NSGA-II selector

Added the `NSGA2Selector` `CandidateSelector` selector, based on the NSGA-II algorithm from [Deb et al. (2002)](https://sci2s.ugr.es/sites/default/files/files/Teaching/OtherPostGraduateCourses/Metaheuristicas/Deb_NSGAII.pdf).

### Core

### PoolMode

- Added `Pool` enum to choose between `IdxMapPool` (keep in memory) and `LoadPool` (load from checkpoint) to manage function states in multi-fidelity optimization.
- Added `PoolMode` an `enum` of singleton `PoolMode::InMemory` and `PoolMode::Persistent`, allowing to decided wether to keep function states in volatile memory, or save and retrieve them from dist memory.
- Added extra experiment builder, `mono_with_pool`, `threaded_with_pool`, `distributed_with_pool`,  `mono_load_with_pool`, `threaded_load_with_pool`, `distributed_load_with_pool`. 
- Modified the `load!` macro to handle PoolMode by adding extra syntaxes.
- `Worker`s now use a `Pool` of function states instead of having internal `HashMap` of function states for multi-fidelity optimization

### Domains

Added new `Domain` types:
- `GridDom<T>`: A domain with a discretized slice of values of type `T`. With `T` following the `GridBounds` constraint.
  `T` must be: `PartialEq + Clone + Display + Debug + Default + Serialize + for<'a> Deserialize<'a>`
- New type aliases
 - `GridReal`: A `GridDom<f64>`
 - `GridInt`: A `GridDom<i64>`
 - `GridNat`: A `GridDom<u64>`
 - `Cat`: A `GridDom<String>`
- `Grid`: An enum of `GridReal`, `GridInt`, `GridNat`, `Cat` used within the `Grid` alternative mode of `hpo!` and `objective!`. And mostly used for `GridSearch`.
- Added `GridReal`, `GridInt`, and `GridNat` to the enum `Mixed`, so one can describe mixes of interval domains and discretized values.
- Implemented `Onto` traits between old domains and `GridDom<T>`.
- Added `grid` functions to `Bounded<T>`, to create `GridDom<T>` with `Bounded` types. Used by the `Grid` alternative mode of `hpo!` and `objective!`.

### Dominate

- Added the method `get_max_objectives` to `Dominate` trait, returning the number of optimized objectives.
- Implemented `Dominate` to `Computed`, `CompPair` and `CompLone`, allowing easy easy non dominating sorting of slices made of these types.

### Codomain

- Added constructor functions to all `ElemCodomain...`.

## Procedural macros

### `hpo!` and `objective!`
Modified the `hpo!` macro to handle the `Grid` alternative mode, describing an objective side grid only searchspace:
```rust
    hpo!(
        a | Grid<Int([-2_i64,-1,0,1,2], Uniform)>                  | ;
        b | Grid<Nat([1_u64,2,3], Uniform)>                        | ;
        c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform) >   | ;
        d | Grid<Bool(Bernoulli(0.5))>                         | ;
    );
```
By modifying `hpo!`, the `objective!` macro now also handles the `Grid` mode:
```rust
objective!(
        pub fn example() -> OutExample {
            let _a = [! a | Grid<Int([-2_i64, -1, 0 ,1, 2] , Uniform)> | !];
            let _b = [! b | Grid<Nat([0_u64, 1, 2, 3, 4] , Uniform)> | !];
            let _c = [! c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | !];
            let _d = [! d | Grid<Bool(Bernoulli(0.5))> | !];
            let e = [! e | Grid<Real([6000.0, 2000.0, 3000.0, 4000.0, 5000.0], Uniform)> | !];

            // ... more variables and computation ...

            OutExample{
                obj: e, // In practice put your accuracy, mse, rmse... here
            }
        }
    );
```

## Fixes

- Solved an issue with `load!` when the feature `mpi` is not active.
- BatchRandom search for stepped functions now always returns a batch of the right size, even if some solutions are missing within the input batch (e.g. due to Step::Error).

## Documentation

- Added extra quick example with mock functions for `RandomSearch`, `GridSearch` and `MoAsha`.
- Corrected `RandomSearch` diagram and `codomain` documentation mistakes.
- Corrected `Bernouilli` sampler doc example mistake.
- Rewrote `Asha` the Note part.
- Corrected wrong `Asha` diagram.
- Added Hyperband to the list of algorithms

## Tests

Added tests for all new features, and for `BestAccumulator` and `ParetoAccumulator`.

## Refactor

Refactored nested `match`es in `Mixed` to `tuple` `match`es `(self, item)`