# Changelog

## Release 0.1.1

### Breaking Changes

#### Core

- Renamed `BestComputed` to `BestAccumulator` and `ParetoComputed` to `ParetoAccumulator`.
- Replaced `Cat` with the type alias `pub type Cat = GridDom<String>`. Existing usage remains unchanged.
- Extended `Mixed` with the new domain variants listed below.
- Updated checkpointing APIs:
    - `load_func_state` now returns `Option<(SolId, FnState)>` instead of `Option<FnState>`.
    - `WorkerCheckpointer` now includes `FuncStateCheckpointer`.
    - `DistCheckpointer` must use the same `FuncStateCheckpointer` as its worker counterpart.
    - Added `WorkerCheckpointer::new_func_state_checkpointer`.
    - Updated the `MessagePack` `DistCheckpointer` and `WorkerCheckpointer` implementations.
- Removed the `FuncState` macro.
- Added `save` and `load` methods to the `FuncState` trait so users can define custom checkpoint method for `Stepped` functions.

### Added

#### Algorithms

- Added `MoAsha`, the multi-objective variant of `Asha` from [Schmucker et al. (2021)](https://arxiv.org/pdf/2106.12639). It is a multi-fidelity, asynchronous `SequentialOptimizer`.
- Added `GridSearch`, a `SequentialAlgorithm` that iterates over the Cartesian product of `GridDom<T>` values on demand. `BatchOptimizer` is intentionally not implemented because the grid grows combinatorially.
- Added the `NonDominatedSorting` trait for `[T]` where `T: Dominate`, with:
    - `non_dominated_sort(&mut self) -> Vec<Vec<&T>>`
    - `non_dominated_argsort(&self) -> Vec<Vec<usize>>`
- Added `front_binary_search` and `arg_front_binary_search` from [Zhang et al. (2014)](http://www.soft-computing.de/ENS.pdf) for efficient front insertion in non-dominated sorting.
- Added `crowding_distance<T: Dominate>(values: &[&T]) -> Vec<f64>` from [Deb et al. (2002)](https://sci2s.ugr.es/sites/default/files/files/Teaching/) to compute distances within a non-dominated front.
- Added the `CandidateSelector` trait with:
    - `select_candidates<'a, T: Dominate>(&self, values: &'a mut [T], size: usize) -> Vec<&'a T>`
    - `arg_select_candidates<T: Dominate>(&self, values: &[T], size: usize) -> Vec<usize>`
- Added `NSGA2Selector`, a `CandidateSelector` based on NSGA-II from [Deb et al. (2002)](https://sci2s.ugr.es/sites/default/files/files/Teaching/OtherPostGraduateCourses/Metaheuristicas/Deb_NSGAII.pdf).

#### Core

- Added `Pool` to manage function states in multi-fidelity optimization with `IdxMapPool` or `LoadPool`.
- Added `PoolMode` with `InMemory` and `Persistent` modes.
- Added experiment builders: `mono_with_pool`, `threaded_with_pool`, `distributed_with_pool`, `mono_load_with_pool`, `threaded_load_with_pool`, and `distributed_load_with_pool`.
- Updated `load!` to support `PoolMode`.
- Workers now use a `Pool` for function states instead of an internal `HashMap`.
- Added `GridDom<T>`, a discretized domain type for values satisfying `GridBounds`.
- Added type aliases: `GridReal`, `GridInt`, `GridNat`, and `Cat`.
- Added `Grid`, the enum used by the `Grid` mode in `hpo!` and `objective!`, and by `GridSearch`.
- Added `GridReal`, `GridInt`, and `GridNat` to `Mixed`.
- Implemented `Onto` conversions between existing domains and `GridDom<T>`.
- Added `grid` constructors on `Bounded<T>`.
- Added `Dominate::get_max_objectives`.
- Implemented `Dominate` for `Computed`, `CompPair`, and `CompLone`.
- Added constructor functions for all `ElemCodomain...` types.

#### Macros

- Extended `hpo!` to support the `Grid` mode for grid-only search spaces.
- Extended `objective!` to support the same `Grid` mode.

### Fixed

- Fixed `load!` when the `mpi` feature is disabled.
- Fixed `BatchRandom` for stepped functions so it always returns a batch of the requested size, even when some input solutions are missing, for example after `Step::Error`.

### Documentation

- Added quick examples with mock functions for `RandomSearch`, `GridSearch`, and `MoAsha`.
- Corrected the `RandomSearch` diagram and codomain documentation.
- Corrected the `Bernouilli` sampler example.
- Rewrote the note section for `Asha`.
- Corrected the `Asha` diagram.
- Added `Hyperband` to the algorithm list.

### Tests

- Added tests for all new features.
- Added tests for `BestAccumulator` and `ParetoAccumulator`.

### Refactoring

- Refactored nested `match` expressions in `Mixed` into tuple matches on `(self, item)`.


## Hotfix 0.1.11

### Documentation

- Added a comprehensive example of a MPI-distributed, asynchronous, multi-fidelity and multi-objective HPO with a Burn network trained on MNIST.

### Fixed

- `MoAsha`: When generating a random sample at the first front when k=0. The sample was associated to the budget of the previously sampled solution.
- `MessagePack`: The `remove_func_state` was removing a `.md` file instead of the folder containing the function state.
- `seqfidevaluator`: If recursive_send was unable to send something to all idle, then it will wait for an incomming message containing a computed that might help unstuck other processes. It was previously stuck within an infinite loop.

### Added

- The `objective!` macro now handles generics from the user-defined objective function


## Release 0.1.2 : PyTantale

### 🚀 Features

- *(pytantale)* Added the `python` module. Allowing optimizing Python function, with Python Outcome (`PyOutcome`), and Python States (`PyState`).
- *(pyhpo!)* Added the `pyhpo!` procedural macro. Similar to `hpo!`, but for python integration.
- *(Runable)* Added the `extract` method to [`Runable`] trait to retrieve components of an experiment.
- *(tantale)* Added the `py` feature, allowing to activate or not python function handling with [pyo3](https://pyo3.rs/v0.28.3/).
- *(hpo!)* The macro now creates an `indices` submodule within the user-defined searchspace module. It contains internal index within searchspace as constants with UPPERCASE variable name.
- *(HasId)*  The `HasId` trait now implements a `ref_id` and `mut_ref_id` methods allowing to borrow and mutably borrow an `Id`.
- *(HasStepId)* [*breaking*] Added the `HasStepId` describing objects containing a `StepId`. Modified all multi-fidelity run and evaluator to further constrained solutions and shapes to implement `HasStepId`.
- *(run/load)* The constructor functions `mono`, `threaded`, `distributed` and related functions are now generic over `Id`. Same for `load!` macro.
- *(StepId)* Added the `StepId` trait describing an Id able to track how many times how function was partially evaluated by a stepped function.
- *(StepSId)* Added the `StepSId` struct implementing `StepId`, and used within multi-fidelity related objects.
- *(Bool)* Added From<Grid>.
- *(Bounded)* Added From<Mixed> for Real, Nat, Int, Unit.
- *(GridDom)* Added From<Mixed> for GridReal, GridNat, GridInt.
- *(Unit)* Added From<Mixed>.
- *(macros)* Added the `mpi` and `py` features to `macros` crate.
- *(MessagePack)* All unwraps for function returning Results<_,CheckPointError> are replaced by a map_err to CheckpointError.
- *(experiment)* Added type alias ExpComponent.

### 🐛 Bug Fixes

- *(asha)* At initialization if the only budget is the minimum one, then consider minimum and maximum budgets `[bmin,bmax]` .
- *(asha)* Solved an issue when `k==0`, the current budget was not reseted to `bmin`.
- *(hyperband)* Now `first_step` method initializes inner batch optimizer's batch size.
- *(sha)* Now SHA builds a vec of all available budgets. Preventing the case were only minimum budget is computable, and replaced with `bmax`, instead of considering `[bmin, bmax]`.
- *(GridDomDistribution)* [*breaking*] Modified generic `T` bounds from `BoundedBounds` to `GridBounds`.
- *(Outcome)* [*breaking*] The `Outcome` derive macro does not handle generics anymore. This simplifies Python integration.
- *(FolderConfig)* [*breaking*] The given path is now transformed with `std::path::absolute` to get the absolute path even if it does not exists. Path existence is checked later.
- *(pytantale)* [*breaking*] Remove `pyconfig` and put everything in init_python! macro.
- *(StepId)* [*breaking*] Replaced `SId` by `StepSId` for all `Fidelity` based experiment. This allows linking by `id` and `id_step` recorded solutions.

### 🚜 Refactor

- *(tests)* Added a cleaner module, containing the tmp folder Cleaner + python tests.

### 📚 Documentation

- *(GridDom)* Added documentation for GridReal, GridInt and GridNat.
- *(tantalexburn)* Minor mistake.
- *(lib)* Solved list issue.
- *(tutorial)* Update multi-objective optimizer tutorials with StepSId.
- *(hyperband)* Rewrote pseudo code.

### 🧪 Testing

- *(reader)* Now all experiment output are verified using the same function, using theoretical number of expected outputs. Or + epsilon when randomnes.
- *(fidelity)* Update all fidelity tests with StepSId.
