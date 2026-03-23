//! This crate contains the procedural macros for Tantale, including:
//! - `Outcome` derive macro to wrap the outcome of the user-defined function with `Outcome` (see core documentation).
//! - `objective!` macro for defining objectives and searchspaces together
//! - `hpo!` macro for concise searchspace definitions

extern crate proc_macro;

mod csvwritable;
mod hpo;
mod objective;
mod optinfo;
mod optstate;
mod outcome;
mod solinfo;

/// The `Outcome` derive macro automates the implementation of result/output types for objective
/// functions. It derives traits that enable logging, serialization,
/// and multi-fidelity evaluation tracking for optimization results.
///
/// ## Purpose
///
/// Objective functions in Tantale must return an `Outcome` - a structured type describing
/// the evaluation result. The `Outcome` macro:
/// 1. Implements the [`Outcome`](crate::Outcome) trait marker
/// 2. Optionally tracks multi-fidelity evaluation state via `Step` fields
///
/// ## Quick Example
///
/// ```
/// use tantale::macros::Outcome;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Outcome, Debug, Serialize, Deserialize)]
/// pub struct ModelResult {
///     pub train_loss: f64,
///     pub val_loss: f64,
///     pub accuracy: f64,
/// }
/// ```
///
/// Generated:
/// - `impl Outcome for ModelResult` - Marks type as objective output
///
/// ## Multi-Fidelity with `Step`
///
/// For multi-fidelity optimization, a field of type `Step` tracks evaluation progress:
///
/// ```
/// use tantale::macros::{Outcome};
/// use tantale::core::Step;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Outcome, Debug, Serialize, Deserialize)]
/// pub struct ProgressiveResult {
///     pub loss: f64,
///     pub epoch: Step,  // Tracks evaluation stage
/// }
///
/// // The macro Outcome generates:
/// // impl FidOutcome for ProgressiveResult {
/// //     fn get_step(&self) -> EvalStep { self.epoch.into() }
/// // }
/// ```
#[proc_macro_derive(Outcome)]
pub fn outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    outcome::proc_outcome(input)
}

/// The `objective!` macro provides a high-level interface for defining both an optimization
/// searchspace and an objective function in a single declaration. It automatically extracts the searchspace from
/// the user-defined function body, and wrap this function into the `Objective` or `Stepped` objects.
///
/// ## Purpose
///
/// When optimizing a function, you typically need:
/// 1. A **searchspace** defining what variables to optimize
/// 2. An **objective function** that evaluates those variables
///
/// The `objective!` macro combines both, keeping the definition close to the usage:
///
/// ```ignore
/// use tantale::core::{Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
/// use tantale::macros::{objective, Outcome};
///
/// #[derive(Outcome, Debug, serde::Serialize, serde::Deserialize)]
/// struct OutExample {
///     obj: f64,
///     info: f64,
/// }
///
/// objective!(
///     pub fn example() -> OutExample {
///         let _a = [! a | Int(0,100, Uniform) | !];
///         let _b = [! b | Nat(0,100, Uniform) | !];
///         let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
///         let _d = [! d | Bool(Bernoulli(0.5)) | !];
///         let e = [! e | Real(1000.0,2000.0, Uniform) | !];
///         // ... more variables and computation ...
///         OutExample{
///             obj: e,
///             info: [! f | Real(10.0,20.0, Uniform) | !],
///         }
///     }
/// );
/// ```
///
/// ## Special Keywords
///
/// * `[! MPI_RANK !]` - Get the current MPI process rank (for distributed optimization)
/// * `[! MPI_SIZE !]` - Get the total number of MPI processes
/// * `[! STATE !]` - Access the current evaluation state in multi-fidelity optimization
///
/// ## Alternative mode
///
/// The macro also supports an alternative mode called `Grid`, allowing to define a grid search space with a similar syntax.
/// In this cases the grid can only be defined within the objective domain; while the optimizer domain should remain empty.
///
///
/// ```ignore
/// use tantale::core::{Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
/// use tantale::macros::{objective, Outcome};
///
/// #[derive(Outcome, Debug, serde::Serialize, serde::Deserialize)]
/// struct OutExample {
///     obj: f64,
///     info: f64,
/// }
///
/// objective!(
///     pub fn example() -> OutExample {
///         let _a = [! a | Grid<Int([-2, -1, 0, 1, 2], Uniform)> | !];
///         let _b = [! b | Grid<Nat([0, 1, 2, 3, 4], Uniform)> | !];
///         let _c = [! c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform)> | !];
///         let _d = [! d | Grid<Bool(Bernoulli(0.5))> | !];
///         let e = [! e | Grid<Real([1000.0, 2000.0, 3000.0, 4000.0], Uniform)> | !];
///         // ... more variables and computation ...
///         OutExample{
///             obj: e,
///             info: [! f | Grid<Real([10.0, 20.0], Uniform)> | !],
///         }
///     }
/// );
/// ```
///
/// ## Generated Output
///
/// The macro generates three public items:
///
/// ```ignore
/// pub type ObjType = /* Mixed or single domain type */;
/// pub type OptType = /* Mixed or single domain type */;
/// pub fn get_searchspace() -> Sp<ObjType, OptType> { /* ... */ }
/// pub fn get_function() -> Objective<Arc<[TypeDom<ObjType>]>, Output> { /* ... */ }
/// ```
///
/// ## Multi-Fidelity Support
///
/// ```rust,ignore
/// use tantale::core::{FuncState, Bool, Cat, Int, Nat, Real, Bernoulli, Uniform, Step};
/// use tantale::macros::{objective, Outcome};
///
/// #[derive(Outcome, Debug, serde::Serialize, serde::Deserialize)]
/// struct OutExample {
///     obj: f64,
///     info: f64,
///     step: Step,
/// }
///
/// #[derive(Serialize, Deserialize)]
/// pub struct FnState {
///     pub something: isize,
/// }
/// impl FuncState for FnState {
///     fn save(&self, path: std::path::PathBuf) -> std::io::Result<()>{
///     // Implement saving logic here (e.g., serialize to file)
///     }
///     fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
///     // Implement loading logic here (e.g., deserialize from file)
///     }
/// }
/// 
/// objective!(
///     pub fn example() -> (OutExample, FnState) {
///         let _a = [! a | Int(0,100, Uniform) | !];
///         let _b = [! b | Nat(0,100, Uniform) | !];
///         let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
///         let _d = [! d | Bool(Bernoulli(0.5)) | !];
///         let e = [! e | Real(1000.0,2000.0, Uniform) | !];
///         // ... more variables and computation ...
///         
///         // Manage the internal state
///         state.something += 1;
///         let evalstate = if state.something == 5 {Step::Evaluated} else{Step::Partially(state.something)};
///         
///         (
///             OutExample{
///                 obj: e,
///                 info: [! f | Real(10.0,20.0, Uniform) | !],
///                 step: evalstate,
///             },
///             state
///         )
///     }
/// );
/// ```
///
/// For multi-fidelity objectives (function returns `(Outcome, FuncState)`):
/// ```ignore
/// pub fn get_function() -> Stepped<Arc<[TypeDom<ObjType>]>, Output, State> { /* ... */ }
/// ```
#[proc_macro]
pub fn objective(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    objective::obj(input)
}

/// The `hpo!` procedural macro provides a declarative way to define optimization searchspaces in Tantale.
/// It generates a type-safe, dual-domain searchspace from a concise variable specification.
///
/// ## Purpose
///
/// AutoML problems involve optimizing functions across complex, mixed/heterogeneous domains where:
/// - Objective functions may require specific types (integers, categories, booleans)
/// - Optimizers migth work in its own defined space (e.g. within the unit hypercube)
/// - Variables need custom sampling and mapping strategies
///
/// The `hpo!` macro abstracts these details, allowing users to simplify variable definitions,
/// while the macro generates the appropriate searchspace implementation.
///
/// ## Syntax
///
/// ```ignore
/// hpo!(
///     var_name | ObjectiveDomain(args) | OptimizationDomain(args) ;
///     // ... more variables ...
/// );
/// ```
///
/// ## Generated Output
///
/// Each `hpo!` invocation generates three public items:
///
/// ```ignore
/// pub type ObjType = /* ... */;  // Objective domain (Mixed or single type)
/// pub type OptType = /* ... */;  // Optimizer domain (Mixed or single type)
/// pub fn get_searchspace() -> Sp<ObjType, OptType> { /* ... */ }
/// ```
///
/// The resultant `Sp<ObjType, OptType>` is a `Searchspace` containing all variables.
///
/// ## Quick Example
///
/// ```ignore
/// use tantale::macros::hpo;
/// use tantale::core::{Real, Nat, Cat, Bool, Uniform, Bernoulli};
///
/// hpo!(
///     learning_rate | Real(1e-5, 0.1, Uniform)      | ;
///     batch_size    | Nat(16, 512, Uniform)         | ;
///     optimizer     | Cat(["adam", "sgd"], Uniform) | ;
///     dropout       | Bool(Bernoulli(0.5))          | ;
/// );
///
/// // Now use the generated searchspace:
/// let sp = get_searchspace();
/// let mut rng = rand::rng();
/// let solution = sp.sample_obj(&mut rng, info);
/// ```
///
/// ## Variable Definition
///
/// Each variable is defined as:
/// ```ignore
/// name{repetition} | ObjectiveDomain(args) | OptimizerDomain(args) ;
/// ```
///
/// - **name** : Variable name used for recording.
/// - **{repetition}** : Optional integer count for replication (e.g., `decay{10}` -> `decay0..9`)
/// - **ObjectiveDomain** : Required - the domain for objective function inputs
///   - Supported types : `Real`, `Nat`, `Int`, `Bool`, `Cat`, `Unit`
///     ** The domain types must have a `new(args)` constructor that takes the specified arguments.
///     The macro automatically add new() after the type, to simplify notations and improve readability of the searchspace.**
/// - **(args)**: Domain-specific arguments (bounds, options, samplers)
/// - **OptimizerDomain** : Optional - domain for optimizer to search within
///   **The domain types must have a `new(args)` constructor that takes the specified arguments. The macro automatically add new() after the type, to simplify notations and improve readability of the searchspace.**
///   - If omitted, OptimizerDomain is set to NoDomain. Which will be further interpreted as using the ObjectiveDomain
/// - **;** : Semicolon terminates the variable definition
///
/// ## Domain Types
///
/// | Domain | TypeDom | Use Case | Example |
/// |--------|---------|----------|---------|
/// | [`Real`] | `f64` | Continuous values | `Real(0.0, 100.0, Uniform)` |
/// | [`Nat`] | `u32` | Natural numbers | `Nat(1, 20, Uniform)` |
/// | [`Int`] | `i32` | Any integers | `Int(-10, 10, Uniform)` |
/// | [`Bool`] | `bool` | Binary choices | `Bool(Bernoulli(0.5))` |
/// | [`Cat`] | `&'static str` | Categorical | `Cat(["a", "b"], Uniform)` |
/// | [`Unit`] | `f64` | Unit hypercube domain | `Unit(Uniform)` |
///
/// ## Dual Domain Architecture
///
/// Each variable has two associated domains:
///
/// - **Objective Domain (`Obj`)**: The domain of values the objective function receives
/// - **Optimizer Domain (`Opt`)**: The domain the optimizer searches within
///
/// ```ignore
/// // Optimizer searches [0,1], objective gets integer [1,20]
/// name | Nat(1, 20, Uniform) | Real(0.0, 1.0, Uniform) ;
/// ```
///
/// This enables optimizers to work in normalized continuous spaces while objectives
/// receive natural problem-specific types.
///
/// ## Mixed Domains
///
/// When variables have mixed/heterogeneous objective domains:
/// ```ignore
/// hpo!(
///     x | Real(0.0, 1.0, Uniform) | ;
///     y | Nat(0, 100, Uniform)    | ;
///     z | Bool(Uniform)           | ;
/// );
/// // ObjType = Mixed (enum for Real, Nat, Bool variants)
/// ```
///
/// When all objective domains are **the same type**, `Mixed` is not used:
/// ```ignore
/// hpo!(
///     x | Real(0.0, 1.0, Uniform) | ;
///     y | Real(-5.0, 5.0, Uniform) | ;
/// );
/// // ObjType = Real (not Mixed)
/// ```
///
/// ## Alternative mode
///
/// The macro also supports an alternative mode called `Grid`, allowing to define a grid search space with a similar syntax.
/// In this cases the grid can only be defined within the objective domain; while the optimizer domain should remain empty.
///
/// ### Example
///
/// ```ignore
/// use tantale::core::domain::{Bool, Cat, Int, Nat};
///     use tantale::core::sampler::{Bernoulli, Uniform};
///     use tantale::macros::hpo;
/// hpo!(
///         a | Grid<Int([-2_i64,-1,0,1,2], Uniform)>              | ;
///         b | Grid<Nat([1_u64,2,3], Uniform)>                    | ;
///         c | Grid<Cat(["relu", "tanh", "sigmoid"], Uniform) >   | ;
///         d | Grid<Bool(Bernoulli(0.5))>                         | ;
///     );
/// ```
///
/// ## Variable Replication
///
/// Create multiple similar variables using `{count}` syntax:
///
/// ```ignore
/// hpo!(
///     // Creates: decay0, decay1, ..., decay9
///     decay{10} | Real(-1.0, 1.0, Uniform) | ;
/// );
/// ```
///
/// Each replicated variable gets a numeric suffix starting from 0.
///
/// ## Examples
///
/// ### Example 1: Simple Hyperparameter Search
/// ```ignore
/// hpo!(
///     learning_rate | Real(1e-4, 1e-2, Uniform) | ;
///     momentum      | Real(0.7, 0.99, Uniform)  | ;
///     weight_decay  | Real(0.0, 0.01, Uniform)  | ;
/// );
/// ```
///
/// ### Example 2: Mixed/Heterogeneous Objective Domains
/// ```ignore
/// hpo!(
///     num_layers | Nat(1, 10, Uniform)                       | ;
///     dropout    | Real(0.0, 0.5, Uniform)                   | ;
///     activation | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
/// );
/// // ObjType = Mixed (enum for Nat, Real, Cat variants)
/// ```
///
/// ### Example 3: Different Objective and Optimizer Domains
/// ```ignore
/// hpo!(
///     layer_size    | Nat(10, 512, Uniform)    | Real(10.0, 20.0, Uniform) ;
///     learning_rate | Real(1e-5, 0.1, Uniform) |                         ;
///     depth         | Nat(1, 100, Uniform)     | Real(0.0, 100.0, Uniform) ;
/// );
/// ```
///
/// ### Example 4: Replications
/// ```ignore
/// hpo!(
///     decay_layer{3}   | Unit(Uniform) | ;
///     dropout_layer{4} | Unit(Uniform) | ;
/// );
/// // Total: 3 + 4 = 7 variables
/// ```
#[proc_macro]
pub fn hpo(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    hpo::hpo(input)
}

/// The `OptInfo` derive macro implements the `OptInfo` trait for types
/// representing per-iteration metadata produced by an `Optimizer`.
///
/// ## Purpose
///
/// During optimization, algorithms often generate metadata associated to a batch at each iteration,
/// that isn't part of the core optimizer state. For example the acquisition
/// value in q-Expected Improvement Bayesian optimization, associating a score to a batch of candidates solutions.
///
/// `OptInfo` provides a structured way to attach this metadata to batches of solutions.
///
/// ## Requirements
///
/// The struct must:
/// - Derive or implement `Serialize`, `Deserialize`, `Debug`, and `Default` from standard traits
///
/// ## Example
///
/// ```ignore
/// use tantale::macros::OptInfo;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(OptInfo, Serialize, Deserialize, Debug, Default)]
/// pub struct QEiInfo {
///     pub qei: f64,
/// }
/// ```
#[proc_macro_derive(OptInfo)]
pub fn optinfo(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    optinfo::proc_optinfo(input)
}

/// The `OptState` derive macro implements the `OptState` trait for types
/// representing the internal state of an `Optimizer`.
///
/// ## Purpose
///
/// The optimizer state captures all information needed to:
/// 1. **Resume optimization** after interruption or failure
/// 2. **Checkpoint progress** for long-running experiments
/// 3. **Reconstruct optimizer** with identical behavior from saved state
///
/// ## Requirements
///
/// The struct must:
/// - Derive or implement `Serialize` and `Deserialize` from `serde`
/// - Contain **complete** optimizer state for full reconstruction
/// - Include all fields necessary to resume optimization identically
///
/// ## Quick Example
///
/// ```ignore
/// use tantale::macros::OptState;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(OptState, Serialize, Deserialize)]
/// pub struct MyOptimizerState {
///     pub iteration: usize,
///     pub population: Vec<Vec<f64>>,
///     pub fitness_values: Vec<f64>,
///     pub best_solution: Vec<f64>,
/// }
/// ```
#[proc_macro_derive(OptState)]
pub fn optstate(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    optstate::proc_optstate(input)
}

/// The `SolInfo` derive macro implements the `SolInfo` trait for types
/// representing metadata associated with individual solutions during optimization.
///
/// ## Purpose
///
/// `SolInfo` provides a structured way to attach metadata to individual solutions, such as:
/// - Evaluation fidelity level
/// - Acquisition function values
/// - Uncertainty estimates
/// - Any other per-solution information relevant to the optimization process
///
/// ## Requirements
/// The struct must:
/// - Derive or implement `Serialize`, `Deserialize`, and `Debug` from standard traits
/// - Contain fields relevant to per-solution metadata
///
/// ## Example
/// ```ignore
/// use tantale::macros::SolInfo;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(SolInfo, Serialize, Deserialize, Debug)]
/// pub struct MySolInfo {
///     pub acquisition_value: f64,
///     pub uncertainty: f64,
/// }
/// ```
#[proc_macro_derive(SolInfo)]
pub fn solinfo(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    solinfo::proc_solinfo(input)
}

/// The `CSVWritable` derive macro implements the `CSVWritable` trait for types
/// that can be serialized to CSV format.
///
/// ## Purpose
///
/// `CSVWritable` provides a structured way to write objects to CSV files.
///
/// ## Quick Example
///
/// ```ignore
/// use tantale::macros::CSVWritable;
///
/// #[derive(CSVWritable)]
/// pub struct ModelResult {
///     pub train_loss: f64,
///     pub val_loss: f64,
///     pub accuracy: f64,
/// }
/// ```
///
/// Generated:
/// - `impl CSVWritable` - Enables CSV logging of results
///   - Headers: `["train_loss", "val_loss", "accuracy"]`
///   - Rows: for example `["0.523", "0.612", "0.845"]`
///
/// ## Supported Field Types
///
/// The macro automatically detects field types and generates appropriate serialization:
///
/// | Field Type | Conversion to CSV | Example |
/// |-----------|---|---------|
/// | `f64`, `f32` | to_string() | `0.523` |
/// | `i32`, `i64`, `isize` | to_string() | `-42` |
/// | `u32`, `u64`, `usize` | to_string() | `100` |
/// | `bool` | to_string() | `true` |
/// | `String` | to_string() | `"sigmoid"` |
/// | `Vec<T>` | Debug format `[...]` | `[1.0, 2.0, 3.0]` |
/// | `Step` | to_string() | `Evaluated` |
///
/// ## CSV Output Format
///
/// The macro implements [`CSVWritable`](crate::recorder::CSVWritable) for automatic result logging:
///
/// ```ignore
/// #[derive(CSVWritable)]
/// pub struct Metrics {
///     pub rmse: f64,
///     pub mae: f64,
///     pub r2: f64,
/// }
///
/// // Generates:
/// // header() -> ["rmse", "mae", "r2"]
/// // write()  -> ["0.234", "0.189", "0.856"] (for each evaluation)
/// ```
///
/// The header is derived automatically from field names, and values are serialized
/// based on their types (numeric to_string, Vec with debug format, etc.).
///
/// ## Fields with Vectors
///
/// Vec fields are serialized using debug format for multi-valued results:
///
/// ```ignore
/// #[derive(CSVWritable)]
/// pub struct PredictionResult {
///     pub predictions: Vec<f64>,  // [0.1, 0.2, 0.3, ...]
///     pub error: f64,              // 0.05
/// }
///
/// // CSV output:
/// // header() -> ["predictions", "error"]
/// // write()  -> ["[0.1, 0.2, 0.3]", "0.05"]
/// ```
///
/// ## Field Naming
///
/// Field names automatically become CSV column headers:
///
/// ```ignore
/// #[derive(CSVWritable)]
/// pub struct Metrics {
///     pub train_loss: f64,     // CSV header: "train_loss"
///     pub val_loss: f64,       // CSV header: "val_loss"
///     pub fit_time_ms: u32,    // CSV header: "fit_time_ms"
/// }
/// ```
///
/// ## Limitations
///
/// 1. **Supported types only** - Custom types require implementing CSVWritable manually
/// 2. **Field order preserved** - CSV columns match struct field order
#[proc_macro_derive(CSVWritable)]
pub fn csvwritable(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    csvwritable::proc_csvwritable(input)
}
