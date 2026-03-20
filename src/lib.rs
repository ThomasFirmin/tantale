//! # Tantale
//!
//! *This is the first version and release of the Tantale core library.
//! It is a core library with very few algorithms, but it contains all the necessary building blocks to create your own optimizers, search spaces, and experiments.
//! The library is designed to be flexible and extensible, allowing users to easily create their own custom components.*
//!
//! Tantale is a library dedicated to AutoML containing utilitaries to build search spaces, objective functions, algorithms, and parallelization. It is a core library with very few algorithms.
//!
//! ## Tutorials
//! * [Quick start](QuickStart)
//! * [Random Search example](RSExample)
//! * [Grid Search example](GSExample)
//! * [Multi-fidelity, Multi-objective optimization with MO-ASHA](MOExample)
//!
//! ## Advanced guides
//! * [Create your batched optimizer](CreateBatchOptimizer)
//! * [Create your sequential optimizer](CreateSequentialOptimizer)
//! * [Create your domain](CreateDomain) (Coming soon)
//! * [Create your search space](CreateSearchSpace) (Coming soon)
//! * [Create your solution](CreateSolution) (Coming soon)
//! * [Create your experiment](CreateExperiment) (Coming soon)
//! * [Create your stop criterion](CreateStopCriterion) (Coming soon)
//! * [Create your recorder](CreateRecorder) (Coming soon)
//! * [Create your checkpointer](CreateCheckpointer) (Coming soon)
//!
//! # Notes on future releases
//!
//! * The library is in its early stages and is expected to evolve.
//!
//! Interfacing Tantale with a Python API is currently under study.
//!
//! Six algorithms are currently implemented:
//! - [`random_search`](crate::algos::random_search) - A simple random search algorithm that can be used in sequential or batch mode.
//! - [`grid_search`](crate::algos::grid_search) - A simple grid search algorithm that can be used in sequential.
//! - [`sha`](crate::algos::sha) - A simple batched implementation of the SHA algorithm for multi-fidelity optimization.
//! - [`asha`](crate::algos::asha) - A simple sequential implementation of the ASHA algorithm for multi-fidelity optimization.
//! - [`moasha`](crate::algos::moasha) - The multi-objective version of the ASHA algorithm.
//! - [`hyperband`](crate::algos::hyperband) - A simple implementation of the Hyperband algorithm for multi-fidelity optimization. 
//!   It can be combined combined with the SHA or ASHA algorithms.
//! - More algorithms and extensions of the core library are coming soon.
//!
//! Tantale handle simple [`Objective`](crate::core::Objective) functions, but also [`Stepped`](crate::core::Stepped) functions,
//! which are functions that can be evaluated at different fidelity levels. This is useful for multi-fidelity optimization algorithms such as ASHA and SHA.
//!
//! For now Tantale mostly handles HyperParameter Optimization (HPO) problems,
//! but it is designed to be flexible and extensible enough to also handle Neural Architecture Search (NAS) problems.
//! **NAS is planned for a future release.**
//!
//! This first version of the core has all the necessary building blocks to create multi-objective optimizers.
//! **Multi-objective optimization is planned for a future release.**
//!
//! For now recording results is only possible in CSV format, but the library is designed to be flexible and extensible enough to also handle other formats or strategies such as JSON or SQLite.
//! **More recording strategies are planned for a future release.**
//!
//! For now checkpointing is only possible using the MessagePack format, but the library is designed to be flexible and extensible enough to also handle other formats.
//! **More checkpointing strategies are planned for a future release.**

#[doc = include_str!("quick_start.md")]
pub struct QuickStart;
#[doc = include_str!("create_batch_optimizer.md")]
pub struct CreateBatchOptimizer;
#[doc = include_str!("create_seq_optimizer.md")]
pub struct CreateSequentialOptimizer;

#[doc = include_str!("create_domain.md")]
pub struct CreateDomain;
#[doc = include_str!("create_searchspace.md")]
pub struct CreateSearchSpace;
#[doc = include_str!("create_solution.md")]
pub struct CreateSolution;
#[doc = include_str!("create_experiment.md")]
pub struct CreateExperiment;
#[doc = include_str!("create_stop_criterion.md")]
pub struct CreateStopCriterion;
#[doc = include_str!("create_recorder.md")]
pub struct CreateRecorder;
#[doc = include_str!("create_checkpointer.md")]
pub struct CreateCheckpointer;
#[doc = include_str!("rs_example.md")]
pub struct RSExample;
#[doc = include_str!("gs_example.md")]
pub struct GSExample;
#[doc = include_str!("mo_example.md")]
pub struct MOExample;

#[doc(inline)]
pub use macros::Outcome;
#[doc(inline)]
pub use macros::hpo;
#[doc(inline)]
pub use macros::objective;
#[doc(inline)]
pub use tantale_algos as algos;
#[doc(inline)]
pub use tantale_core as core;
#[doc(inline)]
pub use tantale_macros as macros;
