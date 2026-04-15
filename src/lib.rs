//! # Tantale
//!
//! *This is an early version of the Tantale library.
//! It is a core library with very few algorithms, but it contains all the necessary building blocks to create your own optimizers, search spaces, and experiments.
//! The library is designed to be flexible and extensible, allowing users to easily create their own custom components.*
//!
//! Tantale is a library dedicated to AutoML containing utilitaries to build search spaces, objective functions, algorithms, and parallelization. It is a core library with very few algorithms.
//!
//! ## Tutorials
//! * [Quick start](crate::docs::QuickStart)
//! * [Random Search example](crate::docs::RSExample)
//! * [Grid Search example](crate::docs::GSExample)
//! * [Multi-fidelity, Multi-objective optimization with MO-ASHA](crate::docs::MOExample)
//! * [Tantale $\times$ Burn](crate::docs::TantaleBurnExample)
//! * [PyTorch $\times$ Burn](crate::docs::TantalePyTorchExample)
//!
//! ## Advanced guides
//! * [Create your batched optimizer](crate::docs::CreateBatchOptimizer)
//! * [Create your sequential optimizer](crate::docs::CreateSequentialOptimizer)
//! * [Create your domain](crate::docs::CreateDomain) (Coming soon)
//! * [Create your search space](crate::docs::CreateSearchSpace) (Coming soon)
//! * [Create your solution](crate::docs::CreateSolution) (Coming soon)
//! * [Create your experiment](crate::docs::CreateExperiment) (Coming soon)
//! * [Create your stop criterion](crate::docs::CreateStopCriterion) (Coming soon)
//! * [Create your recorder](crate::docs::CreateRecorder) (Coming soon)
//! * [Create your checkpointer](crate::docs::CreateCheckpointer) (Coming soon)
//!
//! # Features
//! * `py`: Enable Python bindings and utilities. This is required to use the [`python`](crate::python) module, 
//!   which provides utilities to interface Tantale with Python code, including the [`init_python!`](crate::python::init_python) macro 
//!   for wiring Python objective functions into Tantale experiments.
//! * `mpi`: Enable MPI-based distributed optimization. This is required to use the MPI-specific execution helpers [`distributed`](crate::core::distributed).
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

pub mod docs {
    //! This module is used to include the documentation for the tutorials and advanced guides in the crate-level documentation.
    #[doc = include_str!("quick_start.md")]
    #[allow(non_snake_case)]
    pub mod QuickStart {}
    #[doc = include_str!("create_batch_optimizer.md")]
    #[allow(non_snake_case)]
    pub mod CreateBatchOptimizer {}
    #[doc = include_str!("create_seq_optimizer.md")]
    #[allow(non_snake_case)]
    pub mod CreateSequentialOptimizer {}
    #[doc = include_str!("create_domain.md")]
    #[allow(non_snake_case)]
    pub mod CreateDomain {}
    #[doc = include_str!("create_searchspace.md")]
    #[allow(non_snake_case)]
    pub mod CreateSearchSpace {}
    #[doc = include_str!("create_solution.md")]
    #[allow(non_snake_case)]
    pub mod CreateSolution {}
    #[doc = include_str!("create_experiment.md")]
    #[allow(non_snake_case)]
    pub mod CreateExperiment {}
    #[doc = include_str!("create_stop_criterion.md")]
    #[allow(non_snake_case)]
    pub mod CreateStopCriterion {}
    #[doc = include_str!("create_recorder.md")]
    #[allow(non_snake_case)]
    pub mod CreateRecorder {}
    #[doc = include_str!("create_checkpointer.md")]
    #[allow(non_snake_case)]
    pub mod CreateCheckpointer {}
    #[doc = include_str!("rs_example.md")]
    #[allow(non_snake_case)]
    pub mod RSExample {}
    #[doc = include_str!("gs_example.md")]
    #[allow(non_snake_case)]
    pub mod GSExample {}
    #[doc = include_str!("mo_example.md")]
    #[allow(non_snake_case)]
    pub mod MOExample {}
    #[doc = include_str!("tantale_x_burn.md")]
    #[allow(non_snake_case)]
    pub mod TantaleBurnExample {}
    #[doc = include_str!("tantale_x_pytorch.md")]
    #[allow(non_snake_case)]
    pub mod TantalePyTorchExample {}
}

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
#[cfg(feature = "py")]
#[doc(inline)]
pub use tantale_python as python;
