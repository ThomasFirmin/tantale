//! # Tantale
//!
//! Tantale is a Rust framework for defining and running hyperparameter
//! optimization problems. The facade re-exports the core problem and
//! experiment APIs, optimization algorithms, and procedural macros.
//!
//! A typical workflow defines an outcome and search space, creates a problem,
//! selects a sampler and stopping rule, and runs an experiment. The
//! `objective!` macro can generate the problem helpers from an objective
//! function and its domain declarations. For an introduction, see the
//! [Quick start](crate::docs::QuickStart).
//!
//! ## Tutorials
//!
//! - [Quick start](crate::docs::QuickStart)
//! - [Random search and batching](crate::docs::RandomSearch)
//! - [Grid search](crate::docs::GridSearch)
//! - [Multi-fidelity, multi-objective MO-ASHA](crate::docs::MOASHA)
//! - [Python objective integration](crate::docs::PythonIntegration)
//! - [MO-ASHA with Burn](crate::docs::TantaleBurnExample)
//! - [MO-ASHA with PyTorch](crate::docs::TantalePyTorchExample)
//!
//! ## Advanced guides
//!
//! - [Implementing a sequential sampler](crate::docs::CreateSingleSolutionSampler)
//! - [Implementing a batched sampler](crate::docs::CreateBatchedSampler)
//!
//! ## Crate re-exports
//!
//! - [`core`] contains problems, domains, outcomes, experiments, recorders,
//!   checkpointing, and stopping criteria.
//! - [`algos`] contains random and grid search, multi-fidelity samplers and
//!   pruners, and the optional Bayesian algorithms.
//! - [`macros`] contains the objective, search-space, and derive macros.
//! - `tantale::python` is available with the `py` or `pyspikes` feature.
//!
//! ## Features
//!
//! - `mpi` enables MPI communication for parallel experiments.
//! - `py` enables Python objective integration.
//! - `bayes` enables Bayesian optimization algorithms.
//!
//! See the `core`, `algos`, and `macros` module documentation for the
//! corresponding API details.

pub mod docs {
    //! Tutorials and advanced guides included in the crate-level documentation.
    #[doc = include_str!("quick_start.md")]
    #[allow(non_snake_case)]
    pub mod QuickStart {}
    #[doc = include_str!("rs_example.md")]
    #[allow(non_snake_case)]
    pub mod RandomSearch {}
    #[doc = include_str!("gs_example.md")]
    #[allow(non_snake_case)]
    pub mod GridSearch {}
    #[doc = include_str!("mo_example.md")]
    #[allow(non_snake_case)]
    pub mod MOASHA {}
    #[doc = include_str!("create_single_optimizer.md")]
    #[allow(non_snake_case)]
    pub mod CreateSingleSolutionSampler {}
    #[doc = include_str!("create_batch_optimizer.md")]
    #[allow(non_snake_case)]
    pub mod CreateBatchedSampler {}
    #[doc = include_str!("python_integration.md")]
    #[allow(non_snake_case)]
    pub mod PythonIntegration {}
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
pub use macros::basic_bounds;
#[doc(inline)]
pub use macros::fidelity_bounds;
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
#[cfg(any(feature = "py", feature = "pyspikes"))]
#[doc(inline)]
pub use tantale_python as python;
