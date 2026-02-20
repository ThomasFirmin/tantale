//! # Tantale
//!
//! Tantale is a library dedicated to AutoML containing utilitaries to build search spaces, objective functions, algorithms, and parallelization. It is a core library with very few algorithms.
//! 
//! ## Tutorials
//! * [Quick start](QuickStart)
//! * [Create your own optimizer](CreateOptimizer)

#[doc = include_str!("quick_start.md")]
pub struct QuickStart {}
#[doc = include_str!("create_optimizer.md")]
pub struct CreateOptimizer {}

#[doc(inline)]
pub use tantale_core as core;
#[doc(inline)]
pub use tantale_macros as macros;
#[doc(inline)]
pub use tantale_algos as algos;
#[doc(inline)]
pub use macros::objective;
#[doc(inline)]
pub use macros::hpo;
#[doc(inline)]
pub use macros::Outcome;

