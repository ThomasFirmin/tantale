//! # Tantale
//!
//! Tantale is a library dedicated to AutoML containing utilitaries to build search spaces, objective functions, algorithms, and parallelization. It is a core library with very few algorithms.
//!
//! ## Tutorials
//! * [Quick start](QuickStart)
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

#[doc = include_str!("quick_start.md")]
pub struct QuickStart {}
#[doc = include_str!("create_batch_optimizer.md")]
pub struct CreateBatchOptimizer {}
#[doc = include_str!("create_seq_optimizer.md")]
pub struct CreateSequentialOptimizer {}

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
