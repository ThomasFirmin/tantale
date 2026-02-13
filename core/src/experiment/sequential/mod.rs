//! Module dedicated to sequential experiment implementations.
//! A sequential experiment uses a [`SequentialOptimizer`](crate::SequentialOptimizer)
//! to generate single [`SolutionShape`](crate::solution::SolutionShape) of [`Uncomputed`](crate::solution::Uncomputed)
//! at a time.
//! For threaded and distributed experiments, [`SolutionShape`](crate::solution::SolutionShape) are generated on demand,
//! when a resource requests a new solution to evaluate.
//!
//! # Parallelization philosophy
//!
//! * For [batched experiments](crate::experiment::batched) : a batch should be evaluated as quickly as possible, using all available ressources.
//!   Evaluation times for each solution in the batch are expected to be similar.
//!   * Mono-threaded: [`MonoExperiment`](crate::MonoExperiment).
//!   * Distributed: [`MPIExperiment`](crate::MPIExperiment) using [`mpi`].
//!   * Threaded: [`ThrExperiment`](crate::ThrExperiment), using [`rayon`].
//!   * Distributed + Threaded: Not yet implemented.
//! * For [sequential experiments](crate::experiment::sequential) : [`Uncomputed`](crate::solution::Uncomputed) are generated on the fly, on demand,
//!   in order to maximize ressources usage.
//!   Evaluation times for each solution are expected to vary.
//!   * Mono-threaded: [`MonoExperiment`](crate::MonoExperiment).
//!   * Distributed: [`MPIExperiment`](crate::MPIExperiment).
//!   * Threaded: [`ThrExperiment`](crate::ThrExperiment).
//!   * Distributed + Threaded: Not yet implemented.

pub mod seqevaluator;
pub mod seqfidevaluator;
pub mod seqrun;
