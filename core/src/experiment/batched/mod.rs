//! Module dedicated to batched experiment implementations.
//! A batched experiment uses a [`BatchOptimizer`](crate::BatchOptimizer)
//! to generate [`Batch`](crate::Batch)es of solutions.
//! The batches of solutions are evaluated sequentially.
//! For threaded and distributed experiments, batches are still evaluated sequentially, but
//! [`Uncomputed`](crate::Uncomputed) solutions within each batch are evaluated in parallel
//! using multiple threads or distributed processes.
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


pub mod batchevaluator;
pub mod batchfidevaluator;
pub mod batchrun;
