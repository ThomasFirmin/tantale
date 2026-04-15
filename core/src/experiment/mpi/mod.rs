//! MPI-related modules for distributed experiments.
//! Contains mostly utilities for [`MPIExperiment`](crate::MPIExperiment).
//!

#[cfg(feature = "mpi")]
pub mod utils;
#[cfg(feature = "mpi")]
pub mod worker;
