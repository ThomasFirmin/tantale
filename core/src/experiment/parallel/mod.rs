#[cfg(feature = "mpi")]
pub mod mpi;
#[cfg(feature = "mpi")]
pub use mpi::{MPIChannel, MPICommunicator};

pub mod threaded;
pub use threaded::{ThreadChannel, ThreadCommunicator};

pub mod parexperiment;
pub use parexperiment::{Master, ParExperiment, Worker};

pub mod parutils;
