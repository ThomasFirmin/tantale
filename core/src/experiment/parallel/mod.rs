#[cfg(feature = "mpi")]
pub mod mpi;
#[cfg(feature = "mpi")]
pub use mpi::{MPICommunicator, MPIChannel};

pub mod threaded;
pub use threaded::{ThreadCommunicator, ThreadChannel};

pub mod parexperiment;
pub use parexperiment::{ParExperiment, Master, Worker};

pub mod parutils;