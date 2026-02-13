//! # Saver Configuration
//!
//! Configuration utilities for file-based persistence used by [`Recorder`](crate::Recorder)
//! and [`Checkpointer`](crate::Checkpointer). This module defines simple traits that
//! construct configuration objects and a default folder layout used throughout Tantale.
//!
//! ## Overview
//!
//! Tantale stores experiment artifacts in a consistent folder hierarchy that includes:
//! - Evaluation records produced by a [`Recorder`](crate::Recorder)
//! - Checkpoint state produced by a [`Checkpointer`](crate::Checkpointer)
//! - Optional worker folders for distributed (MPI) experiments
//!
//! The central configuration type is [`FolderConfig`], which provides paths for recorder
//! outputs, checkpointer outputs, and worker checkpoints. This configuration is typically
//! wrapped in an [`Arc`] for cheap sharing across components.
//!
//! ## Trait Roles
//!
//! - [`SaverConfig`] is the base trait for constructing sharable configuration objects.
//! - [`DistSaverConfig`] extends [`SaverConfig`] with MPI-aware initialization.
//!
//! ## Default Folder Layout
//!
//! A typical layout created by [`FolderConfig::new`] looks like this:
//!
//! ```text
//! path/
//! |-- recorder/
//! |   |-- obj.csv
//! |   |-- opt.csv
//! |   |-- info.csv
//! |   |-- out.csv
//! |-- checkpointer/
//!     |-- state_opt.mp
//!     |-- state_stp.mp
//!     |-- state_eval.mp
//!     |-- state_param.mp
//! ```
//!
//! With the `mpi` feature enabled, distributed runs can add rank-specific subfolders.
//!
//! ## See Also
//!
//! - [`Recorder`](crate::Recorder) - File-based evaluation recorder
//! - [`Checkpointer`](crate::Checkpointer) - Experiment checkpointing
//! - [`MessagePack`](crate::checkpointer::MessagePack) - Default checkpointer implementation
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use crate::experiment::mpi::utils::MPIProcess;

/// Base trait for configuration objects used by recorders and checkpointers.
///
/// Implementations of this trait provide an initialization method that returns
/// a shareable configuration instance wrapped in [`Arc`].
///
/// # Example
///
/// ```ignore
/// let config = FolderConfig::new("./my_run").init();
/// let checkpointer = MessagePack::new(config.clone());
/// let recorder = CSVRecorder::new(config);
/// ```
pub trait SaverConfig {
    /// Initialize the configuration and return it in a shared [`Arc`].
    fn init(self) -> Arc<Self>;
}

#[cfg(feature = "mpi")]
/// MPI-aware configuration for distributed experiments.
///
/// This trait extends [`SaverConfig`] with an MPI-aware initializer that can
/// append rank-specific subfolders or other distributed metadata to paths.
/// It is only available when the `mpi` feature is enabled.
pub trait DistSaverConfig: SaverConfig {
    /// Initialize the configuration for a specific MPI process.
    fn init(self, proc: &MPIProcess) -> Arc<Self>;
}

/// Folder hierarchy for file-based [`Recorder`](crate::Recorder) and [`Checkpointer`](crate::Checkpointer).
///
/// # Notes
///
/// The 4 csv files information are linked by the unique [`Id`](crate::Id) of computed [`Solution`](crate::Solution).
///
/// * `path`
///  * evaluations
///   * obj.csv             (points from the [`FuncWrapper`](crate::FuncWrapper) view)
///   * opt.csv             (points from the [`Optimizer`](crate::Optimizer) view)
///   * info.csv            ([`SolInfo`](crate::SolInfo) and [`OptInfo`](crate::OptInfo))
///   * out.csv             ([`Outcome`](crate::Outcome))
///  * checkpoint
///   * state_opt.mp      ([`OptState`](crate::OptState))
///   * state_stp.mp      ([`Stop`](crate::Stop))
///   * state_eval.mp     ([`Evaluate`](crate::experiment::Evaluate))
///   * state_param.mp    ([`GlobalParameters`](crate::GlobalParameters))
pub struct FolderConfig {
    pub path: PathBuf,
    pub path_rec: PathBuf,
    pub path_check: PathBuf,
    pub path_work: PathBuf,
    /// Whether this configuration has already been initialized for distributed runs.
    pub is_dist: bool,
}

impl FolderConfig {
    /// Create a new file-based configuration rooted at `path`.
    ///
    /// This constructor defines the default folder hierarchy for both recorder
    /// and checkpointer outputs. It only builds path structures.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = FolderConfig::new("./my_run");
    /// let config = config.init();
    /// ```
    pub fn new(path: &str) -> Self {
        let path = PathBuf::from(path);
        let path_rec = path.join(Path::new("recorder"));
        let path_check = path.join(Path::new("checkpointer"));
        let path_work = path_check.join(Path::new("workers"));
        FolderConfig {
            path,
            path_rec,
            path_check,
            path_work,
            is_dist: false,
        }
    }
}

impl SaverConfig for FolderConfig {
    /// Initialize and wrap the configuration in an [`Arc`].
    fn init(self) -> Arc<Self> {
        Arc::new(self)
    }
}
/// No-op configuration used when persistence is disabled.
///
/// This configuration is used with [`NoCheck`](crate::checkpointer::NoCheck).
pub struct NoConfig;
impl SaverConfig for NoConfig {
    /// Initialize the no-op configuration in an [`Arc`].
    fn init(self) -> Arc<Self> {
        Arc::new(self)
    }
}

#[cfg(feature = "mpi")]
impl DistSaverConfig for FolderConfig {
    /// Initialize a distributed configuration for a specific MPI rank.
    ///
    /// This method appends rank-specific subfolders to recorder and checkpointer
    /// paths and marks the configuration as distributed.
    fn init(mut self, proc: &MPIProcess) -> Arc<Self> {
        self.path_rec = self
            .path_rec
            .join(Path::new(&format!("recorder_rank{}", proc.rank)));
        self.path_work = self.path_check.join(Path::new("workers"));
        self.path_check = self
            .path_check
            .join(Path::new(&format!("checkpointer_rank{}", proc.rank)));
        self.is_dist = true;
        Arc::new(self)
    }
}

#[cfg(feature = "mpi")]
impl DistSaverConfig for NoConfig {
    /// Initialize the no-op configuration for MPI runs.
    fn init(self, _proc: &MPIProcess) -> Arc<Self> {
        Arc::new(self)
    }
}
