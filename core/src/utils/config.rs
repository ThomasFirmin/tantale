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
//! |   |-- cod.csv
//! |   |-- info.csv
//! |   |-- out.csv
//! |-- checkpointer/
//!     |-- state_optim.mp
//!     |-- state_acc.mp
//!     |-- state_stop.mp
//!     |-- state_eval.mp
//!     |-- state_config.mp
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
///   * cod.csv             ([`Codomain`](crate::Codomain) elements)
///   * info.csv            ([`SolInfo`](crate::SolInfo) and [`OptInfo`](crate::OptInfo))
///   * out.csv             ([`Outcome`](crate::Outcome))
///  * checkpoint
///   * state_optim.mp      ([`OptState`](crate::OptState))
///   * state_acc.mp      ([`Accumulator`](crate::Accumulator))
///   * state_stop.mp      ([`Stop`](crate::Stop))
///   * state_eval.mp     ([`Evaluate`](crate::experiment::Evaluate))
///   * state_config.mp    ([`GlobalParameters`](crate::GlobalParameters))
#[derive(Debug, Clone)]
pub struct FolderConfig {
    pub path: PathBuf,
    pub path_rec: PathBuf,
    pub path_check: PathBuf,
    pub path_work: PathBuf,
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
        let path = std::path::absolute(PathBuf::from(path)).unwrap();
        let path_rec = path.join(Path::new("recorder"));
        let path_check = path.join(Path::new("checkpointer"));
        let path_work = path_check.join(Path::new("workers"));
        FolderConfig {
            path,
            path_rec,
            path_check,
            path_work,
        }
    }
    /// Set the path for the recorder relatively to the root path. This allows for custom recorder folder names.
    pub fn set_path_rec(&mut self, path: &str) {
        self.path_rec = self.path.join(PathBuf::from(path));
    }
    /// Set the path for the checkpointer relatively to the root path. This allows for custom checkpointer folder names.
    pub fn set_path_check(&mut self, path: &str) {
        self.path_check = self.path.join(PathBuf::from(path));
    }
    /// Set the path for the worker folders relatively to the root path. This allows for custom worker folder names.
    pub fn set_path_work(&mut self, path: &str) {
        self.path_work = self.path.join(PathBuf::from(path));
    }
}