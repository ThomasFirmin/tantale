//! Filesystem paths for recorder and checkpointer output.
//!
//! [`FolderConfig`] stores a root path and paths for recorder output,
//! checkpoint data, and worker data. [`FolderConfig::new`] computes these
//! paths but does not create directories. Use the setters to choose custom
//! child paths. The crate-level documentation shows the configuration used
//! with an experiment.

use std::path::{Path, PathBuf};

/// Root and child paths used by file-based recorders and checkpointers.
///
/// The default child paths are `recorder`, `checkpointer`, and
/// `checkpointer/workers` under the root. Constructing this value only
/// computes paths; recorder and checkpointer operations create storage as
/// needed.
#[derive(Debug, Clone)]
pub struct FolderConfig {
    pub path: PathBuf,
    pub path_rec: PathBuf,
    pub path_check: PathBuf,
    pub path_work: PathBuf,
}

impl FolderConfig {
    /// Compute the default output paths beneath `path`.
    ///
    /// This does not create any directories.
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
