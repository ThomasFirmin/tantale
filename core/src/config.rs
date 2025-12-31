use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use crate::experiment::mpi::utils::MPIProcess;

pub trait SaverConfig {
    fn init(self) -> Arc<Self>;
}

#[cfg(feature = "mpi")]
pub trait DistSaverConfig: SaverConfig {
    fn init(self, proc: &MPIProcess) -> Arc<Self>;
}

/// Describes a folders and files hierarchy for file-based [`Recorder`] and [`Checkpointer`].
///
/// # Notes on File hierarchy
///
/// The 4 csv files information are linked by the unique [`Id`] of computed [`Solution`].
///
/// * `path`
///  * evaluations
///   * obj.csv             (points from the [`Objective`] view)
///   * opt.csv             (points from the [`Optimizer`] view)
///   * info.csv            ([`SolInfo`] and [`OptInfo`])
///   * out.csv             ([`Outcome`])
///  * checkpoint
///   * state_opt.mp      ([`OptState`])
///   * state_stp.mp      ([`Stop`])
///   * state_eval.mp     ([`Evaluate`])
///   * state_param.mp    (Various global parameters such as the [`Id`] or experiment identifier.)
pub struct FolderConfig {
    pub path: PathBuf,
    pub path_rec: PathBuf,
    pub path_check: PathBuf,
    pub path_work: PathBuf,
    pub is_dist: bool,
}

impl FolderConfig {
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
    fn init(self) -> Arc<Self> {
        Arc::new(self)
    }
}
pub struct NoConfig;
impl SaverConfig for NoConfig {
    fn init(self) -> Arc<Self> {
        Arc::new(self)
    }
}

#[cfg(feature = "mpi")]
impl DistSaverConfig for FolderConfig {
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
    fn init(self, _proc: &MPIProcess) -> Arc<Self> {
        Arc::new(self)
    }
}
