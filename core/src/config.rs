use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "mpi")]
use mpi::Rank;

#[cfg(feature = "mpi")]
use crate::experiment::mpi::tools::MPIProcess;

pub trait SaverConfig {
    fn init(&mut self);
    fn after_load(&mut self);
}

#[cfg(feature = "mpi")]
pub trait DistSaverConfig: SaverConfig {
    fn init(&mut self, proc: &MPIProcess);
    fn after_load(&mut self, proc: &MPIProcess);
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
    pub fn new(path: &str) -> Arc<Self> {
        let path = PathBuf::from(path);
        let path_rec = path.join(Path::new("recorder"));
        let path_check = path.join(Path::new("checkpointer"));
        let path_work = path_check.join(Path::new("workers"));
        Arc::new(FolderConfig {
            path,
            path_rec,
            path_check,
            path_work,
            is_dist: false,
        })
    }

    #[cfg(feature = "mpi")]
    pub fn to_dist(&mut self, rank: Rank) {
        self.path_rec = self.path.join(Path::new(&format!("recorder_rank{}", rank)));
        self.path_check = self
            .path
            .join(Path::new(&format!("checkpointer_rank{}", rank)));
        self.path_work = self.path_check.join(Path::new("workers"));
        self.is_dist = true;
    }
}

impl SaverConfig for FolderConfig {
    fn init(&mut self) {}
    fn after_load(&mut self) {}
}
pub struct NoConfig;
impl SaverConfig for NoConfig {
    fn init(&mut self) {}
    fn after_load(&mut self) {}
}

#[cfg(feature = "mpi")]
impl DistSaverConfig for FolderConfig {
    fn init(&mut self, _proc: &MPIProcess) {}
    fn after_load(&mut self, _proc: &MPIProcess) {}
}

#[cfg(feature = "mpi")]
impl DistSaverConfig for NoConfig {
    fn init(&mut self, _proc: &MPIProcess) {}
    fn after_load(&mut self, _proc: &MPIProcess) {}
}
