use crate::{
    checkpointer::{CheckpointError, Checkpointer},
    optimizer::OptState,
    stop::Stop,
    experiment::Evaluate,
    FolderConfig, GlobalParameters, OPT_ID, RUN_ID, SOL_ID,
};

use core::panic;
use rmp_serde;
use std::{
    fs::{create_dir_all, File},
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::mpi::{worker::WorkerState, tools::MPIProcess},
};
#[cfg(feature = "mpi")]
use mpi::{traits::CommunicatorCollectives, Rank};

/// # Attribute
/// * `config` : [`FolderConfig`] - A [`Config`].
/// * `checkpoint` : [`usize`] - If `>0`, a checkpoint will be created every `checkpoint` call to [`step`](Optimizer::step).
///
/// # Notes on File hierarchy
///
/// * `path` from [`Config`]
///  * checkpoint
///   * `state_optim.mp`  --  ([`OptState`])
///   * `state_stop.mp`  --  ([`Stop`])
///   * `state_eval.mp`  --  ([`Evaluate`])
///   * `state_config.mp`  --  Various global parameters as the [`Id`] or experiment identifier.)
pub struct MessagePack {
    pub config: Arc<FolderConfig>,
    pub checkpoint: usize,
    path_stop: PathBuf,
    path_eval: PathBuf,
    path_optim: PathBuf,
    path_config: PathBuf,
}

impl MessagePack {
    pub fn new(config: Arc<FolderConfig>, checkpoint: usize) -> Option<Self> {
        if checkpoint > 0 {
            let path_stop = config.path_check.join(Path::new("state_optim.mp"));
            let path_eval = config.path_check.join(Path::new("state_stop.mp"));
            let path_optim = config.path_check.join(Path::new("state_eval.mp"));
            let path_config = config.path_check.join(Path::new("state_config.mp"));
            Some(MessagePack {
                config,
                checkpoint,
                path_stop,
                path_eval,
                path_optim,
                path_config,
            })
        } else {
            panic!("The `checkpoint` parameter should be >0, otherwise don't use any Checkpointer by passing None.");
        }
    }
}

impl Checkpointer for MessagePack {
    type Config = FolderConfig;

    fn init(&mut self) {
        let does_exist = self.config.path_check.try_exists().unwrap();
        if does_exist {
            panic!(
                "The checkpointer folder path already exists, {}.",
                self.config.path_check.display()
            )
        } else if self.config.path_check.is_file() {
            panic!(
                "The checkpointer path cant point to a file, {}.",
                self.config.path_check.display()
            )
        } else {
            create_dir_all(&self.config.path_check).unwrap();
        }
    }

    fn after_load(&mut self) {
        // Check if all folder and files exist
        if self.config.path_check.try_exists().unwrap() {
            if !self.path_config.try_exists().unwrap() {
                panic!(
                    "The `config` file does not exist in {}",
                    self.path_config.display()
                )
            }
            if !self.path_optim.try_exists().unwrap() {
                panic!(
                    "The `optimizer` file does not exist in {}",
                    self.path_optim.display()
                )
            }
            if !self.path_stop.try_exists().unwrap() {
                panic!(
                    "The `stop` file does not exist in {}",
                    self.path_stop.display()
                )
            }
            if !self.path_eval.try_exists().unwrap() {
                panic!(
                    "The `eval` file does not exist in {}",
                    self.path_eval.display()
                )
            }
        } else {
            panic!(
                "The checkpointer folder does not exists, {}.",
                self.config.path_check.display()
            )
        }
    }

    fn save_state<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
    ) {
        let mut wrt = File::create(&self.path_optim).unwrap();
        rmp_serde::encode::write(&mut wrt, state).unwrap();
        let mut wrt = File::create(&self.path_stop).unwrap();
        rmp_serde::encode::write(&mut wrt, stop).unwrap();
        let mut wrt = File::create(&self.path_eval).unwrap();
        rmp_serde::encode::write(&mut wrt, eval).unwrap();

        let global = GlobalParameters {
            sold_id: SOL_ID.load(Ordering::Relaxed),
            opt_id: OPT_ID.load(Ordering::Relaxed),
            run_id: RUN_ID.load(Ordering::Relaxed),
        };

        let mut wrt = File::create(&self.path_config).unwrap();
        rmp_serde::encode::write(&mut wrt, &global).unwrap();
    }

    fn load<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
    ) -> Result<(OState, St, Eval), CheckpointError> {
        let global: GlobalParameters = self.load_parameters()?;
        SOL_ID.store(global.sold_id, Ordering::Release);
        OPT_ID.store(global.sold_id, Ordering::Release);
        Ok((
            self.load_optimizer()?,
            self.load_stop()?,
            self.load_evaluate()?,
        ))
    }

    fn load_stop<St: Stop>(&self) -> Result<St, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap() {
            if self.path_stop.is_file() {
                let rdr = File::open(&self.path_stop).unwrap();
                Ok(rmp_serde::decode::from_read(rdr).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "The `Stop` file state_stop.mp does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_optimizer<OState: OptState>(&self) -> Result<OState, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap() {
            if self.path_optim.is_file() {
                let rdr = File::open(&self.path_optim).unwrap();
                Ok(rmp_serde::decode::from_read(rdr).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "The `Optimizer` file state_optim.mp does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_evaluate<Eval: Evaluate>(&self) -> Result<Eval, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap() {
            if self.path_eval.is_file() {
                let rdr = File::open(&self.path_eval).unwrap();
                Ok(rmp_serde::decode::from_read(rdr).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "The `Evaluate` file state_eval.mp does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_parameters(&self) -> Result<GlobalParameters, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap() {
            if self.path_config.is_file() {
                let rdr = File::open(&self.path_config).unwrap();
                Ok(rmp_serde::decode::from_read(rdr).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "The `Config` file state_config.mp does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }
}

#[cfg(feature = "mpi")]
impl DistCheckpointer for MessagePack {
    type WCheck<WState: WorkerState> = WCheckMessagePack;
    fn init_dist(&mut self, proc: &MPIProcess) {
        if self.config.is_dist {
            let does_exist = self.config.path_check.try_exists().unwrap();
            if does_exist {
                panic!(
                    "The checkpointer folder path already exists, {}.",
                    self.config.path_check.display()
                )
            } else if self.config.path_check.is_file() {
                panic!(
                    "The checkpointer path cant point to a file, {}.",
                    self.config.path_check.display()
                )
            } else {
                create_dir_all(&self.config.path_check).unwrap();
                create_dir_all(&self.config.path_work).unwrap();
            }
        } else {
            panic!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.")
        }
        // Wait for worker
        proc.world.barrier();
    }

    fn after_load_dist(&mut self, proc: &MPIProcess) {
        if self.config.is_dist {
            // Check if all folder and files exist
            if self.config.path_check.try_exists().unwrap() {
                if !self.path_config.try_exists().unwrap() {
                    panic!(
                        "The `config` file does not exist in {}",
                        self.path_config.display()
                    )
                }
                if !self.path_optim.try_exists().unwrap() {
                    panic!(
                        "The `optimizer` file does not exist in {}",
                        self.path_optim.display()
                    )
                }
                if !self.path_stop.try_exists().unwrap() {
                    panic!(
                        "The `stop` file does not exist in {}",
                        self.path_stop.display()
                    )
                }
                if !self.path_eval.try_exists().unwrap() {
                    panic!(
                        "The `eval` file does not exist in {}",
                        self.path_eval.display()
                    )
                }
            } else {
                panic!(
                    "The checkpointer folder does not exists, {}.",
                    self.config.path_check.display()
                )
            }
        } else {
            panic!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.")
        }
        proc.world.barrier();
    }

    fn save_state_dist<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
        _rank: Rank,
    ) {
        let mut wrt = File::create(&self.path_optim).unwrap();
        rmp_serde::encode::write(&mut wrt, state).unwrap();
        let mut wrt = File::create(&self.path_stop).unwrap();
        rmp_serde::encode::write(&mut wrt, stop).unwrap();
        let mut wrt = File::create(&self.path_eval).unwrap();
        rmp_serde::encode::write(&mut wrt, eval).unwrap();

        let global = GlobalParameters {
            sold_id: SOL_ID.load(Ordering::Relaxed),
            opt_id: OPT_ID.load(Ordering::Relaxed),
            run_id: RUN_ID.load(Ordering::Relaxed),
        };

        let mut wrt = File::create(&self.path_config).unwrap();
        rmp_serde::encode::write(&mut wrt, &global).unwrap();
    }

    fn load_dist<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        rank: Rank,
    ) -> Result<(OState, St, Eval), CheckpointError> {
        let global = self.load_parameters_dist(rank)?;
        SOL_ID.store(global.sold_id, Ordering::Release);
        OPT_ID.store(global.opt_id, Ordering::Release);
        Ok((
            self.load_optimizer_dist(rank)?,
            self.load_stop_dist(rank)?,
            self.load_evaluate_dist(rank)?,
        ))
    }

    fn load_stop_dist<St: Stop>(&self, _rank: Rank) -> Result<St, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap() {
            if self.path_stop.is_file() {
                let rdr = File::open(&self.path_stop).unwrap();
                Ok(rmp_serde::decode::from_read(rdr).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "The `Stop` file state_stop.mp does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_optimizer_dist<OState: OptState>(
        &self,
        _rank: Rank,
    ) -> Result<OState, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap() {
            if self.path_optim.is_file() {
                let rdr = File::open(&self.path_optim).unwrap();
                Ok(rmp_serde::decode::from_read(rdr).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "The `Optimizer` file state_optim.mp does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_evaluate_dist<Eval: Evaluate>(&self, _rank: Rank) -> Result<Eval, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap() {
            if self.path_eval.is_file() {
                let rdr = File::open(&self.path_eval).unwrap();
                Ok(rmp_serde::decode::from_read(rdr).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "The `Evaluate` file state_eval.mp does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_parameters_dist(&self, _rank: Rank) -> Result<GlobalParameters, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap() {
            if self.path_config.is_file() {
                let rdr = File::open(&self.path_config).unwrap();
                Ok(rmp_serde::decode::from_read(rdr).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "The `Config` file state_config.mp does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn get_check_worker<WState: WorkerState>(&self) -> Self::WCheck<WState> {
        todo!()
    }
}

#[cfg(feature = "mpi")]
/// # Attribute
/// * `config` : [`FolderConfig`] - A [`Config`].
/// * `ignore_missing` : During [`load`](WorkerCheckpointer::load), ignore missing checkpoint file and create new ones.
///   It is used when the number of [`Worker`] is greater than the previous run of the optimization.
/// * `checkpoint` : usize - If `>0`, a checkpoint will be created every `checkpoint` call to [`step`](Optimizer::step).
///
/// # Notes on File hierarchy
///
/// * `path` from [`Config`]
///  * checkpoint
///   * workers
///     * `worker_state_rank{}` ([`WorkerState`])
pub struct WCheckMessagePack {
    pub config: Arc<FolderConfig>,
    path_worker: PathBuf,
}

#[cfg(feature = "mpi")]
impl WCheckMessagePack {
    pub fn new(proc: &MPIProcess, config: Arc<FolderConfig>) -> Self {
        let path_worker = config.path_work.join(format!("worker_rank{}", proc.rank));
        WCheckMessagePack {
            config,
            path_worker,
        }
    }
}

#[cfg(feature = "mpi")]
impl<WState: WorkerState> WorkerCheckpointer<WState> for WCheckMessagePack {
    fn init(&mut self, proc: &MPIProcess) {
        proc.world.barrier();
        let path = &self.config.path_work;
        if self.config.is_dist {
            let does_exist = path.try_exists().unwrap();
            if does_exist {
                panic!("The worker folder path already exists, {}.", path.display())
            } else if path.is_file() {
                panic!("The worker path cant point to a file, {}.", path.display())
            } else {
                create_dir_all(&self.path_worker).unwrap();
            }
        } else {
            panic!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.")
        }
    }

    fn after_load(&mut self, proc: &MPIProcess) {
        proc.world.barrier();
        if self.config.is_dist {
            // Check if all folder and files exist
            if !self.path_worker.try_exists().unwrap() {
                panic!(
                    "The Worker folder {} in checkpoint does not exists.",
                    self.path_worker.display()
                )
            }
        } else {
            panic!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.")
        }
    }

    fn save_state(&self, state: &WState, _rank: Rank) {
        let mut wrt = File::create(&self.path_worker).unwrap();
        rmp_serde::encode::write(&mut wrt, state).unwrap();
    }

    fn load(&self, _rank: Rank) -> Result<WState, CheckpointError> {
        // Check if file exist
        if self.path_worker.try_exists().unwrap() {
            let rdr = File::open(&self.path_worker).unwrap();
            Ok(rmp_serde::decode::from_read(rdr).unwrap())
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }
}
