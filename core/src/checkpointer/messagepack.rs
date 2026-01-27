use crate::{
    FolderConfig, GlobalParameters, OPT_ID, RUN_ID, SOL_ID,
    checkpointer::{CheckpointError, Checkpointer, ThrCheckpointer},
    experiment::{Evaluate},
    optimizer::OptState,
    stop::Stop,
};

use core::panic;
use rmp_serde;
use std::{
    fs::{File, create_dir_all},
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
};

#[cfg(feature = "mpi")]
use crate::{
    checkpointer::{DistCheckpointer, WorkerCheckpointer},
    experiment::mpi::{utils::MPIProcess, worker::WorkerState},
};
#[cfg(feature = "mpi")]
use mpi::{Rank, traits::CommunicatorCollectives};

/// A [`Checkpointer`] based on [`rmp_serde`]. See the [MessagePack page](https://msgpack.org/)
/// 
/// # Attribute
/// * `config` : [`FolderConfig`] - Describes the folder hierarchy.
///
/// # Notes on File hierarchy
///
/// * `path` from [`Config`]
///  * checkpoint
///   * `state_optim.mp`  --  ([`OptState`])
///   * `state_stop.mp`  --  ([`Stop`])
///   * `state_eval.mp`  --  ([`Evaluate`], might vary according to [`Runable`](crate::Runable) type)
///   * `state_config.mp`  --  Various global parameters, see [`GlobalParameters`])
pub struct MessagePack {
    pub config: Arc<FolderConfig>,
    path_stop: PathBuf,
    path_eval: PathBuf,
    path_optim: PathBuf,
    path_config: PathBuf,
}

impl MessagePack {
    /// Returns a [`MessagePack`] wrapped within an [`Option`] to match optional [`Checkpointer`] in [`Runable`](crate::Runable).
    pub fn new(config: Arc<FolderConfig>) -> Option<Self> {
        let path_stop = config.path_check.join(Path::new("state_stop.mp"));
        let path_eval = config.path_check.join(Path::new("state_eval.mp"));
        let path_optim = config.path_check.join(Path::new("state_optim.mp"));
        let path_config = config.path_check.join(Path::new("state_config.mp"));
        Some(MessagePack {
            config,
            path_stop,
            path_eval,
            path_optim,
            path_config,
        })
    }
}

impl Checkpointer for MessagePack {
    type Config = FolderConfig;

    /// Checks if folders and files already exists, if so [`panic!`]. Otherwise, the function
    /// creates the folder hierarchy based on [`FolderConfig`].
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
    /// Ran after [`init`](Checkpointer::init), and after a [`load!`](crate::load).
    /// Checks if the folder and file hierarchy exists, based on [`FolderConfig`]. If not, then [`panic!`].
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
        } else {
            panic!(
                "The checkpointer folder does not exists, {}.",
                self.config.path_check.display()
            )
        }
    }

    /// Saves a checkpoint from all different states, [`OptState`], [`Stop`], [`Evaluate`], and [`GlobalParameters`],
    /// according to a [`FolderConfig`].
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

    /// Loads a checkpoint from already saved states, [`OptState`], [`Stop`], [`Evaluate`], and [`GlobalParameters`],
    /// according to a [`FolderConfig`].
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

    /// Loads a checkpoint from an already saved  [`Stop`], according to a [`FolderConfig`].
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

    /// Loads a checkpoint from an already saved  [`OptState`], according to a [`FolderConfig`].
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
    /// Loads a checkpoint from an already saved [`Evaluate`], according to a [`FolderConfig`].
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

    /// Loads a checkpoint from an already saved  [`GlobalParameters`], according to a [`FolderConfig`].
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

impl ThrCheckpointer for MessagePack {
    type Config = FolderConfig;

    /// Checks if folders and files already exists, if so [`panic!`]. Otherwise, the function
    /// creates the folder hierarchy based on [`FolderConfig`].
    fn init_thr(&mut self) {
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

    /// Ran after [`init`](Checkpointer::init), and after a [`load!`](crate::load).
    /// Checks if the folder and file hierarchy exists, based on [`FolderConfig`]. If not, then [`panic!`].
    /// 
    /// # Note 
    /// 
    /// Here the [`Evaluate`] checkpoints is a [`Vec`] of [`Evaluate`] states, corresponding to each state of threads
    /// involved in previous computions.
    /// The number of [`Evaluate`] checkpoints might vary according to how many
    /// threads were actually active during previous [`run`](crate::Runable::run). Then, some [`Evaluate`] checkpoints for some
    /// threads might not have been previously created. If so, they are replaced by an empty [`Evaluate`].
    fn after_load_thr(&mut self) {
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

    /// Saves a checkpoint from all different states, [`OptState`], [`Stop`], [`Evaluate`], and [`GlobalParameters`],
    /// according to a [`FolderConfig`].
    /// 
    /// # Note 
    /// 
    /// Here the [`Evaluate`] checkpoints is a [`Vec`] of [`Evaluate`] states, corresponding to each state of threads
    /// involved in previous computions.
    /// The number of [`Evaluate`] checkpoints might vary according to how many
    /// threads were actually active during previous [`run`](crate::Runable::run). Then, some [`Evaluate`] checkpoints for some
    /// threads might not have been previously created. If so, they are replaced by an empty [`Evaluate`].
    fn save_state_thr<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
        state: &OState,
        stop: &St,
        eval: &Eval,
        k: usize,
    ) {
        let mut wrt = File::create(&self.path_optim).unwrap();
        rmp_serde::encode::write(&mut wrt, state).unwrap();
        let mut wrt = File::create(&self.path_stop).unwrap();
        rmp_serde::encode::write(&mut wrt, stop).unwrap();
        let path_eval = self
            .config
            .path_check
            .join(Path::new(&format!("state_eval_thr_{}.mp", k)));
        let mut wrt = File::create(&path_eval).unwrap();
        rmp_serde::encode::write(&mut wrt, eval).unwrap();

        let global = GlobalParameters {
            sold_id: SOL_ID.load(Ordering::Relaxed),
            opt_id: OPT_ID.load(Ordering::Relaxed),
            run_id: RUN_ID.load(Ordering::Relaxed),
        };

        let mut wrt = File::create(&self.path_config).unwrap();
        rmp_serde::encode::write(&mut wrt, &global).unwrap();
    }

    /// Loads a checkpoint from already saved states, [`OptState`], [`Stop`], [`Evaluate`], and [`GlobalParameters`],
    /// according to a [`FolderConfig`].
    fn load_thr<OState: OptState, St: Stop, Eval: Evaluate>(
        &self,
    ) -> Result<(OState, St, Vec<Eval>), CheckpointError> {
        let global: GlobalParameters = self.load_parameters()?;
        SOL_ID.store(global.sold_id, Ordering::Release);
        OPT_ID.store(global.sold_id, Ordering::Release);
        Ok((
            self.load_optimizer_thr()?,
            self.load_stop_thr()?,
            self.load_evaluate_thr()?,
        ))
    }

    /// Loads a checkpoint from an already saved  [`Stop`], according to a [`FolderConfig`].
    fn load_stop_thr<St: Stop>(&self) -> Result<St, CheckpointError> {
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

    /// Loads a checkpoint from an already saved  [`OptState`], according to a [`FolderConfig`].
    fn load_optimizer_thr<OState: OptState>(&self) -> Result<OState, CheckpointError> {
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

    /// Loads a checkpoint from an already saved [`Evaluate`], according to a [`FolderConfig`].
    /// 
    /// # Note 
    /// 
    /// Here the [`Evaluate`] checkpoints is a [`Vec`] of [`Evaluate`] states, corresponding to each state of threads
    /// involved in previous computions.
    /// The number of [`Evaluate`] checkpoints might vary according to how many
    /// threads were actually active during previous [`run`](crate::Runable::run). Then, some [`Evaluate`] checkpoints for some
    /// threads might not have been previously created. If so, they are replaced by an empty [`Evaluate`].
    /// 
    /// # Panic
    /// If no [`Evaluate`] checkpoint was found, then [`panic!`].
    fn load_evaluate_thr<Eval: Evaluate>(&self) -> Result<Vec<Eval>, CheckpointError> {
        // Check if file exist
        let k = num_cpus::get();
        let mut vec_eval = Vec::new();
        if self.config.path_check.try_exists().unwrap() {
            for i in 0..k {
                let path_eval = self
                    .config
                    .path_check
                    .join(Path::new(&format!("state_eval_thr_{}.mp", i)));
                if path_eval.is_file() {
                    let rdr = File::open(&path_eval).unwrap();
                    vec_eval.push(rmp_serde::decode::from_read(rdr).unwrap());
                } else {
                    println!(
                        "INFO: The Evaluate checkpoint file for thread {} does not exists. It is replaced by a new empty Evaluate.",
                        i
                    );
                }
            }
        } else {
            return Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )));
        }
        if !vec_eval.is_empty() {
            Ok(vec_eval)
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any Evaluate checkpoint file",
            )))
        }
    }

    /// Loads a checkpoint from an already saved  [`GlobalParameters`], according to a [`FolderConfig`].
    fn load_parameters_thr(&self) -> Result<GlobalParameters, CheckpointError> {
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

    /// Checks if folders and files already exists, if so [`panic!`]. Otherwise, the function
    /// creates the folder hierarchy based on [`FolderConfig`].
    /// 
    /// # Note
    /// 
    /// Here, we suppose a [`MasterWorker`](crate::MasterWorker) distribution. The Master creates
    /// and cheks the folder hierarchy. Then, once done, waits for all workers to reach the same point,
    /// with a [`barrier`](mpi::collective::CommunicatorCollectives::barrier). 
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
            panic!(
                "The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method on the config object."
            )
        }
        // Wait for worker
        proc.world.barrier();
    }

    /// Ran after [`init`](Checkpointer::init), and after a [`load!`](crate::load).
    /// Checks if the folder and file hierarchy exists, based on [`FolderConfig`]. If not, then [`panic!`].
    /// 
    /// /// # Note
    /// 
    /// Here, we suppose a [`MasterWorker`](crate::MasterWorker) distribution.
    /// The Master checks the folder hierarchy, if some files or folder are missing, then [`panic!`].
    /// Then, once done, it waits for all workers to reach the same point,
    /// with a [`barrier`](mpi::collective::CommunicatorCollectives::barrier).
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
            panic!(
                "The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method."
            )
        }
        proc.world.barrier();
    }

    /// A no-operation function to synchronize all processes after initialization, if no [`WorkerCheckpointer`] is used.
    fn no_check_init(proc: &MPIProcess) {
        proc.world.barrier();
    }

    /// Saves a checkpoint from all different states, [`OptState`], [`Stop`], [`Evaluate`], and [`GlobalParameters`],
    /// according to a [`FolderConfig`].
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

    /// Loads a checkpoint from already saved states, [`OptState`], [`Stop`], [`Evaluate`], and [`GlobalParameters`],
    /// according to a [`FolderConfig`].
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

    /// Loads a checkpoint from an already saved  [`Stop`], according to a [`FolderConfig`].
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

    /// Loads a checkpoint from an already saved  [`OptState`], according to a [`FolderConfig`].
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

    /// Loads a checkpoint from an already saved [`Evaluate`], according to a [`FolderConfig`].
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

    /// Loads a checkpoint from an already saved  [`GlobalParameters`], according to a [`FolderConfig`].
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

    /// Returns a [`WorkerCheckpointer`] based on the given [`MPIProcess`].
    fn get_check_worker<WState: WorkerState>(&self, proc: &MPIProcess) -> Self::WCheck<WState> {
        WCheckMessagePack::new(self.config.clone(), proc)
    }
}

#[cfg(feature = "mpi")]
/// A [`WorkerCheckpointer`] based on [`rmp_serde`]. See the [MessagePack page](https://msgpack.org/)
/// # Attribute
/// * `config` : [`FolderConfig`] - A [`Config`].
///   It is used when the number of [`Worker`](crate::Worker) is greater than the previous run of the optimization.
///
/// # Notes
/// The worker checkpoint file hierarchy is as follows, it varies from usual mono-thread [`Checkpointer`]:
/// * `path` from [`FolderConfig`]
///  * checkpoint
///   * workers <===== Modification here
///     * `worker_state_rank{}` ([`WorkerState`]) <====== And here
pub struct WCheckMessagePack(PathBuf);

#[cfg(feature = "mpi")]
impl WCheckMessagePack {
    /// Returns a [`WCheckMessagePack`] for the given [`MPIProcess`].
    pub fn new(config: Arc<FolderConfig>, proc: &MPIProcess) -> Self {
        if !config.is_dist {
            panic!(
                "The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method."
            )
        }
        WCheckMessagePack(
            config
                .path_work
                .join(format!("worker_rank{}.mp", proc.rank)),
        )
    }
}

#[cfg(feature = "mpi")]
impl<WState: WorkerState> WorkerCheckpointer<WState> for WCheckMessagePack {
    /// Checks if folders and files already exists, if so [`panic!`] for [WorkerCheckpointer].
    /// 
    /// # Note
    /// 
    /// Here, we suppose a [`MasterWorker`](crate::MasterWorker) distribution.
    /// The [`Worker`](crate::Worker) wait for the Master to create and check the folder hierarchy,
    /// with a [`barrier`](mpi::collective::CommunicatorCollectives::barrier).
    /// Then the [`Worker`](crate::Worker) checks if its own checkpoint file exists.
    fn init(&mut self, proc: &MPIProcess) {
        proc.world.barrier();
        if self.0.try_exists().unwrap() {
            panic!(
                "The Worker folder {} in checkpoint already exists.",
                self.0.display()
            )
        }
    }
    /// Ran after [`init`](WorkerCheckpointer::init), and after a [`load!`](crate::load).
    /// Checks if the folder and file hierarchy exists, based on [`FolderConfig`]. If not, then [`panic!`].
    /// 
    /// # Note
    /// 
    /// Here, we suppose a [`MasterWorker`](crate::MasterWorker) distribution.
    /// The [`Worker`](crate::Worker) wait for the Master to check the folder hierarchy,
    /// with a [`barrier`](mpi::collective::CommunicatorCollectives::barrier).
    /// Then the [`Worker`](crate::Worker) checks if its own checkpoint file exists.
    fn after_load(&mut self, proc: &MPIProcess) {
        proc.world.barrier();
        // Check if all folder and files exist
        if !self.0.try_exists().unwrap() {
            panic!(
                "The Worker folder {} in checkpoint does not exists.",
                self.0.display()
            )
        }
    }

    /// Saves a checkpoint from the given [`WorkerState`], according to a [`FolderConfig`].
    fn save_state(&self, state: &WState, _rank: Rank) {
        let mut wrt = File::create(&self.0).unwrap();
        rmp_serde::encode::write(&mut wrt, state).unwrap();
    }

    /// Loads a checkpoint from an already saved [`WorkerState`], according to a [`FolderConfig`].
    fn load(&self, _rank: Rank) -> Result<WState, CheckpointError> {
        // Check if file exist
        if self.0.try_exists().unwrap() {
            let rdr = File::open(&self.0).unwrap();
            Ok(rmp_serde::decode::from_read(rdr).unwrap())
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }
}
