
use crate::{
    FolderConfig, GlobalParameters, OPT_ID, Onto, OptInfo, Partial, RUN_ID, SOL_ID, SaverConfig, SolInfo,
    checkpointer::{CheckpointError, Checkpointer, DistCheckpointer},
    domain::{Domain, onto::OntoDom},
    experiment::Evaluate,
    objective::{Codomain, Outcome},
    optimizer::Optimizer,
    recorder::Recorder,
    searchspace::Searchspace,
    solution::{BatchType, Id, Solution},
    stop::Stop,
};

use bincode::config::Config;
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use rmp_serde;
use core::panic;
use std::{
    fs::{File, create_dir_all}, path::{Path, PathBuf}, sync::{Arc, atomic::Ordering}
};

#[cfg(feature = "mpi")]
use crate::recorder::DistRecorder;
#[cfg(feature = "mpi")]
use mpi::Rank;


/// # Attribute
/// * `checkpoint` : usize - If `>0`, a checkpoint will be created every `checkpoint` call to [`step`](Optimizer::step).
/// 
/// # Notes on File hierarchy
/// 
/// * `path` from [`Config`]
///  * checkpoint
///   * `state_optim.mp`  --  ([`OptState`])
///   * `state_stop.mp`  --  ([`Stop`])
///   * `state_eval.mp`  --  ([`Evaluate`])
///   * `state_config.mp`  --  Various global parameters as the [`Id`] or experiment identifier.)
pub struct MessagePack<SolId,Obj,Opt,Out,Scp,Op>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out:Outcome,
    Op: Optimizer<SolId,Obj,Opt,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Obj,Opt,Op::SInfo>,
{
    pub config: FolderConfig<Op::Sol,Scp,SolId,Obj,Opt,Op::SInfo,Op::Cod,Out>,
    pub checkpoint: usize,
    path_stop: PathBuf,
    path_eval: PathBuf,
    path_optim: PathBuf,
    path_config: PathBuf,
}

impl<SolId,Obj,Opt,Out,Scp,Op> MessagePack<SolId,Obj,Opt,Out,Scp,Op>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out:Outcome,
    Op: Optimizer<SolId,Obj,Opt,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Obj,Opt,Op::SInfo>,
{
    pub fn new(
        config: Arc<FolderConfig<Op::Sol,Scp,SolId,Obj,Opt,Op::SInfo,Op::Cod,Out>>,
        checkpoint:usize,
    ) -> Self {
        if checkpoint > 0 {
            let path_stop = config.path_check.join(Path::new("state_optim.mp"));
            let path_eval = config.path_check.join(Path::new("state_stop.mp"));
            let path_optim = config.path_check.join(Path::new("state_eval.mp"));
            let path_config = config.path_check.join(Path::new("state_config.mp"));
            MessagePack{
                config,
                checkpoint,
                path_stop,
                path_eval,
                path_optim,
                path_config,
            }
        } else {
            panic!("The `checkpoint` parameter should be >0, otherwise don't use any Checkpointer by passing None.");
        };
    }
}


impl<SolId,St,Obj,Opt,Out,Scp,Op,Eval> Checkpointer<SolId,St,Obj,Opt,Out,Scp,Op,Eval> for MessagePack<SolId,Obj,Opt,Out,Scp,Op>
where
    SolId: Id,
    St: Stop,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out:Outcome,
    Op: Optimizer<SolId,Obj,Opt,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Obj,Opt,Op::SInfo>,
    Eval: Evaluate,
{
    type Config = FolderConfig<Op::Sol,Scp,SolId,Obj,Opt,Op::SInfo,Op::Cod,Out>;

    fn init(&mut self) {
        let does_exist = self.config.path_check.try_exists().unwrap();
        if does_exist {
            panic!("The checkpointer folder path already exists, {}.", self.config.path_check.display())
        } else if self.path.is_file() {
            panic!("The checkpointer path cant point to a file, {}.",self.config.path_check.display())
        } else {
                let path = self.path.join(Path::new("checkpoint"));
                create_dir_all(path.as_path()).unwrap();
        }
    }

    fn after_load(&mut self) {
        // Check if all folder and files exist
        if self.config.path_check.try_exists().unwrap(){
            if !self.path_config.try_exists().unwrap(){panic!("The `config` file does not exist in {}", self.path_config.display())}
            if !self.path_optim.try_exists().unwrap(){panic!("The `optimizer` file does not exist in {}", self.path_optim.display())}
            if !self.path_stop.try_exists().unwrap(){panic!("The `stop` file does not exist in {}", self.path_stop.display())}
            if !self.path_eval.try_exists().unwrap(){panic!("The `eval` file does not exist in {}", self.path_eval.display())}
        } else{panic!("The checkpointer folder does not exists, {}.", self.config.path_check.display())}
    }

    fn save_state(&self, state: &Op::State, stop: &St, eval: &Eval) {
        let mut wrt = File::create(self.path_optim).unwrap();
        rmp_serde::encode::write(&mut wrt, state).unwrap();
        let mut wrt = File::create(self.path_stop).unwrap();
        rmp_serde::encode::write(&mut wrt, stop).unwrap();
        let mut wrt = File::create(self.path_eval).unwrap();
        rmp_serde::encode::write(&mut wrt, eval).unwrap();

        let global = GlobalParameters {
            sold_id: SOL_ID.load(Ordering::Relaxed),
            opt_id: OPT_ID.load(Ordering::Relaxed),
            run_id: RUN_ID.load(Ordering::Relaxed),
        };

        let mut wrt = File::create(self.path_config).unwrap();
        rmp_serde::encode::write(&mut wrt, &global).unwrap();
    }

    fn load(&self) -> Result<(St, Op, Eval), CheckpointError> {
        let global: GlobalParameters = self.load_parameters()?;
        SOL_ID.store(global.sold_id, Ordering::Release);
        OPT_ID.store(global.sold_id, Ordering::Release);
        Ok((self.load_stop()?,self.load_optimizer()?,self.load_evaluate()?))
    }

    fn load_stop(&self) -> Result<St, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap(){
            if self.path_stop.is_file() {
                let rdr = File::open(self.path_stop).unwrap();
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

    fn load_optimizer(&self) -> Result<Op, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap(){
            if self.path_optim.is_file() {
                let rdr = File::open(self.path_stop).unwrap();
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

    fn load_evaluate(&self) -> Result<Eval, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap(){
            if self.path_eval.is_file() {
                let rdr = File::open(self.path_eval).unwrap();
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
        if self.config.path_check.try_exists().unwrap(){
            if self.path_config.is_file() {
                let rdr = File::open(self.path_config).unwrap();
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

#[cfg(feature="mpi")]
impl<SolId,St,Obj,Opt,Out,Scp,Op,Eval> DistCheckpointer<SolId,St,Obj,Opt,Out,Scp,Op,Eval> for MessagePack<SolId,Obj,Opt,Out,Scp,Op>
where
    SolId: Id,
    St: Stop,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out:Outcome,
    Op: Optimizer<SolId,Obj,Opt,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Obj,Opt,Op::SInfo>,
    Eval: Evaluate,
{
    fn init(&mut self, _rank: Rank) {
        if self.config.is_dist{
            let does_exist = self.config.path_check.try_exists().unwrap();
            if does_exist {
                panic!("The checkpointer folder path already exists, {}.", self.config.path_check.display())
            } else if self.path.is_file() {
                panic!("The checkpointer path cant point to a file, {}.",self.config.path_check.display())
            } else {
                    let path = self.path.join(Path::new("checkpoint"));
                    create_dir_all(path.as_path()).unwrap();
            }
        } else { panic!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.") }
    }

    fn after_load(&mut self, _rank: Rank) {
        if self.config.is_dist{
            // Check if all folder and files exist
            if self.config.path_check.try_exists().unwrap(){
                if !self.path_config.try_exists().unwrap(){panic!("The `config` file does not exist in {}", self.path_config.display())}
                if !self.path_optim.try_exists().unwrap(){panic!("The `optimizer` file does not exist in {}", self.path_optim.display())}
                if !self.path_stop.try_exists().unwrap(){panic!("The `stop` file does not exist in {}", self.path_stop.display())}
                if !self.path_eval.try_exists().unwrap(){panic!("The `eval` file does not exist in {}", self.path_eval.display())}
            } else{panic!("The checkpointer folder does not exists, {}.", self.config.path_check.display())}
        } else{ panic!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.") }
    }

    fn save_state(&self, state: &Op::State, stop: &St, eval: &Eval, _rank: Rank) {
        let mut wrt = File::create(self.path_optim).unwrap();
        rmp_serde::encode::write(&mut wrt, state).unwrap();
        let mut wrt = File::create(self.path_stop).unwrap();
        rmp_serde::encode::write(&mut wrt, stop).unwrap();
        let mut wrt = File::create(self.path_eval).unwrap();
        rmp_serde::encode::write(&mut wrt, eval).unwrap();

        let global = GlobalParameters {
            sold_id: SOL_ID.load(Ordering::Relaxed),
            opt_id: OPT_ID.load(Ordering::Relaxed),
            run_id: RUN_ID.load(Ordering::Relaxed),
        };

        let mut wrt = File::create(self.path_config).unwrap();
        rmp_serde::encode::write(&mut wrt, &global).unwrap();
    }

    fn load(&self, _rank: Rank) -> Result<(St, Op, Eval), CheckpointError> {
        let global: GlobalParameters = self.load_parameters()?;
        SOL_ID.store(global.sold_id, Ordering::Release);
        OPT_ID.store(global.sold_id, Ordering::Release);
        Ok((self.load_stop()?,self.load_optimizer()?,self.load_evaluate()?))
    }

    fn load_stop(&self, _rank: Rank) -> Result<St, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap(){
            if self.path_stop.is_file() {
                let rdr = File::open(self.path_stop).unwrap();
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

    fn load_optimizer(&self, _rank: Rank) -> Result<Op, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap(){
            if self.path_optim.is_file() {
                let rdr = File::open(self.path_stop).unwrap();
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

    fn load_evaluate(&self, _rank: Rank) -> Result<Eval, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap(){
            if self.path_eval.is_file() {
                let rdr = File::open(self.path_eval).unwrap();
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

    fn load_parameters(&self, _rank: Rank) -> Result<GlobalParameters, CheckpointError> {
        // Check if file exist
        if self.config.path_check.try_exists().unwrap(){
            if self.path_config.is_file() {
                let rdr = File::open(self.path_config).unwrap();
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