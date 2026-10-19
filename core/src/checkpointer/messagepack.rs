use crate::{
    FolderConfig, FuncState, Problem, SolId, checkpointer::{Checkpointer, CheckpointerTargets}, errors::CheckpointerError, 
    utils::has_trait::HasState,
};

use chrono::Utc;
use rmp_serde;
use std::{
    fs::{self, File, create_dir_all, rename}, path::{PathBuf},
};

/// A [`Checkpointer`] based on [`rmp_serde`]. See the [MessagePack page](https://msgpack.org/)
///
/// # Notes on File hierarchy
///
/// * `path` from [`FolderConfig`]
///  * checkpoint
///   * `state_optim.mp`  --  ([`OptState`])
///   * `state_stop.mp`  --  ([`Stop`])
///   * `state_eval.mp`  --  ([`Evaluate`], might vary according to [`Runable`](crate::Runable) type)
///   * `state_config.mp`  --  Various global parameters, see [`GlobalParameters`])
pub struct MessagePack {
    /// Describes the folder hierarchy.
    pub config: FolderConfig,
    path_stop: PathBuf,
    path_exp: PathBuf,
    path_optim: PathBuf,
    path_config: PathBuf,
    path_acc: PathBuf,
    path_workers: PathBuf,
    path_func: PathBuf,
}

impl MessagePack {
    /// Returns a [`MessagePack`] wrapped within an [`Option`] to match optional [`Checkpointer`] in [`Runable`](crate::Runable).
    pub fn new(config: FolderConfig) -> Option<Self> {
        let path_stop = config.path_check.join("state_stop.mp");
        let path_exp = config.path_check.join("state_exp.mp");
        let path_optim = config.path_check.join("state_optim.mp");
        let path_config = config.path_check.join("state_config.mp");
        let path_acc = config.path_check.join("state_acc.mp");
        let path_workers = config.path_check.join("state_workers");
        let path_func = config.path_check.join("state_func");
        Some(MessagePack {
            config,
            path_stop,
            path_exp,
            path_optim,
            path_config,
            path_acc,
            path_workers,
            path_func,
        })
    }

    pub fn retrieve<'a>(&'a self, target: &'a CheckpointerTargets) -> &'a PathBuf {
        match target {
            CheckpointerTargets::Checkpoint => &self.config.path_check,
            CheckpointerTargets::Backup(path) => path,
            CheckpointerTargets::Optimizer => &self.path_optim,
            CheckpointerTargets::Accumulator => &self.path_acc,
            CheckpointerTargets::Stop => &self.path_stop,
            CheckpointerTargets::Experiment => &self.path_exp,
            CheckpointerTargets::Config => &self.path_config,
            CheckpointerTargets::WorkerFolder => &self.path_workers,
            CheckpointerTargets::FuncFolder => &self.path_func,
            CheckpointerTargets::Other(path) => path,
        }
    }
}

/// Enum to specify how to modify the checkpoint target (e.g., directory structure, file naming) to retrieve a specific checkpoint file.
pub enum PathModifier {
    Prefix(String),
    Suffix(String),
    PrefixSuffix(String, String),
    None,
}

impl<P: Problem> Checkpointer<P, ()> for MessagePack
{
    fn init(&mut self, _p: &P, _context: &()) -> Result<(), CheckpointerError> {
        let does_exist = self.config.path_check.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", self.config.path_check.display(), e)))?;
        if does_exist {
            return Err(CheckpointerError::FileCreationError(format!("{} already exists.", self.config.path_check.display())))
            
        } else if self.config.path_check.is_file() {
            return Err(CheckpointerError::FolderCreationError(format!("{} points to a file.", self.config.path_check.display())))
        } else {
            create_dir_all(&self.config.path_check).map_err(|e| CheckpointerError::FolderCreationError(format!("Failed to create checkpoint folder {}: {}", self.config.path_check.display(), e)))?;
        }
        Ok(())
    }
    
    fn before_load(&mut self, _p: &P, _context: &()) -> Result<(), CheckpointerError> {
        // Check if all folder and files exist
        if self.config.path_check.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", self.config.path_check.display(), e)))? {
            if !self.path_config.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of config file {}: {}", self.path_config.display(), e)))? {
                return Err(CheckpointerError::FileNotFound(format!("The `config` file does not exist in {}",self.path_config.display())))
            }
            if !self.path_optim.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of optimizer file {}: {}", self.path_optim.display(), e)))? {
                return Err(CheckpointerError::FileNotFound(format!(
                    "The `optimizer` file does not exist in {}",
                    self.path_optim.display()
                )))
            }
            if !self.path_stop.try_exists().unwrap() {
                return Err(CheckpointerError::FileNotFound(format!(
                    "The `stop` file does not exist in {}",
                    self.path_stop.display()
                )))
            }
            if !self.path_acc.try_exists().unwrap() {
                return Err(CheckpointerError::FileNotFound(format!(
                    "The `accumulator` file does not exist in {}",
                    self.path_acc.display()
                )))
            }
        } else {
            return Err(CheckpointerError::FileNotFound(format!(
                "The checkpointer folder does not exists, {}.",
                self.config.path_check.display()
            )))
        }
        Ok(())
    }
    
    fn after_load(&mut self, p: &P, context: &()) -> Result<(), CheckpointerError> {
        // rename old checkpoint folder
        let mut i = 0;
        let mut new_path = self
            .config
            .path_check
            .with_file_name(format!("checkpoint_backup_{}", i));
        while new_path.exists() {
            new_path = self
                .config
                .path_check
                .with_file_name(format!("checkpoint_backup_{}", i));
            i += 1;
        }
        rename(&self.config.path_check, &new_path).map_err(|e| CheckpointerError::Other(format!("Failed to rename checkpoint folder {} to backup {}: {}", self.config.path_check.display(), new_path.display(), e)))?;
        self.init(p, context)?;
        // move func states to new folder
        let func_folder = self.retrieve(&CheckpointerTargets::FuncFolder);
        let old_func_folder = new_path.join("state_func");
        if old_func_folder.exists() && old_func_folder.is_dir() {
            // because a function state might be very large and we don't want to duplicate it in the backup folder, we move it instead of copying it.
            // Many function states might be present in the backup folder...
            rename(&old_func_folder, func_folder).map_err(|e| CheckpointerError::Other(format!("Failed to move function state folder {} to new location {}: {}", old_func_folder.display(), func_folder.display(), e)))?;
            // recreate empty func folder for backup for backup integrity | old func state are moved to new folder
            create_dir_all(&old_func_folder).map_err(|e| CheckpointerError::FolderCreationError(format!("Failed to create empty function state folder {}: {}", old_func_folder.display(), e)))?;
        }
        Ok(())
    }
    
    fn save_state<T: HasState>(&self,state: &T,target: &CheckpointerTargets,ctx: &()) -> Result<(), CheckpointerError>
    {
        let bckp = <MessagePack as Checkpointer<P, ()>>::backup(self, target, ctx);
        let target = self.retrieve(target);
        let mut wrt = File::create(target).map_err(|e| CheckpointerError::FileCreationError(format!("Failed to create checkpoint file {}: {}", target.display(), e)))?;
        rmp_serde::encode::write(&mut wrt, state.get_state()).map_err(|e| CheckpointerError::WriteError(e.to_string()))?;
        if let Ok(backup_path) = bckp {
            <MessagePack as Checkpointer<P, ()>>::remove(self, &CheckpointerTargets::Backup(backup_path), ctx)?;
        }
        Ok(())
    }
    
    fn load<T: HasState>(&self,target: &CheckpointerTargets,_ctx: &()) -> Result<T, CheckpointerError>
    {
        let target = self.retrieve(target);
        if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint file {}: {}", target.display(), e)))? {
            if target.is_file() {
                let rdr =
                    File::open(target).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open checkpoint file {}: {}", target.display(), e)))?;
                let state: T::State = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
                Ok(T::from_state(state))
            } else {
                Err(CheckpointerError::FileNotFound(format!("The given path is not a file, {}.", target.display())))
            }
        } else {
            Err(CheckpointerError::FileNotFound(format!("The given path does not have any checkpoint file, {}.", target.display())))
        }
    }
    
    fn load_folder<T: HasState>(&self,target: &CheckpointerTargets,_ctx: &()) -> Result<Vec<T>, CheckpointerError>
    {
        let target = self.retrieve(target);
        if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", target.display(), e)))? {
            if target.is_dir() {
                let paths = fs::read_dir(target).map_err(|e| CheckpointerError::Other(format!("Failed to read checkpoint folder {}: {}", target.display(), e)))?;
                let mut states = Vec::new();
                for p in paths {
                    let path = p.map_err(|e| CheckpointerError::Other(format!("Failed to read entry in checkpoint folder {}: {}", target.display(), e)))?.path();
                    if path.is_file()
                    {
                        let rdr = File::open(&path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open checkpoint file {}: {}", path.display(), e)))?;
                        let state: T::State = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
                        states.push(T::from_state(state));
                    }
                }
                Ok(states)
            } else {
                Err(CheckpointerError::FileNotFound(format!("The given path is not a folder, {}.", target.display())))
            }
        } else {
            Err(CheckpointerError::FileNotFound(format!("The given path does not have any checkpoint folder, {}.", target.display())))
        }
    }

    fn remove(&self,target: &CheckpointerTargets,_ctx: &()) -> Result<(), CheckpointerError>
    {
        let target = self.retrieve(target);
        if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint target {}: {}", target.display(), e)))? {
            if target.is_file() {
                std::fs::remove_file(target).map_err(|e| CheckpointerError::Other(format!("Failed to remove checkpoint file {}: {}", target.display(), e)))?;
                Ok(())
            } else if target.is_dir() {
                std::fs::remove_dir_all(target).map_err(|e| CheckpointerError::Other(format!("Failed to remove checkpoint folder {}: {}", target.display(), e)))?;
                Ok(())
            } else {
                Err(CheckpointerError::Other(format!("The given path is neither a file nor a folder, {}.", target.display())))
            }
        } else {
            Err(CheckpointerError::FileNotFound(format!("The given path does not exist, {}.", target.display())))
        }
    }
    
    fn backup(&self,target: &CheckpointerTargets,_ctx: &()) -> Result<PathBuf, CheckpointerError>
    {
        let target = self.retrieve(target);
        let time_str = Utc::now().format("%Y_%m_%d_%H_%M_%S").to_string();
        let backup = target.with_file_name(format!("{}_backup_{}", target.file_name().unwrap().to_str().unwrap(), time_str));
        if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint target {}: {}", target.display(), e)))? {
            if target.is_file() {
                rename(target, &backup).map_err(|e| CheckpointerError::Other(format!("Failed to backup checkpoint file {} to {}: {}", target.display(), backup.display(), e)))?;
                Ok(backup)
            } else if target.is_dir() {
                rename(target, &backup).map_err(|e| CheckpointerError::Other(format!("Failed to backup checkpoint folder {} to {}: {}", target.display(), backup.display(), e)))?;
                Ok(backup)
            } else {
                Err(CheckpointerError::Other(format!("The given path is neither a file nor a folder, {}.", target.display())))
            }
        } else {
            Err(CheckpointerError::FileNotFound(format!("The given path does not exist, {}.", target.display())))
        }
    }
    
    fn save_fn_state<Id: SolId, FnState: FuncState>(&self,id: &Id,state: &FnState,_ctx: &()) -> Result<(), CheckpointerError>
    {
        let folder = self.retrieve(&CheckpointerTargets::FuncFolder);
        let func_folder = folder.join(format!("func_{}", id.to_string()));
        if !func_folder.exists() {
            create_dir_all(&func_folder).map_err(|e| CheckpointerError::FolderCreationError(format!("Failed to create function state folder {}: {}", func_folder.display(), e)))?;
        }
        let id_path = func_folder.join("id.mp");
        let mut wrt = File::create(&id_path).map_err(|e| CheckpointerError::FileCreationError(format!("Failed to create function state file {}: {}", id_path.display(), e)))?;
        rmp_serde::encode::write(&mut wrt, &id).map_err(|e| CheckpointerError::WriteError(e.to_string()))?;
        state.save(func_folder).map_err(|e | CheckpointerError::WriteError(format!("Failed to save function state. {}", e)))?;
        Ok(())
    }

    fn remove_fn_state<Id: SolId, FnState: FuncState>(&self,id: &Id, _ctx: &()) -> Result<(), CheckpointerError> {
        let folder = self.retrieve(&CheckpointerTargets::FuncFolder);
        let func_folder = folder.join(format!("func_{}", id.to_string()));
        std::fs::remove_dir_all(&func_folder).map_err(|e| CheckpointerError::Other(format!("Failed to remove function state folder {}: {}", func_folder.display(), e)))?;
        Ok(())
    }
    
    fn load_fn_state<Id: SolId, FnState: FuncState>(&self,id: Id, _ctx: &()) -> Result<(Id, FnState), CheckpointerError>
    {
        let folder = self.retrieve(&CheckpointerTargets::FuncFolder);
        let func_folder = folder.join(format!("func_{}", id.to_string()));
        if func_folder.exists() && func_folder.is_dir() {
            let id_path = func_folder.join("id.mp");
            if id_path.exists() && id_path.is_file() {
                let rdr = File::open(&id_path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open function state file {}: {}", id_path.display(), e)))?;
                let loaded_id: Id = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
                if loaded_id != id {
                    return Err(CheckpointerError::Other(format!("Loaded function state ID {} does not match requested ID {}.", loaded_id.to_string(), id.to_string())));
                }
                let fn_state = FnState::load(func_folder).map_err(|e| CheckpointerError::ReadError(format!("Failed to load function state. {}", e)))?;
                Ok((loaded_id, fn_state))
            } else {
                Err(CheckpointerError::FileNotFound(format!("Function state file {} does not exist.", id_path.display())))
            }
        } else {
            Err(CheckpointerError::FileNotFound(format!("Function state folder {} does not exist.", func_folder.display())))
        }
    }
    
    fn load_fn_state_folder<Id: SolId, FnState: FuncState>(&self, _ctx: &()) -> Result<Vec<(Id, FnState)>, CheckpointerError>
    {
        let folder = self.retrieve(&CheckpointerTargets::FuncFolder);
        if folder.exists() && folder.is_dir() {
            let mut states = Vec::new();
            for entry in fs::read_dir(folder).map_err(|e| CheckpointerError::Other(format!("Failed to read function state folder {}: {}", folder.display(), e)))? {
                let entry = entry.map_err(|e| CheckpointerError::Other(format!("Failed to read entry in function state folder {}: {}", folder.display(), e)))?;
                let path = entry.path();
                if path.is_dir() {
                    let id_path = path.join("id.mp");
                    if id_path.exists() && id_path.is_file() {
                        let rdr = File::open(&id_path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open function state file {}: {}", id_path.display(), e)))?;
                        let loaded_id: Id = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
                        let fn_state = FnState::load(path.clone()).map_err(|e| CheckpointerError::ReadError(format!("Failed to load function state. {}", e)))?;
                        states.push((loaded_id, fn_state));
                    }
                }
            }
            Ok(states)
        } else {
            Err(CheckpointerError::FileNotFound(format!("Function state folder {} does not exist.", folder.display())))
        }
    }
}

// impl<P, O, S, R, E> Checkpointer<ThrExperiment<P, O, S, R, MessagePack>> for MessagePack
// where
//     ThrExperiment<P, O, S, R, MessagePack>: Runable,
//     P: Problem,
//     O: Optimizer<P>,
//     S: Stop,
//     R: Recorder<ThrExperiment<P, O, S, R, MessagePack>>,
//     E: Evaluator,
// {
//     fn init(&mut self, _runable: &ThrExperiment<P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         let does_exist = self.config.path_check.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", self.config.path_check.display(), e)))?;
//         if does_exist {
//             return Err(CheckpointerError::FileCreationError(format!("{} already exists.", self.config.path_check.display())))
            
//         } else if self.config.path_check.is_file() {
//             return Err(CheckpointerError::FolderCreationError(format!("{} points to a file.", self.config.path_check.display())))
//         } else {
//             create_dir_all(&self.config.path_check).map_err(|e| CheckpointerError::FolderCreationError(format!("Failed to create checkpoint folder {}: {}", self.config.path_check.display(), e)))?;
//         }
//         Ok(())
//     }
    
//     fn before_load(&mut self, _runable: &ThrExperiment<P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         // Check if all folder and files exist
//         if self.config.path_check.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", self.config.path_check.display(), e)))? {
//             if !self.path_config.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of config file {}: {}", self.path_config.display(), e)))? {
//                 return Err(CheckpointerError::FileNotFound(format!("The `config` file does not exist in {}",self.path_config.display())))
//             }
//             if !self.path_optim.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of optimizer file {}: {}", self.path_optim.display(), e)))? {
//                 return Err(CheckpointerError::FileNotFound(format!(
//                     "The `optimizer` file does not exist in {}",
//                     self.path_optim.display()
//                 )))
//             }
//             if !self.path_stop.try_exists().unwrap() {
//                 return Err(CheckpointerError::FileNotFound(format!(
//                     "The `stop` file does not exist in {}",
//                     self.path_stop.display()
//                 )))
//             }
//             if !self.path_eval.try_exists().unwrap() {
//                 return Err(CheckpointerError::FileNotFound(format!(
//                     "The `eval` file does not exist in {}",
//                     self.path_eval.display()
//                 )))
//             }
//             if !self.path_acc.try_exists().unwrap() {
//                 return Err(CheckpointerError::FileNotFound(format!(
//                     "The `accumulator` file does not exist in {}",
//                     self.path_acc.display()
//                 )))
//             }
//         } else {
//             return Err(CheckpointerError::FileNotFound(format!(
//                 "The checkpointer folder does not exists, {}.",
//                 self.config.path_check.display()
//             )))
//         }
//         Ok(())
//     }
    
//     fn after_load(&mut self, runable: &ThrExperiment<P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         // rename old checkpoint folder
//         let mut i = 0;
//         let mut new_path = self
//             .config
//             .path_check
//             .with_file_name(format!("checkpoint_backup_{}", i));
//         while new_path.exists() {
//             new_path = self
//                 .config
//                 .path_check
//                 .with_file_name(format!("checkpoint_backup_{}", i));
//             i += 1;
//         }
//         rename(&self.config.path_check, &new_path).map_err(|e| CheckpointerError::Other(format!("Failed to rename checkpoint folder {} to backup {}: {}", self.config.path_check.display(), new_path.display(), e)))?;
//         self.init(runable)
//     }
    
//     fn save_state<T: HasState>(&self, state: &T, target: &CheckpointerTargets, ctx: &<ThrExperiment<P, O, S, R, MessagePack> as Runable>::Context) -> Result<(), CheckpointerError> {
//         let bckp = self.backup(target, ctx);
//         let target = self.retrieve(target);
//         let mut wrt = File::create(target).map_err(|e| CheckpointerError::FileCreationError(format!("Failed to create checkpoint file {}: {}", target.display(), e)))?;
//         rmp_serde::encode::write(&mut wrt, state.get_state()).map_err(|e| CheckpointerError::WriteError(e.to_string()))?;
//         if let Ok(backup_path) = bckp {
//             self.remove(&CheckpointerTargets::Backup(backup_path), ctx);
//         }
//         Ok(())
//     }
    
//     fn load<T: HasState>(&self, target: &CheckpointerTargets, _ctx: &<ThrExperiment<P, O, S, R, MessagePack> as Runable>::Context) -> Result<T, CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint file {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 let rdr =
//                     File::open(target).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open checkpoint file {}: {}", target.display(), e)))?;
//                 let state: T::State = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                 Ok(T::from_state(state))
//             } else {
//                 Err(CheckpointerError::FileNotFound(format!("The given path is not a file, {}.", target.display())))
//             }
//         } else {
//             Err(CheckpointerError::FileNotFound(format!("The given path does not have any checkpoint file, {}.", target.display())))
//         }
//     }
    
//     fn load_folder<T: HasState>(&self, target: &CheckpointerTargets, _ctx: &<ThrExperiment<P, O, S, R, MessagePack> as Runable>::Context) -> Result<Vec<T>, CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", target.display(), e)))? {
//             if target.is_dir() {
//                 let paths = fs::read_dir(target).map_err(|e| CheckpointerError::Other(format!("Failed to read checkpoint folder {}: {}", target.display(), e)))?;
//                 let mut states = Vec::new();
//                 for p in paths {
//                     let path = p.map_err(|e| CheckpointerError::Other(format!("Failed to read entry in checkpoint folder {}: {}", target.display(), e)))?.path();
//                     if path.is_file()
//                     {
//                         let rdr = File::open(&path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open checkpoint file {}: {}", path.display(), e)))?;
//                         let state: T::State = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                         states.push(T::from_state(state));
//                     }
//                 }
//                 Ok(states)
//             } else {
//                 Err(CheckpointerError::FileNotFound(format!("The given path is not a folder, {}.", target.display())))
//             }
//         } else {
//             Err(CheckpointerError::FileNotFound(format!("The given path does not have any checkpoint folder, {}.", target.display())))
//         }
//     }
    
//     fn remove(&self, target: &CheckpointerTargets, _ctx: &<ThrExperiment<P, O, S, R, MessagePack> as Runable>::Context) -> Result<(), CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint target {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 std::fs::remove_file(target).map_err(|e| CheckpointerError::Other(format!("Failed to remove checkpoint file {}: {}", target.display(), e)))?;
//                 Ok(())
//             } else if target.is_dir() {
//                 std::fs::remove_dir_all(target).map_err(|e| CheckpointerError::Other(format!("Failed to remove checkpoint folder {}: {}", target.display(), e)))?;
//                 Ok(())
//             } else {
//                 Err(CheckpointerError::Other(format!("The given path is neither a file nor a folder, {}.", target.display())))
//             }
//         } else {
//             Err(CheckpointerError::FileNotFound(format!("The given path does not exist, {}.", target.display())))
//         }
//     }
    
//     fn backup(&self, target: &CheckpointerTargets, _ctx: &<ThrExperiment<P, O, S, R, MessagePack> as Runable>::Context) -> Result<PathBuf, CheckpointerError> {
//         let target = self.retrieve(target);
//         let time_str = Utc::now().format("%Y_%m_%d_%H_%M_%S").to_string();
//         let backup = target.with_file_name(format!("{}_backup_{}", target.file_name().unwrap().to_str().unwrap(), time_str));
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint target {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 rename(target, &backup).map_err(|e| CheckpointerError::Other(format!("Failed to backup checkpoint file {} to {}: {}", target.display(), backup.display(), e)))?;
//                 Ok(backup)
//             } else if target.is_dir() {
//                 rename(target, &backup).map_err(|e| CheckpointerError::Other(format!("Failed to backup checkpoint folder {} to {}: {}", target.display(), backup.display(), e)))?;
//                 Ok(backup)
//             } else {
//                 Err(CheckpointerError::Other(format!("The given path is neither a file nor a folder, {}.", target.display())))
//             }
//         } else {
//             Err(CheckpointerError::FileNotFound(format!("The given path does not exist, {}.", target.display())))
//         }
//     }
    
//     fn save_fn_state<Id: SolId, FnState: FuncState>(
//         &self,
//         state: &super::(Id, FnState),
//         target: &CheckpointerTargets,
//         _ctx: &<ThrExperiment<P, O, S, R, MessagePack> as Runable>::Context,
//     ) -> Result<(), CheckpointerError> {
//         match target {
//             CheckpointerTargets::FuncFolder => {
//                 let folder = self.retrieve(target);
//                 let func_folder = folder.join(format!("func_{}", state.0.to_string()));
//                 if !func_folder.exists() {
//                     create_dir_all(&func_folder).map_err(|e| CheckpointerError::FolderCreationError(format!("Failed to create function state folder {}: {}", func_folder.display(), e)))?;
//                 }
//                 let id_path = func_folder.join("id.mp".to_string());
//                 let mut wrt = File::create(&id_path).map_err(|e| CheckpointerError::FileCreationError(format!("Failed to create function state file {}: {}", id_path.display(), e)))?;
//                 rmp_serde::encode::write(&mut wrt, &state.0).map_err(|e| CheckpointerError::WriteError(e.to_string()))?;
//                 state.1.save(func_folder);
//                 Ok(())
//             }
//             _ => Err(CheckpointerError::Other("Function state can only be saved in the FuncFolder target.".to_string())),
//         }
//     }
    
//     fn load_fn_state<Id: SolId, FnState: FuncState>(
//         &self,
//         id: Id,
//         target: &CheckpointerTargets,
//         _ctx: &<ThrExperiment<P, O, S, R, MessagePack> as Runable>::Context,
//     ) -> Result<(Id, FnState), CheckpointerError> {
//         match target {
//             CheckpointerTargets::FuncFolder => {
//                 let folder = self.retrieve(target);
//                 let func_folder = folder.join(format!("func_{}", id.to_string()));
//                 if func_folder.exists() && func_folder.is_dir() {
//                     let id_path = func_folder.join("id.mp".to_string());
//                     if id_path.exists() && id_path.is_file() {
//                         let rdr = File::open(&id_path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open function state file {}: {}", id_path.display(), e)))?;
//                         let loaded_id: Id = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                         if loaded_id != id {
//                             return Err(CheckpointerError::Other(format!("Loaded function state ID {} does not match requested ID {}.", loaded_id.to_string(), id.to_string())));
//                         }
//                         let fn_state = FnState::load(func_folder).map_err(|e| CheckpointerError::ReadError(format!("Failed to load function state. {}", e)))?;
//                         Ok(IdFnState(loaded_id, fn_state))
//                     } else {
//                         Err(CheckpointerError::FileNotFound(format!("Function state file {} does not exist.", id_path.display())))
//                     }
//                 } else {
//                     Err(CheckpointerError::FileNotFound(format!("Function state folder {} does not exist.", func_folder.display())))
//                 }
//             }
//             _ => Err(CheckpointerError::Other("Function state can only be loaded from the FuncFolder target.".to_string())),
//         }
//     }
    
//     fn load_fn_state_folder<Id: SolId, FnState: FuncState>(
//         &self,
//         target: &CheckpointerTargets,
//         _ctx: &<ThrExperiment<P, O, S, R, MessagePack> as Runable>::Context,
//     ) -> Result<Vec<(Id, FnState)>, CheckpointerError> {
//         match target {
//             CheckpointerTargets::FuncFolder => {
//                 let folder = self.retrieve(target);
//                 if folder.exists() && folder.is_dir() {
//                     let mut states = Vec::new();
//                     for entry in fs::read_dir(folder).map_err(|e| CheckpointerError::Other(format!("Failed to read function state folder {}: {}", folder.display(), e)))? {
//                         let entry = entry.map_err(|e| CheckpointerError::Other(format!("Failed to read entry in function state folder {}: {}", folder.display(), e)))?;
//                         let path = entry.path();
//                         if path.is_dir() {
//                             let id_path = path.join("id.mp".to_string());
//                             if id_path.exists() && id_path.is_file() {
//                                 let rdr = File::open(&id_path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open function state file {}: {}", id_path.display(), e)))?;
//                                 let loaded_id: Id = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                                 let fn_state = FnState::load(path.clone()).map_err(|e| CheckpointerError::ReadError(format!("Failed to load function state. {}", e)))?;
//                                 states.push(IdFnState(loaded_id, fn_state));
//                             }
//                         }
//                     }
//                     Ok(states)
//                 } else {
//                     Err(CheckpointerError::FileNotFound(format!("Function state folder {} does not exist.", folder.display())))
//                 }
//             }
//             _ => Err(CheckpointerError::Other("Function states can only be loaded from the FuncFolder target.".to_string())),
//         }
//     }
// }

// #[cfg(feature = "mpi")]
// impl<'a, P, O, S, R, E> Checkpointer<MPIExperiment<'a, P, O, S, R, MessagePack>> for MessagePack
// where
//     P: Problem,
//     O: Optimizer<P>,
//     S: Stop,
//     R: Recorder<MPIExperiment<'a, P, O, S, R, MessagePack>>,
//     MPIExperiment<'a, P, O, S, R, MessagePack>: Runable<Context = MPIProcess>,
//     E: Evaluator,
// {

//     fn init(&mut self, runable: &MPIExperiment<'a, P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         let proc = runable.proc;
//         if self.config.is_dist {
//             let does_exist = self.config.path_check.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", self.config.path_check.display(), e)))?;
//             if does_exist {
//                 CheckpointerError::FileCreationError(format!("{} already exists.", self.config.path_check.display())).into()
//             } else if self.config.path_check.is_file() {
//                 CheckpointerError::FolderCreationError(format!("{} points to a file.", self.config.path_check.display())).into()
//             } else {
//                 create_dir_all(&self.config.path_check).map_err(|e| CheckpointerError::Other(format!("Failed to create checkpoint folder {}: {}", self.config.path_check.display(), e)))?;
//                 create_dir_all(&self.config.path_work).map_err(|e| CheckpointerError::Other(format!("Failed to create working folder {}: {}", self.config.path_work.display(), e)))?;
//             }
//         } else {
//             CheckpointerError::Other(format!("The FolderConfig should be set for MPI environment.")).into()
//         }
//         // Wait for worker
//         proc.world.barrier();
//         Ok(())
//     }
    
//     fn before_load(&mut self, runable: &MPIExperiment<'a, P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         let proc = runable.proc;
//         if self.config.is_dist {
//             // Check if all folder and files exist
//             if self.config.path_check.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", self.config.path_check.display(), e)))? {
//                 if !self.path_config.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of config file {}: {}", self.path_config.display(), e)))? {
//                     CheckpointerError::FileNotFound(format!("The `config` file does not exist in {}",self.path_config.display())).into()
//                 }
//                 if !self.path_optim.try_exists().unwrap() {
//                     CheckpointerError::FileNotFound(format!("The `optimizer` file does not exist in {}",self.path_optim.display())).into()
//                 }
//                 if !self.path_stop.try_exists().unwrap() {
//                     CheckpointerError::FileNotFound(format!("The `stop` file does not exist in {}",self.path_stop.display())).into()
//                 }
//                 if !self.path_eval.try_exists().unwrap() {
//                     CheckpointerError::FileNotFound(format!("The `eval` file does not exist in {}",self.path_eval.display())).into()
//                 }
//                 if !self.path_acc.try_exists().unwrap() {
//                     CheckpointerError::FileNotFound(format!("The `accumulator` file does not exist in {}",self.path_acc.display())).into()
//                 }
//             } else {
//                 CheckpointerError::FileNotFound(format!("The checkpointer folder does not exists, {}.",self.config.path_check.display())).into()
//             }
//         } else {
//             CheckpointerError::Other(format!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.")).into()
//         }
//         proc.world.barrier();
//         Ok(())
//     }
    
//     fn after_load(&mut self, runable: &MPIExperiment<'a, P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         let proc = runable.proc;
//         self.backup(&CheckpointerTargets::Checkpoint, proc)?;
//         proc.world.barrier();
//         self.init_dist(proc)
//     }
    
//     fn save_state<T: HasState>(&self, state: &T, target: &CheckpointerTargets, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<(), CheckpointerError> {
//         let bckp = self.backup(target, ctx);
//         let target = self.retrieve(target);
//         let mut wrt = File::create(target).map_err(|e| CheckpointerError::FileCreationError(format!("Failed to create checkpoint file {}: {}", target.display(), e)))?;
//         rmp_serde::encode::write(&mut wrt, state.get_state()).map_err(|e| CheckpointerError::WriteError(e.to_string()))?;
//         if let Ok(backup_path) = bckp {
//             self.remove(&CheckpointerTargets::Checkpoint(backup_path), ctx);
//         }
//         Ok(())
//     }
    
//     fn load<T: HasState>(&self, target: &CheckpointerTargets, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<T, CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint file {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 let rdr =
//                     File::open(target).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open checkpoint file {}: {}", target.display(), e)))?;
//                 let state = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                 Ok(state)
//             } else {
//                 CheckpointerError::FileNotFound(format!("The given path is not a file, {}.", target.display())).into()
//             }
//         } else {
//             CheckpointerError::FileNotFound(format!("The given path does not have any checkpoint file, {}.", target.display())).into()
//         }
//     }
    
//     fn load_folder<T: HasState>(&self, target: &CheckpointerTargets, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<Vec<T>, CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", target.display(), e)))? {
//             if target.is_dir() {
//                 let paths = fs::read_dir(target).map_err(|e| CheckpointerError::Other(format!("Failed to read checkpoint folder {}: {}", target.display(), e)))?;
//                 let mut states = Vec::new();
//                 for p in paths {
//                     let path = p.map_err(|e| CheckpointerError::Other(format!("Failed to read entry in checkpoint folder {}: {}", target.display(), e)))?.path();
//                     if path.is_file()
//                     {
//                         let rdr = File::open(&path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open checkpoint file {}: {}", path.display(), e)))?;
//                         let state = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                         states.push(state);
//                     }
//                 }
//                 Ok(states)
//             } else {
//                 CheckpointerError::FileNotFound(format!("The given path is not a folder, {}.", target.display())).into()
//             }
//         } else {
//             CheckpointerError::FileNotFound(format!("The given path does not have any checkpoint folder, {}.", target.display())).into()
//         }
//     }
    
//     fn remove(&self, target: &Self::Target, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<(), CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint target {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 std::fs::remove_file(target).map_err(|e| CheckpointerError::Other(format!("Failed to remove checkpoint file {}: {}", target.display(), e)))?;
//                 Ok(())
//             } else if target.is_dir() {
//                 std::fs::remove_dir_all(target).map_err(|e| CheckpointerError::Other(format!("Failed to remove checkpoint folder {}: {}", target.display(), e)))?;
//                 Ok(())
//             } else {
//                 CheckpointerError::Other(format!("The given path is neither a file nor a folder, {}.", target.display())).into()
//             }
//         } else {
//             CheckpointerError::FileNotFound(format!("The given path does not exist, {}.", target.display())).into()
//         }
//     }
    
//     fn backup(&self, target: &Self::Target, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<PathBuf, CheckpointerError> {
//         let target = self.retrieve(target);
//         let time_str = Utc::now().format("%Y_%m_%d_%H_%M_%S").to_string();
//         let backup = target.with_file_name(format!("{}_backup_{}", target.file_name().unwrap().to_str().unwrap(), time_str));
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint target {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 rename(target, &backup).map_err(|e| CheckpointerError::Other(format!("Failed to backup checkpoint file {} to {}: {}", target.display(), backup.display(), e)))?;
//                 Ok(backup)
//             } else if target.is_dir() {
//                 rename(target, &backup).map_err(|e| CheckpointerError::Other(format!("Failed to backup checkpoint folder {} to {}: {}", target.display(), backup.display(), e)))?;
//                 Ok(backup)
//             } else {
//                 CheckpointerError::Other(format!("The given path is neither a file nor a folder, {}.", target.display())).into()
//             }
//         } else {
//             CheckpointerError::FileNotFound(format!("The given path does not exist, {}.", target.display())).into()
//         }
//     }

//     fn save_fn_state<Id: SolId, FnState: FuncState>(
//         &self,
//         state: &super::(Id, FnState),
//         target: &CheckpointerTargets,
//         _ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context,
//     ) -> Result<(), CheckpointerError> {
//         match target {
//             CheckpointerTargets::FuncFolder => {
//                 let folder = self.retrieve(target);
//                 let func_folder = folder.join(format!("func_{}", state.0.to_string()));
//                 if !func_folder.exists() {
//                     create_dir_all(&func_folder).map_err(|e| CheckpointerError::FolderCreationError(format!("Failed to create function state folder {}: {}", func_folder.display(), e)))?;
//                 }
//                 let id_path = func_folder.join("id.mp".to_string());
//                 let mut wrt = File::create(&id_path).map_err(|e| CheckpointerError::FileCreationError(format!("Failed to create function state file {}: {}", id_path.display(), e)))?;
//                 rmp_serde::encode::write(&mut wrt, &state.0).map_err(|e| CheckpointerError::WriteError(e.to_string()))?;
//                 state.1.save(func_folder);
//                 Ok(())
//             }
//             _ => Err(CheckpointerError::Other("Function state can only be saved in the FuncFolder target.".to_string())),
//         }
//     }
    
//     fn load_fn_state<Id: SolId, FnState: FuncState>(
//         &self,
//         id: Id,
//         target: &CheckpointerTargets,
//         _ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context,
//     ) -> Result<(Id, FnState), CheckpointerError> {
//         match target {
//             CheckpointerTargets::FuncFolder => {
//                 let folder = self.retrieve(target);
//                 let func_folder = folder.join(format!("func_{}", id.to_string()));
//                 if func_folder.exists() && func_folder.is_dir() {
//                     let id_path = func_folder.join("id.mp".to_string());
//                     if id_path.exists() && id_path.is_file() {
//                         let rdr = File::open(&id_path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open function state file {}: {}", id_path.display(), e)))?;
//                         let loaded_id: Id = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                         if loaded_id != id {
//                             return Err(CheckpointerError::Other(format!("Loaded function state ID {} does not match requested ID {}.", loaded_id.to_string(), id.to_string())));
//                         }
//                         let fn_state = FnState::load(func_folder).map_err(|e| CheckpointerError::ReadError(format!("Failed to load function state. {}", e)))?;
//                         Ok(IdFnState(loaded_id, fn_state))
//                     } else {
//                         Err(CheckpointerError::FileNotFound(format!("Function state file {} does not exist.", id_path.display())))
//                     }
//                 } else {
//                     Err(CheckpointerError::FileNotFound(format!("Function state folder {} does not exist.", func_folder.display())))
//                 }
//             }
//             _ => Err(CheckpointerError::Other("Function state can only be loaded from the FuncFolder target.".to_string())),
//         }
//     }
    
//     fn load_fn_state_folder<Id: SolId, FnState: FuncState>(
//         &self,
//         target: &CheckpointerTargets,
//         _ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context,
//     ) -> Result<Vec<(Id, FnState)>, CheckpointerError> {
//         match target {
//             CheckpointerTargets::FuncFolder => {
//                 let folder = self.retrieve(target);
//                 if folder.exists() && folder.is_dir() {
//                     let mut states = Vec::new();
//                     for entry in fs::read_dir(folder).map_err(|e| CheckpointerError::Other(format!("Failed to read function state folder {}: {}", folder.display(), e)))? {
//                         let entry = entry.map_err(|e| CheckpointerError::Other(format!("Failed to read entry in function state folder {}: {}", folder.display(), e)))?;
//                         let path = entry.path();
//                         if path.is_dir() {
//                             let id_path = path.join("id.mp".to_string());
//                             if id_path.exists() && id_path.is_file() {
//                                 let rdr = File::open(&id_path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open function state file {}: {}", id_path.display(), e)))?;
//                                 let loaded_id: Id = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                                 let fn_state = FnState::load(path.clone()).map_err(|e| CheckpointerError::ReadError(format!("Failed to load function state. {}", e)))?;
//                                 states.push(IdFnState(loaded_id, fn_state));
//                             }
//                         }
//                     }
//                     Ok(states)
//                 } else {
//                     Err(CheckpointerError::FileNotFound(format!("Function state folder {} does not exist.", folder.display())))
//                 }
//             }
//             _ => Err(CheckpointerError::Other("Function states can only be loaded from the FuncFolder target.".to_string())),
//         }
//     }
// }

// #[cfg(feature = "mpi")]
// impl<'a, P, O, S, R, E> Checkpointer<BaseWorker<>> for MessagePack
// where
//     P: Problem,
//     O: Optimizer<P>,
//     S: Stop,
//     R: Recorder<MPIExperiment<'a, P, O, S, R, MessagePack>>,
//     MPIExperiment<'a, P, O, S, R, MessagePack>: Runable<Context = MPIProcess>,
//     E: Evaluator,
// {

//     fn init(&mut self, runable: &MPIExperiment<'a, P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         let proc = runable.proc;
//         if self.config.is_dist {
//             let does_exist = self.config.path_check.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", self.config.path_check.display(), e)))?;
//             if does_exist {
//                 CheckpointerError::FileCreationError(format!("{} already exists.", self.config.path_check.display())).into()
//             } else if self.config.path_check.is_file() {
//                 CheckpointerError::FolderCreationError(format!("{} points to a file.", self.config.path_check.display())).into()
//             } else {
//                 create_dir_all(&self.config.path_check).map_err(|e| CheckpointerError::Other(format!("Failed to create checkpoint folder {}: {}", self.config.path_check.display(), e)))?;
//                 create_dir_all(&self.config.path_work).map_err(|e| CheckpointerError::Other(format!("Failed to create working folder {}: {}", self.config.path_work.display(), e)))?;
//             }
//         } else {
//             CheckpointerError::Other(format!("The FolderConfig should be set for MPI environment.")).into()
//         }
//         // Wait for worker
//         proc.world.barrier();
//         Ok(())
//     }
    
//     fn before_load(&mut self, runable: &MPIExperiment<'a, P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         let proc = runable.proc;
//         if self.config.is_dist {
//             // Check if all folder and files exist
//             if self.config.path_check.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", self.config.path_check.display(), e)))? {
//                 if !self.path_config.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of config file {}: {}", self.path_config.display(), e)))? {
//                     CheckpointerError::FileNotFound(format!("The `config` file does not exist in {}",self.path_config.display())).into()
//                 }
//                 if !self.path_optim.try_exists().unwrap() {
//                     CheckpointerError::FileNotFound(format!("The `optimizer` file does not exist in {}",self.path_optim.display())).into()
//                 }
//                 if !self.path_stop.try_exists().unwrap() {
//                     CheckpointerError::FileNotFound(format!("The `stop` file does not exist in {}",self.path_stop.display())).into()
//                 }
//                 if !self.path_eval.try_exists().unwrap() {
//                     CheckpointerError::FileNotFound(format!("The `eval` file does not exist in {}",self.path_eval.display())).into()
//                 }
//                 if !self.path_acc.try_exists().unwrap() {
//                     CheckpointerError::FileNotFound(format!("The `accumulator` file does not exist in {}",self.path_acc.display())).into()
//                 }
//             } else {
//                 CheckpointerError::FileNotFound(format!("The checkpointer folder does not exists, {}.",self.config.path_check.display())).into()
//             }
//         } else {
//             CheckpointerError::Other(format!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.")).into()
//         }
//         proc.world.barrier();
//         Ok(())
//     }
    
//     fn after_load(&mut self, runable: &MPIExperiment<'a, P, O, S, R, MessagePack>) -> Result<(), CheckpointerError> {
//         let proc = runable.proc;
//         self.backup(&CheckpointerTargets::Checkpoint, proc)?;
//         proc.world.barrier();
//         self.init_dist(proc)
//     }
    
//     fn save_state<T: HasState>(&self, state: &T, target: &CheckpointerTargets, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<(), CheckpointerError> {
//         let bckp = self.backup(target, ctx);
//         let target = self.retrieve(target);
//         let mut wrt = File::create(target).map_err(|e| CheckpointerError::FileCreationError(format!("Failed to create checkpoint file {}: {}", target.display(), e)))?;
//         rmp_serde::encode::write(&mut wrt, state.get_state()).map_err(|e| CheckpointerError::WriteError(e.to_string()))?;
//         if let Ok(backup_path) = bckp {
//             self.remove(&CheckpointerTargets::Checkpoint(backup_path), ctx);
//         }
//         Ok(())
//     }
    
//     fn load<T: HasState>(&self, target: &CheckpointerTargets, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<T, CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint file {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 let rdr =
//                     File::open(target).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open checkpoint file {}: {}", target.display(), e)))?;
//                 let state = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                 Ok(state)
//             } else {
//                 CheckpointerError::FileNotFound(format!("The given path is not a file, {}.", target.display())).into()
//             }
//         } else {
//             CheckpointerError::FileNotFound(format!("The given path does not have any checkpoint file, {}.", target.display())).into()
//         }
//     }
    
//     fn load_folder<T: HasState>(&self, target: &CheckpointerTargets, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<Vec<T>, CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint folder {}: {}", target.display(), e)))? {
//             if target.is_dir() {
//                 let paths = fs::read_dir(target).map_err(|e| CheckpointerError::Other(format!("Failed to read checkpoint folder {}: {}", target.display(), e)))?;
//                 let mut states = Vec::new();
//                 for p in paths {
//                     let path = p.map_err(|e| CheckpointerError::Other(format!("Failed to read entry in checkpoint folder {}: {}", target.display(), e)))?.path();
//                     if path.is_file()
//                     {
//                         let rdr = File::open(&path).map_err(|e| CheckpointerError::FileNotFound(format!("Failed to open checkpoint file {}: {}", path.display(), e)))?;
//                         let state = rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointerError::ReadError(e.to_string()))?;
//                         states.push(state);
//                     }
//                 }
//                 Ok(states)
//             } else {
//                 CheckpointerError::FileNotFound(format!("The given path is not a folder, {}.", target.display())).into()
//             }
//         } else {
//             CheckpointerError::FileNotFound(format!("The given path does not have any checkpoint folder, {}.", target.display())).into()
//         }
//     }
    
//     fn remove(&self, target: &Self::Target, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<(), CheckpointerError> {
//         let target = self.retrieve(target);
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint target {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 std::fs::remove_file(target).map_err(|e| CheckpointerError::Other(format!("Failed to remove checkpoint file {}: {}", target.display(), e)))?;
//                 Ok(())
//             } else if target.is_dir() {
//                 std::fs::remove_dir_all(target).map_err(|e| CheckpointerError::Other(format!("Failed to remove checkpoint folder {}: {}", target.display(), e)))?;
//                 Ok(())
//             } else {
//                 CheckpointerError::Other(format!("The given path is neither a file nor a folder, {}.", target.display())).into()
//             }
//         } else {
//             CheckpointerError::FileNotFound(format!("The given path does not exist, {}.", target.display())).into()
//         }
//     }
    
//     fn backup(&self, target: &Self::Target, ctx: &<MPIExperiment<'a, P, O, S, R, MessagePack> as Runable>::Context) -> Result<PathBuf, CheckpointerError> {
//         let target = self.retrieve(target);
//         let time_str = Utc::now().format("%Y_%m_%d_%H_%M_%S").to_string();
//         let backup = target.with_file_name(format!("{}_backup_{}", target.file_name().unwrap().to_str().unwrap(), time_str));
//         if target.try_exists().map_err(|e| CheckpointerError::Other(format!("Failed to check existence of checkpoint target {}: {}", target.display(), e)))? {
//             if target.is_file() {
//                 rename(target, &backup).map_err(|e| CheckpointerError::Other(format!("Failed to backup checkpoint file {} to {}: {}", target.display(), backup.display(), e)))?;
//                 Ok(backup)
//             } else if target.is_dir() {
//                 rename(target, &backup).map_err(|e| CheckpointerError::Other(format!("Failed to backup checkpoint folder {} to {}: {}", target.display(), backup.display(), e)))?;
//                 Ok(backup)
//             } else {
//                 CheckpointerError::Other(format!("The given path is neither a file nor a folder, {}.", target.display())).into()
//             }
//         } else {
//             CheckpointerError::FileNotFound(format!("The given path does not exist, {}.", target.display())).into()
//         }
//     }
// }

// #[cfg(feature = "mpi")]
// impl WCheckMessagePack {
//     /// Returns a [`WCheckMessagePack`] for the given [`MPIProcess`].
//     pub fn new(config: Arc<FolderConfig>, proc: &MPIProcess) -> Self {
//         if !config.is_dist {
//             panic!(
//                 "The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method."
//             )
//         }
//         let path_check = config
//             .path_work
//             .join(format!("worker_rank{}.mp", proc.rank));
//         WCheckMessagePack(path_check)
//     }
// }

// #[cfg(feature = "mpi")]
// impl<WState: WorkerState> WorkerCheckpointer<WState> for WCheckMessagePack {
//     type FnStateCheck = MPFnStateCheckpointer;
//     /// Checks if folders and files already exists, if so [`panic!`] for [WorkerCheckpointer].
//     ///
//     /// # Note
//     ///
//     /// Here, we suppose a [`MasterWorker`](crate::MasterWorker) distribution.
//     /// The [`Worker`](crate::Worker) wait for the Master to create and check the folder hierarchy,
//     /// with a [`barrier`](mpi::collective::CommunicatorCollectives::barrier).
//     /// Then the [`Worker`](crate::Worker) checks if its own checkpoint file exists.
//     fn init(&mut self, proc: &MPIProcess) {
//         proc.world.barrier();
//         if self.0.try_exists().unwrap() {
//             panic!(
//                 "The Worker folder {} in checkpoint already exists.",
//                 self.0.display()
//             )
//         }
//     }
//     /// Ran after [`init`](WorkerCheckpointer::init), and after a [`load!`](crate::load).
//     /// Checks if the folder and file hierarchy exists, based on [`FolderConfig`]. If not, then [`panic!`].
//     ///
//     /// # Note
//     ///
//     /// Here, we suppose a [`MasterWorker`](crate::MasterWorker) distribution.
//     /// The [`Worker`](crate::Worker) wait for the Master to check the folder hierarchy,
//     /// with a [`barrier`](mpi::collective::CommunicatorCollectives::barrier).
//     /// Then the [`Worker`](crate::Worker) checks if its own checkpoint file exists.
//     fn before_load(&mut self, proc: &MPIProcess) {
//         proc.world.barrier();
//         // Check if all folder and files exist
//         if !self.0.try_exists().unwrap() {
//             panic!(
//                 "The Worker folder {} in checkpoint does not exists.",
//                 self.0.display()
//             )
//         }
//     }

//     fn after_load(&mut self, proc: &MPIProcess) {
//         proc.world.barrier(); // First barrier for all workers to finish checking their checkpoint file
//         proc.world.barrier(); // Second barrier for all workers to be ready to load their checkpoint file
//     }

//     /// Saves a checkpoint from the given [`WorkerState`], according to a [`FolderConfig`].
//     fn save_state(&self, state: &WState, _rank: Rank) {
//         let mut wrt = File::create(&self.0).unwrap();
//         rmp_serde::encode::write(&mut wrt, state).unwrap();
//     }

//     /// Loads a checkpoint from an already saved [`WorkerState`], according to a [`FolderConfig`].
//     fn load(&self, _rank: Rank) -> Result<WState, CheckpointError> {
//         // Check if file exist
//         if self
//             .0
//             .try_exists()
//             .map_err(|e| CheckpointError(e.to_string()))?
//         {
//             let rdr = File::open(&self.0).map_err(|e| CheckpointError(e.to_string()))?;
//             Ok(rmp_serde::decode::from_read(rdr).map_err(|e| CheckpointError(e.to_string()))?)
//         } else {
//             Err(CheckpointError(String::from(
//                 "The given path does not have any checkpoint folder",
//             )))
//         }
//     }

//     fn new_func_state_checkpointer(&self) -> Self::FnStateCheck {
//         let path = self.0.parent().unwrap().to_path_buf();
//         MPFnStateCheckpointer { path }
//     }
// }
