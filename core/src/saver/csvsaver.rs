use crate::{
    GlobalParameters, OPT_ID, RUN_ID, SOL_ID, domain::Domain, experiment::Evaluate, objective::{Codomain, LinkedOutcome, Outcome}, optimizer::{ArcVecArc, Optimizer, opt::{Batch, BatchType}}, saver::{CheckpointError, Saver}, searchspace::Searchspace, solution::{Computed, Id, Solution}, stop::Stop
};

use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json;
use std::{
    fs::{create_dir, create_dir_all, File, OpenOptions},
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc, Mutex},
};

#[cfg(feature = "mpi")]
use crate::saver::DistributedSaver;
#[cfg(feature = "mpi")]
use mpi::Rank;

/// A [`CSVWritable`] is an object for wich a CSV header can be given,
/// and how its components can be written as a [`Vec`] of [`String`].
pub trait CSVWritable<H, C> {
    fn header(elem: &H) -> Vec<String>;
    fn write(&self, comp: &C) -> Vec<String>;
}

/// A [`CSVLeftRight`] describes a [`CSVWritable`] object made of two components (eg. `Obj` and `Opt`).
pub trait CSVLeftRight<H, L, R> {
    fn header(elem: &H) -> Vec<String>;
    fn write_left(&self, comp: &L) -> Vec<String>;
    fn write_right(&self, comp: &R) -> Vec<String>;
}

/// A [`CSVWrite`] describes the object for which the component has to be written within a CSV file.
pub trait CSVWrite<T>
{
    fn write(elem: &H) -> Vec<String>;
}

/// A [`CSVSaver`] taking a path of where the save folder should be created.
/// The computed [`Codomain`] are always saved by default.
///
/// # Attribute
///
/// * `path` : `&'static` [`str`]  - The path to where the folder should be created.
///   Creates all parents folder that might not  exist yet.
/// * `save_obj` : bool - If `true` computed objective [`Partial`] will be saved.
/// * `save_opt` : bool - If `true` computed optimizer [`Partial`] will be saved.
/// * `save_out` : bool - If `true` computed [`Outcome`] will be saved.
/// * `checkpoint` : usize - If `>0`, a checkpoint will be created every `checkpoint` call to [`step`](Optimizer::step).
#[derive(Serialize, Deserialize)]
pub struct CSVSaver {
    pub path: PathBuf,
    pub save_obj: bool,
    pub save_opt: bool,
    pub save_out: bool,
    pub checkpoint: usize,
    path_pobj: Option<PathBuf>,
    path_popt: Option<PathBuf>,
    path_codom: PathBuf,
    path_out: Option<PathBuf>,
    path_check: Option<(PathBuf, PathBuf, PathBuf, PathBuf)>,
}

impl CSVSaver {
    pub fn new(
        path: &str,
        save_obj: bool,
        save_opt: bool,
        save_out: bool,
        checkpoint: usize,
    ) -> CSVSaver {
        let true_path = PathBuf::from(path);
        let path_evals = true_path.join(Path::new("evaluations"));

        let path_pobj = match save_obj {
            true => Some(path_evals.join(Path::new("obj.csv"))),
            false => None,
        };
        let path_popt = match save_opt {
            true => Some(path_evals.join(Path::new("opt.csv"))),
            false => None,
        };
        let path_out = match save_out {
            true => Some(path_evals.join(Path::new("out.csv"))),
            false => None,
        };

        let path_check = if checkpoint > 0 {
            let path = true_path.join(Path::new("checkpoint"));
            let pste = path.join(Path::new("state_opt.json"));
            let pstp = path.join(Path::new("state_stp.json"));
            let peva = path.join(Path::new("state_eval.json"));
            let ppar = path.join(Path::new("state_param.json"));
            Some((pste, pstp, peva, ppar))
        } else {
            None
        };
        let path_codom = path_evals.join(Path::new("cod.csv"));

        CSVSaver {
            path: true_path,
            save_obj,
            save_opt,
            save_out,
            checkpoint,
            path_pobj,
            path_popt,
            path_codom,
            path_out,
            path_check,
        }
    }
}

impl<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>
    Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval> for CSVSaver
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    St: Stop + Serialize + DeserializeOwned,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>
        + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>
        + Send
        + Sync,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    Eval: Evaluate,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::State: Serialize + DeserializeOwned,
{
    fn init(&mut self, sp: &Scp, cod: &Cod) {
        let does_exist = self.path.try_exists().unwrap();
        if does_exist {
            panic!("The folder path already exists, {}.", self.path.display())
        } else if self.path.is_file() {
            panic!(
                "The given path cannot point to a file, {}.",
                self.path.display()
            )
        } else {
            let path_evals = self.path.join(Path::new("evaluations"));
            create_dir_all(self.path.as_path()).unwrap();
            create_dir(path_evals.as_path()).unwrap();

            if let Some(ppobj) = &self.path_pobj {
                let mut wrt = csv::Writer::from_path(ppobj.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(sp));
                idstr.extend(Op::SInfo::header(&()));
                idstr.extend(Op::Info::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }

            if let Some(ppopt) = &self.path_popt {
                let mut wrt = csv::Writer::from_path(ppopt.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(sp));
                idstr.extend(Op::SInfo::header(&()));
                idstr.extend(Op::Info::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }

            if let Some(ppout) = &self.path_out {
                let mut wrt = csv::Writer::from_path(ppout.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Out::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }

            {
                let mut wrt = csv::Writer::from_path(self.path_codom.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Cod::header(cod));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }

            if self.path_check.is_some() {
                let path = self.path.join(Path::new("checkpoint"));
                create_dir(path.as_path()).unwrap();
            }
        }
    }

    fn after_load(&mut self, _sp: &Scp, _cod: &Cod) {
        let does_exist = self.path.try_exists().unwrap();
        if does_exist {
            let path_evals = self.path.join(Path::new("evaluations"));
            if !path_evals.try_exists().unwrap() {
                panic!(
                    "The folder path for evaluations does not exists, {}.",
                    path_evals.display()
                )
            }

            if let Some(ppobj) = &self.path_pobj {
                if !ppobj.try_exists().unwrap() {
                    panic!(
                        "The file path for Objective solutions does not exists, {}.",
                        ppobj.display()
                    )
                }
            }

            if let Some(ppopt) = &self.path_popt {
                if !ppopt.try_exists().unwrap() {
                    panic!(
                        "The file path for Optimizer solutions does not exists, {}.",
                        ppopt.display()
                    )
                }
            }

            if let Some(ppout) = &self.path_out {
                if !ppout.try_exists().unwrap() {
                    panic!(
                        "The file path for Output does not exists, {}.",
                        ppout.display()
                    )
                }
            }

            if !self.path_codom.try_exists().unwrap() {
                panic!(
                    "The file path for Codomain does not exists, {}.",
                    self.path_codom.display()
                )
            }

            if self.path_check.is_some() {
                let path = self.path.join(Path::new("checkpoint"));
                if !path.try_exists().unwrap() {
                    panic!(
                        "The folder path for checkpoints does not exists, {}.",
                        path.display()
                    )
                }
            }
        } else {
            panic!("The folder path does not exists, {}.", self.path.display())
        }
    }

    fn save_partial(
        &self,
        batch : Op::BType,
        sp: Arc<Scp>,
        _cod: Arc<Cod>,
    ) {
        if let Some(ppobj) = &self.path_pobj {
            let file = OpenOptions::new()
                .append(true)
                .open(ppobj.as_path())
                .unwrap();
            let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));
            obj.par_iter().for_each(|op| {
                let id = op.get_id();
                let sinfo = op.get_info();
                let mut idstr = id.write(&());
                idstr.extend(sp.write_left(&op.get_x().clone()));
                idstr.extend(sinfo.write(&()));
                idstr.extend(info.write(&()));
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
            });
        }
        if let Some(ppopt) = &self.path_popt {
            let file = OpenOptions::new()
                .append(true)
                .open(ppopt.as_path())
                .unwrap();
            let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));

            opt.par_iter().for_each(|op| {
                let id = op.get_id();
                let sinfo = op.get_info();
                let mut idstr = id.write(&());
                idstr.extend(sp.write_right(&op.get_x().clone()));
                idstr.extend(sinfo.write(&()));
                idstr.extend(info.write(&()));
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
            });
        }
    }

    fn save_codom(
        &self,
        obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        _sp: Arc<Scp>,
        cod: Arc<Cod>,
    ) {
        let file = OpenOptions::new()
            .append(true)
            .open(self.path_codom.as_path())
            .unwrap();
        let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));
        obj.par_iter().for_each(|op| {
            let id = op.get_id();
            let codom = op.get_y();
            let mut idstr = id.write(&());
            idstr.extend(cod.write(codom.as_ref()));
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn save_out(&self, out: Vec<LinkedOutcome<Out, SolId, Obj, Op::SInfo>>, _sp: Arc<Scp>) {
        if let Some(ppout) = &self.path_out {
            let file = OpenOptions::new()
                .append(true)
                .open(ppout.as_path())
                .unwrap();
            let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));

            out.par_iter().for_each(|o| {
                let id = o.sol.get_id();
                let output = &o.out;
                let mut idstr = id.write(&());
                idstr.extend(output.write(&()));
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
            });
        }
    }

    fn save_state(&self, _sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval) {
        if let Some(path) = &self.path_check {
            let wrt = File::create(path.0.as_path()).unwrap();
            serde_json::to_writer(wrt, state).unwrap();
            let wrt = File::create(path.1.as_path()).unwrap();
            serde_json::to_writer(wrt, stop).unwrap();
            let wrt = File::create(path.2.as_path()).unwrap();
            serde_json::to_writer(wrt, eval).unwrap();

            let global = GlobalParameters {
                sold_id: SOL_ID.load(Ordering::Relaxed),
                opt_id: OPT_ID.load(Ordering::Relaxed),
                run_id: RUN_ID.load(Ordering::Relaxed),
            };

            let wrt = File::create(path.3.as_path()).unwrap();
            serde_json::to_writer(wrt, &global).unwrap();
        }
    }

    fn clean(self) {
        std::fs::remove_dir_all(&self.path).unwrap();
    }

    fn load_stop(&self, _sp: &Scp, _cod: &Cod) -> Result<St, CheckpointError> {
        let path_check = self.path.join(Path::new("checkpoint"));
        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_stp = path_check.join(Path::new("state_stp.json"));
            if path_stp.is_file() {
                let rdrstp = File::open(path_stp).unwrap();
                Ok(serde_json::from_reader(rdrstp).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "Stop state file state_stp.json does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_optimizer(&self, _sp: &Scp, _cod: &Cod) -> Result<Op, CheckpointError> {
        let path_check = self.path.join(Path::new("checkpoint"));
        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_opt = path_check.join(Path::new("state_opt.json"));
            if path_opt.is_file() {
                let rdropt = File::open(path_opt).unwrap();
                Ok(Op::from_state(serde_json::from_reader(rdropt).unwrap()))
            } else {
                Err(CheckpointError(String::from(
                    "Optimizer state file state_opt.json does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_evaluate(&self, _sp: &Scp, _cod: &Cod) -> Result<Eval, CheckpointError> {
        let path_check = self.path.join(Path::new("checkpoint"));
        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_eva = path_check.join(Path::new("state_eval.json"));
            if path_eva.is_file() {
                let rdreva = File::open(path_eva).unwrap();
                Ok(serde_json::from_reader(rdreva).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "Evaluator state file state_eval.json does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_parameters(&self, _sp: &Scp, _cod: &Cod) -> Result<GlobalParameters, CheckpointError> {
        let path_check = self.path.join(Path::new("checkpoint"));
        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_par = path_check.join(Path::new("state_param.json"));
            if path_par.is_file() {
                let rdrpar = File::open(path_par).unwrap();
                Ok(serde_json::from_reader(rdrpar).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "Parameters state file state_param.json does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load(&self, _sp: &Scp, _cod: &Cod) -> Result<(St, Op, Eval), CheckpointError> {
        let global: GlobalParameters =
            <CSVSaver as Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>>::load_parameters(
                self, _sp, _cod,
            )?;
        SOL_ID.store(global.sold_id, Ordering::Release);
        OPT_ID.store(global.sold_id, Ordering::Release);
        Ok((
            <CSVSaver as Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>>::load_stop(
                self, _sp, _cod,
            )?,
            <CSVSaver as Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>>::load_optimizer(
                self, _sp, _cod,
            )?,
            <CSVSaver as Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>>::load_evaluate(
                self, _sp, _cod,
            )?,
        ))
    }
}

/// Version of [`CSVSaver`] for MPI-distributed algorithms..
/// The computed [`Codomain`] are always saved by default.
///
/// # Attribute
///
/// * `path` : `&'static` [`str`]  - The path to where the folder should be created.
///   Creates all parents folder that might not  exist yet.
/// * `save_obj` : bool - If `true` computed objective [`Partial`] will be saved.
/// * `save_opt` : bool - If `true` computed optimizer [`Partial`] will be saved.
/// * `save_out` : bool - If `true` computed [`Outcome`] will be saved.
/// * `checkpoint` : usize - If `>0`, a checkpoint will be created every `checkpoint` call to [`step`](Optimizer::step).
#[cfg(feature = "mpi")]
impl<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval>
    DistributedSaver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Eval> for CSVSaver
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    St: Stop + Serialize + DeserializeOwned,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom> + Send + Sync,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<SolId, Obj, Opt, Op::SInfo>
        + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>
        + Send
        + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op: Optimizer<SolId, Obj, Opt, Cod, Out, Scp>,
    BType: BatchType<SolId,Obj,Opt,Op::SInfo,Op::Info>,
    Eval: Evaluate,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::State: Serialize + DeserializeOwned,
{
    fn init(&mut self, sp: &Scp, cod: &Cod, rank: Rank) {
        let does_exist = self.path.try_exists().unwrap();
        if does_exist {
            if rank == 0 {
                panic!("The folder path already exists, {}.", self.path.display())
            }
        } else if self.path.is_file() {
            panic!(
                "The given path cannot point to a file, {}.",
                self.path.display()
            )
        } else if rank == 0 {
            create_dir_all(self.path.as_path()).unwrap();
            let path_evals = self
                .path
                .join(Path::new(&format!("evaluations_rk{}", rank)));
            create_dir(path_evals.as_path()).unwrap();

            if self.path_pobj.is_some() {
                let nppobj = path_evals.join(Path::new("obj.csv"));
                let mut wrt = csv::Writer::from_path(nppobj.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(sp));
                idstr.extend(Op::SInfo::header(&()));
                idstr.extend(Op::Info::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
                self.path_pobj.replace(nppobj);
            }

            if self.path_popt.is_some() {
                let nppopt = path_evals.join(Path::new("opt.csv"));
                let mut wrt = csv::Writer::from_path(nppopt.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(sp));
                idstr.extend(Op::SInfo::header(&()));
                idstr.extend(Op::Info::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
                self.path_popt.replace(nppopt);
            }

            if self.path_out.is_some() {
                let nppout = path_evals.join(Path::new("out.csv"));
                let mut wrt = csv::Writer::from_path(nppout.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Out::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
                self.path_out.replace(nppout);
            }

            {
                let nppcod = path_evals.join(Path::new("cod.csv"));
                let mut wrt = csv::Writer::from_path(nppcod.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Cod::header(cod));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
                self.path_codom = nppcod;
            }

            if self.path_check.is_some() {
                let path = self.path.join(Path::new(&format!("checkpoint_rk{}", rank)));
                let pste = path.join(Path::new("state_opt.json"));
                let pstp = path.join(Path::new("state_stp.json"));
                let peva = path.join(Path::new("state_eval.json"));
                let ppar = path.join(Path::new("state_param.json"));
                self.path_check.replace((pste, pstp, peva, ppar));
                create_dir(path.as_path()).unwrap();
            }
        }
    }

    fn after_load(&mut self, _sp: &Scp, _cod: &Cod, rank: Rank) {
        let does_exist = self.path.try_exists().unwrap();
        if does_exist {
            let path_evals = self
                .path
                .join(Path::new(&format!("evaluations_rk{}", rank)));

            if !path_evals.try_exists().unwrap() {
                panic!(
                    "The folder path for evaluations does not exists, {}.",
                    path_evals.display()
                )
            }

            if self.path_pobj.is_some() {
                let nppobj = path_evals.join(Path::new("obj.csv"));
                if !nppobj.try_exists().unwrap() {
                    panic!(
                        "The file path for Objective solutions does not exists, {}.",
                        nppobj.display()
                    )
                }
                self.path_pobj = Some(nppobj);
            }

            if self.path_popt.is_some() {
                let nppopt = path_evals.join(Path::new("opt.csv"));
                if !nppopt.try_exists().unwrap() {
                    panic!(
                        "The file path for Optimizer solutions does not exists, {}.",
                        nppopt.display()
                    )
                }
                self.path_popt = Some(nppopt);
            }

            if self.path_out.is_some() {
                let nppout = path_evals.join(Path::new("out.csv"));
                if !nppout.try_exists().unwrap() {
                    panic!(
                        "The file path for Output does not exists, {}.",
                        nppout.display()
                    )
                }
                self.path_out = Some(nppout);
            }

            let nppcod = path_evals.join(Path::new("cod.csv"));
            if !nppcod.try_exists().unwrap() {
                panic!(
                    "The file path for Codomain does not exists, {}.",
                    nppcod.display()
                )
            }
            self.path_codom = nppcod;

            if self.path_check.is_some() {
                let path = self.path.join(Path::new(&format!("checkpoint_rk{}", rank)));
                if !path.try_exists().unwrap() {
                    panic!(
                        "The folder path for checkpoints does not exists, {}.",
                        path.display()
                    )
                }
                let pste = path.join(Path::new("state_opt.json"));
                let pstp = path.join(Path::new("state_stp.json"));
                let peva = path.join(Path::new("state_eval.json"));
                let ppar = path.join(Path::new("state_param.json"));
                self.path_check.replace((pste, pstp, peva, ppar));
            }
        } else {
            panic!("The folder path does not exists, {}.", self.path.display())
        }
    }

    fn save_partial(
        &self,
        batch : Op::BType,
        sp: Arc<Scp>,
        _cod: Arc<Cod>,
        _rank: Rank,
    ) {
        if let Some(ppobj) = &self.path_pobj {
            let file = OpenOptions::new()
                .append(true)
                .open(ppobj.as_path())
                .unwrap();
            let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));
            obj.par_iter().for_each(|op| {
                let id = op.get_id();
                let sinfo = op.get_info();
                let mut idstr = id.write(&());
                idstr.extend(sp.write_left(&op.get_x().clone()));
                idstr.extend(sinfo.write(&()));
                idstr.extend(info.write(&()));
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
            });
        }
        if let Some(ppopt) = &self.path_popt {
            let file = OpenOptions::new()
                .append(true)
                .open(ppopt.as_path())
                .unwrap();
            let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));

            opt.par_iter().for_each(|op| {
                let id = op.get_id();
                let sinfo = op.get_info();
                let mut idstr = id.write(&());
                idstr.extend(sp.write_right(&op.get_x().clone()));
                idstr.extend(sinfo.write(&()));
                idstr.extend(info.write(&()));
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
            });
        }
    }

    fn save_codom(
        &self,
        obj: ArcVecArc<Computed<SolId, Obj, Cod, Out, Op::SInfo>>,
        _sp: Arc<Scp>,
        cod: Arc<Cod>,
        _rank: Rank,
    ) {
        let file = OpenOptions::new()
            .append(true)
            .open(self.path_codom.as_path())
            .unwrap();
        let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));
        obj.par_iter().for_each(|op| {
            let id = op.get_id();
            let codom = op.get_y();
            let mut idstr = id.write(&());
            idstr.extend(cod.write(codom.as_ref()));
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn save_out(
        &self,
        out: Vec<LinkedOutcome<Out, SolId, Obj, Op::SInfo>>,
        _sp: Arc<Scp>,
        _rank: Rank,
    ) {
        if let Some(ppout) = &self.path_out {
            let file = OpenOptions::new()
                .append(true)
                .open(ppout.as_path())
                .unwrap();
            let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));

            out.par_iter().for_each(|o| {
                let id = o.sol.get_id();
                let output = &o.out;
                let mut idstr = id.write(&());
                idstr.extend(output.write(&()));
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
            });
        }
    }

    fn save_state(&self, _sp: Arc<Scp>, state: &Op::State, stop: &St, eval: &Eval, _rank: Rank) {
        if let Some(path) = &self.path_check {
            let wrt = File::create(path.0.as_path()).unwrap();
            serde_json::to_writer(wrt, state).unwrap();
            let wrt = File::create(path.1.as_path()).unwrap();
            serde_json::to_writer(wrt, stop).unwrap();
            let wrt = File::create(path.2.as_path()).unwrap();
            serde_json::to_writer(wrt, eval).unwrap();

            let global = GlobalParameters {
                sold_id: SOL_ID.load(Ordering::Relaxed),
                opt_id: OPT_ID.load(Ordering::Relaxed),
                run_id: RUN_ID.load(Ordering::Relaxed),
            };

            let wrt = File::create(path.3.as_path()).unwrap();
            serde_json::to_writer(wrt, &global).unwrap();
        }
    }

    fn load_stop(&self, _sp: &Scp, _cod: &Cod, rank: Rank) -> Result<St, CheckpointError> {
        let path_check = self.path.join(Path::new(&format!("checkpoint_rk{}", rank)));
        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_stp = path_check.join(Path::new("state_stp.json"));
            if path_stp.is_file() {
                let rdrstp = File::open(path_stp).unwrap();
                Ok(serde_json::from_reader(rdrstp).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "Stop state file state_stp.json does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_optimizer(&self, _sp: &Scp, _cod: &Cod, rank: Rank) -> Result<Op, CheckpointError> {
        let path_check = self.path.join(Path::new(&format!("checkpoint_rk{}", rank)));
        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_opt = path_check.join(Path::new("state_opt.json"));
            if path_opt.is_file() {
                let rdropt = File::open(path_opt).unwrap();
                Ok(Op::from_state(serde_json::from_reader(rdropt).unwrap()))
            } else {
                Err(CheckpointError(String::from(
                    "Optimizer state file state_opt.json does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_evaluate(&self, _sp: &Scp, _cod: &Cod, rank: Rank) -> Result<Eval, CheckpointError> {
        let path_check = self.path.join(Path::new(&format!("checkpoint_rk{}", rank)));
        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_eva = path_check.join(Path::new("state_eval.json"));
            if path_eva.is_file() {
                let rdreva = File::open(path_eva).unwrap();
                Ok(serde_json::from_reader(rdreva).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "Evaluator state file state_eval.json does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_parameters(
        &self,
        _sp: &Scp,
        _cod: &Cod,
        rank: Rank,
    ) -> Result<GlobalParameters, CheckpointError> {
        let path_check = self.path.join(Path::new(&format!("checkpoint_rk{}", rank)));
        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_par = path_check.join(Path::new("state_param.json"));
            if path_par.is_file() {
                let rdrpar = File::open(path_par).unwrap();
                Ok(serde_json::from_reader(rdrpar).unwrap())
            } else {
                Err(CheckpointError(String::from(
                    "Parameters state file state_param.json does not exists.",
                )))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load(&self, _sp: &Scp, _cod: &Cod, rank: Rank) -> Result<(St, Op, Eval), CheckpointError> {
        let global: GlobalParameters = <CSVSaver as DistributedSaver<
            SolId,
            St,
            Obj,
            Opt,
            Cod,
            Out,
            Scp,
            Op,
            Eval,
        >>::load_parameters(self, _sp, _cod, rank)?;
        SOL_ID.store(global.sold_id, Ordering::Release);
        OPT_ID.store(global.sold_id, Ordering::Release);
        RUN_ID.store(global.run_id, Ordering::Release);
        Ok(
            (
                <CSVSaver as DistributedSaver<
                    SolId,
                    St,
                    Obj,
                    Opt,
                    Cod,
                    Out,
                    Scp,
                    Op,
                    Eval,
                >>::load_stop(self, _sp, _cod, rank)?,
                <CSVSaver as DistributedSaver<
                    SolId,
                    St,
                    Obj,
                    Opt,
                    Cod,
                    Out,
                    Scp,
                    Op,
                    Eval,
                >>::load_optimizer(self, _sp, _cod, rank)?,
                <CSVSaver as DistributedSaver<
                    SolId,
                    St,
                    Obj,
                    Opt,
                    Cod,
                    Out,
                    Scp,
                    Op,
                    Eval,
                >>::load_evaluate(self, _sp, _cod, rank)?,
            ),
        )
    }
}
