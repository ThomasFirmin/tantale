use crate::{
    Objective, domain::{Domain, TypeDom}, experiment::Evaluate, objective::{Codomain, LinkedOutcome, Outcome}, optimizer::{ArcVecArc, Optimizer}, saver::{CheckpointError, Saver}, searchspace::Searchspace, solution::{Computed, Id, Partial, Solution}, stop::Stop
};
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json;
use std::{
    fs::{create_dir, create_dir_all, File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// A CSV [`Saver`] taking a path of where the save folder should be created.
///
/// # Attribute
///
/// * `path` : `&'static` [`str`]  - The path to where the folder should be created.
///   Creates all parents folder that might not  exist yet.
/// * `sep` : [`char] - The separator between columns of the CSV files.
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
    _headers_done: bool,
}

/// A [`CSVWritable`] is an object for wich a CSV header can be given,
/// and where its components can be written as a [`Vec`] of [`String`].
pub trait CSVWritable<H, C> {
    fn header(elem: &H) -> Vec<String>;
    fn write(&self, comp: &C) -> Vec<String>;
}

/// A [`CSVLeftRight`] describes a CSV writable object made of two components (eg. `Obj` and `Opt`).
pub trait CSVLeftRight<H, L, R> {
    fn header(elem: &H) -> Vec<String>;
    fn write_left(&self, comp: &L) -> Vec<String>;
    fn write_right(&self, comp: &R) -> Vec<String>;
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
        let does_exist = true_path.try_exists().unwrap();
        if does_exist {
            panic!("The folder path already exists, {}.", path)
        } else if true_path.is_file() {
            panic!("The given path cannot be pointing at a file, {}.", path)
        } else {
            let path_evals = true_path.join(Path::new("evaluations"));
            create_dir_all(true_path.as_path()).unwrap();
            create_dir(path_evals.as_path()).unwrap();

            let path_pobj = match save_obj {
                true => {
                    let path = path_evals.join(Path::new("obj.csv"));
                    csv::Writer::from_path(path.as_path()).unwrap();
                    Some(path)
                }
                false => None,
            };
            let path_popt = match save_opt {
                true => {
                    let path = path_evals.join(Path::new("opt.csv"));
                    csv::Writer::from_path(path.as_path()).unwrap();
                    Some(path)
                }
                false => None,
            };
            let path_out = match save_out {
                true => {
                    let path = path_evals.join(Path::new("out.csv"));
                    csv::Writer::from_path(path.as_path()).unwrap();
                    Some(path)
                }
                false => None,
            };

            let path_check = if checkpoint > 0 {
                let path = true_path.join(Path::new("checkpoint"));
                create_dir(path.as_path()).unwrap();
                let pste = path.join(Path::new("state_opt.json"));
                let pstp = path.join(Path::new("state_stp.json"));
                let psav = path.join(Path::new("state_sav.json"));
                let peva = path.join(Path::new("state_eval.json"));
                Some((pste, pstp, psav, peva))
            } else {
                None
            };
            let path_codom = path_evals.join(Path::new("cod.csv"));
            csv::Writer::from_path(path_codom.as_path()).unwrap();

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
                _headers_done: false,
            }
        }
    }
}

impl<SolId, St, Obj, Opt, Cod, Out, Scp,Op, Ob, Eval>
    Saver<SolId, St, Obj, Opt, Cod, Out, Scp, Op, Ob, Eval> for CSVSaver
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
    TypeDom<Obj>: Send + Sync,
    TypeDom<Opt>: Send + Sync,
    Op: Optimizer<SolId,Obj,Opt,Cod,Out,Scp>,
    Op::Info:  CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::State: Serialize + DeserializeOwned,
    Ob: Objective<Obj,Cod,Out>,
    Eval: Evaluate<Ob,St,Obj,Opt,Out,Cod,Op::Info,Op::SInfo,SolId>,
{
    fn init(&mut self, sp: Arc<Scp>, cod: Arc<Cod>) {
        if !self._headers_done {
            if let Some(ppobj) = &self.path_pobj {
                let file = OpenOptions::new()
                    .append(true)
                    .open(ppobj.as_path())
                    .unwrap();
                let mut wrt = csv::Writer::from_writer(file);
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(sp.as_ref()));
                idstr.extend(Op::SInfo::header(&()));
                idstr.extend(Op::Info::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }
            if let Some(ppopt) = &self.path_popt {
                let file = OpenOptions::new()
                    .append(true)
                    .open(ppopt.as_path())
                    .unwrap();
                let mut wrt = csv::Writer::from_writer(file);
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(sp.as_ref()));
                idstr.extend(Op::SInfo::header(&()));
                idstr.extend(Op::Info::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }
            if let Some(ppout) = &self.path_out {
                let file = OpenOptions::new()
                    .append(true)
                    .open(ppout.as_path())
                    .unwrap();
                let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));
                let mut idstr = SolId::header(&());
                idstr.extend(Out::header(&()));
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
            }
            {
                let file = OpenOptions::new()
                    .append(true)
                    .open(self.path_codom.as_path())
                    .unwrap();
                let wrt = Arc::new(Mutex::new(csv::Writer::from_writer(file)));
                let mut idstr = SolId::header(&());
                idstr.extend(Cod::header(cod.as_ref()));
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
            }
        }
    }

    fn save_partial(
        &self,
        obj: ArcVecArc<Partial<SolId, Obj, Op::SInfo>>,
        opt: ArcVecArc<Partial<SolId, Opt, Op::SInfo>>,
        sp: Arc<Scp>,
        _cod: Arc<Cod>,
        info: Arc<Op::Info>,
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

    fn save_state(&self, _sp: Arc<Scp>, state: &Op::State, stop: &St, eval:&Eval) {
        if let Some(path) = &self.path_check {
            let wrt = File::create(path.0.as_path()).unwrap();
            serde_json::to_writer(wrt, state).unwrap();
            let wrt = File::create(path.1.as_path()).unwrap();
            serde_json::to_writer(wrt, stop).unwrap();
            let wrt = File::create(path.2.as_path()).unwrap();
            serde_json::to_writer(wrt, self).unwrap();
            let wrt = File::create(path.3.as_path()).unwrap();
            serde_json::to_writer(wrt, eval).unwrap();
        }
    }

    fn clean(self) {
        std::fs::remove_dir_all(&self.path).unwrap();
    }

    fn load_saver(path: &str,_sp:&Scp, _cod:&Cod) -> Result<Self, CheckpointError> {
        let path = Path::new(path);
        let path_check = path.join(Path::new("checkpoint"));

        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_sav = path_check.join(Path::new("state_sav.json"));
            if path_sav.is_file(){
                let rdrsav = File::open(path_sav).unwrap();
                Ok(serde_json::from_reader(rdrsav).unwrap())
            } else {
                Err(CheckpointError(String::from("Saver state file state_sav.json does not exists.")))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }


    fn load_stop(path: &str,_sp:&Scp, _cod:&Cod) -> Result<St, CheckpointError> {
        let path = Path::new(path);
        let path_check = path.join(Path::new("checkpoint"));

        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_stp = path_check.join(Path::new("state_stp.json"));
            if path_stp.is_file() {
                let rdrstp = File::open(path_stp).unwrap();
                Ok(serde_json::from_reader(rdrstp).unwrap())
            } else {
                Err(CheckpointError(String::from("Stop state file state_stp.json does not exists.")))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }


    fn load_optimizer(path: &str,_sp:&Scp, _cod:&Cod) -> Result<Op, CheckpointError> {
        let path = Path::new(path);
        let path_check = path.join(Path::new("checkpoint"));

        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_opt = path_check.join(Path::new("state_opt.json"));
            if path_opt.is_file() {
                let rdropt = File::open(path_opt).unwrap();
                Ok(Op::from_state(serde_json::from_reader(rdropt).unwrap()))
            } else {
                Err(CheckpointError(String::from("Optimizer state file state_opt.json does not exists.")))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

    fn load_evaluate(path: &str,_sp:&Scp, _cod:&Cod) -> Result<Eval, CheckpointError> {
        let path = Path::new(path);
        let path_check = path.join(Path::new("checkpoint"));

        let does_exist = path_check.try_exists().unwrap();
        if does_exist {
            let path_eva = path_check.join(Path::new("state_eva.json"));
            if path_eva.is_file() {
                let rdreva = File::open(path_eva).unwrap();
                Ok(serde_json::from_reader(rdreva).unwrap())
            } else {
                Err(CheckpointError(String::from("One state file does not exists among state_sav.json, state_opt.json, state_stp.json.")))
            }
        } else {
            Err(CheckpointError(String::from(
                "The given path does not have any checkpoint folder",
            )))
        }
    }

}
