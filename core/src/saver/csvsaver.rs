use crate::{
    domain::Domain,
    objective::{Codomain, LinkedOutcome, Outcome},
    optimizer::{ArcVecArc, OptInfo, OptState},
    saver::Saver,
    searchspace::Searchspace,
    solution::{Computed, Id, Partial, SolInfo, Solution},
    stop::Stop,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::{
    fmt::{Debug, Display}, fs::{create_dir, create_dir_all, File}, path::{Path, PathBuf}, sync::{Arc, Mutex}
};

/// A CSV [`Saver`] taking a path of where the save folder should be created.
///
/// # Attribute
///
/// * `path` : `&'static` [`str`]  - The path to where the folder should be created.
///   Creates all parents folder that might not  exist yet.
/// * `sep` : [`char] - The separator between columns of the CSV files.
///
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
    path_check: Option<(PathBuf,PathBuf)>,
    _header_init_obj: bool,
    _header_init_opt: bool,
    _header_init_codom: bool,
    _header_init_out: bool,
}

/// A [`CSVWritable`] is an object for wich a CSV header can be given,
/// and where its components can be written as a [`Vec`] of [`String`].
pub trait CSVWritable<C> {
    fn header(&self) -> Vec<String>;
    fn write(&self, comp: &C) -> Vec<String>;
}

/// A [`CSVLeftRight`] describes a CSV writable object made of two components (eg. `Obj` and `Opt`).
pub trait CSVLeftRight<L, R> {
    fn header(&self) -> Vec<String>;
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
            let path_evals = true_path.join(Path::new("evals"));
            create_dir_all(true_path.as_path()).unwrap();
            create_dir(path_evals.as_path()).unwrap();

            let path_pobj = match save_obj {
                true => {
                    let path = path_evals.join(Path::new("obj.csv"));
                    csv::Writer::from_path(path.as_path()).unwrap();
                    Some(path)
                },
                false => None,
            };
            let path_popt = match save_opt {
                true => {
                    let path = path_evals.join(Path::new("opt.csv"));
                    csv::Writer::from_path(path.as_path()).unwrap();
                    Some(path)
                },
                false => None,
            };
            let path_out = match save_out {
                true => {
                    let path = path_evals.join(Path::new("out.csv")) ;
                    csv::Writer::from_path(path.as_path()).unwrap();
                    Some(path)
                },
                false => None,
            };

            let path_check = if checkpoint > 0 {
                let path = true_path.join(Path::new("check"));
                create_dir(path.as_path()).unwrap();
                let pste = path.join(Path::new("state_opt.json"));
                let pstp = path.join(Path::new("state_stp.json"));
                Some((pste,pstp))
            } else {
                None
            };
            let path_codom = path_evals.join(Path::new("codom.csv"));
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
                _header_init_obj: false,
                _header_init_opt: false,
                _header_init_codom: false,
                _header_init_out: false,
            }
        }
    }
}

impl<'de, SolId, St, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>
    Saver<SolId, St, PObj, POpt, Obj, Opt, SInfo, Cod, Out, Scp, Info, State> for CSVSaver
where
    State: OptState + Serialize + Deserialize<'de>,
    St: Stop + Serialize + Deserialize<'de>,
    PObj: Partial<SolId, Obj, SInfo> + Send + Sync,
    POpt: Partial<SolId, Opt, SInfo> + Send + Sync,
    Obj: Domain + Clone + Display + Debug + Send + Sync,
    Opt: Domain + Clone + Display + Debug + Send + Sync,
    SInfo: SolInfo + CSVWritable<()> + Send + Sync,
    Cod: Codomain<Out> + Send + Sync,
    Cod::TypeCodom: CSVWritable<()> + Send + Sync,
    Out: Outcome + CSVWritable<()> + Send + Sync,
    Scp: Searchspace<SolId, PObj, POpt, Obj, Opt, SInfo>
        + CSVLeftRight<Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>
        + Send
        + Sync,
    Info: OptInfo + CSVWritable<()> + Send + Sync,
    SolId: Id + PartialEq + Copy + Clone + CSVWritable<()> + Send + Sync,
{
    fn init(&mut self) {}

    fn save_partial(
        &mut self,
        obj: ArcVecArc<PObj>,
        opt: ArcVecArc<POpt>,
        sp: Arc<Scp>,
        info: Arc<Info>,
    ) {
        if let Some(ppobj) = &self.path_pobj {
            let wrt = Arc::new(Mutex::new(csv::Writer::from_path(ppobj.as_path()).unwrap()));

            if !self._header_init_obj {
                let op = &obj[0];
                let id = op.get_id();
                let sinfo = op.get_info();
                let mut idstr = id.header();
                idstr.extend(sp.header());
                idstr.extend(sinfo.header());
                idstr.extend(info.header());
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(idstr).unwrap();
                wrt_local.flush().unwrap();
                self._header_init_obj = true;
            }

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
            let wrt = Arc::new(Mutex::new(csv::Writer::from_path(ppopt.as_path()).unwrap()));

            if !self._header_init_opt {
                let op = &opt[0];
                let id = op.get_id();
                let sinfo = op.get_info();
                let mut idstr = id.header();
                idstr.extend(sp.header());
                idstr.extend(sinfo.header());
                idstr.extend(info.header());
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(idstr).unwrap();
                wrt_local.flush().unwrap();
                self._header_init_opt = true;
            }

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

    fn save_codom(&mut self, obj: ArcVecArc<Computed<SolId, PObj, Obj, Cod, Out, SInfo>>) {
        let wrt: Arc<Mutex<csv::Writer<std::fs::File>>> = Arc::new(Mutex::new(
            csv::Writer::from_path(self.path_codom.as_path()).unwrap(),
        ));
        if !self._header_init_codom {
            let op = &obj[0];
            let id = op.get_id();
            let codom = op.get_y();
            let mut idstr = id.header();
            idstr.extend(codom.header());
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
            self._header_init_codom = true;
        }

        obj.par_iter().for_each(|op| {
            let id = op.get_id();
            let codom = op.get_y();
            let mut idstr = id.write(&());
            idstr.extend(codom.write(&()));
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn save_out(&mut self, out: Vec<LinkedOutcome<Out, PObj, SolId, Obj, SInfo>>) {
        if let Some(ppout) = &self.path_out {
            let wrt = Arc::new(Mutex::new(csv::Writer::from_path(ppout.as_path()).unwrap()));

            if !self._header_init_out {
                let o = &out[0];
                let id = o.sol.get_id();
                let output = &o.out;
                let mut idstr = id.header();
                idstr.extend(output.header());
                {
                    let mut wrt_local = wrt.lock().unwrap();
                    wrt_local.write_record(&idstr).unwrap();
                    wrt_local.flush().unwrap();
                }
                self._header_init_out = true;
            }

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

    fn save_state(&mut self, _sp: Arc<Scp>, state: &State, stop: &St) {
        if let Some(path) = &self.path_check{
            let wrt = File::create(path.0.as_path()).unwrap();
            serde_json::to_writer(wrt, state).unwrap();
            let wrt = File::create(path.1.as_path()).unwrap();
            serde_json::to_writer(wrt, stop).unwrap();
        } 
    }

    fn clean(&mut self) {
        println!("{:?}",&self.path);
        std::fs::remove_dir_all(&self.path).unwrap();
    }
}
