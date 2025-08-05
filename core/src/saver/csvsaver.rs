use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    optimizer::{OptInfo, Optimizer,ArcVecArc},
    saver::Saver,
    searchspace::Searchspace,
    solution::{Computed, Partial, SolInfo, Id},
};
use csv::Writer;
use std::{
    sync::Arc,
    fmt::{Debug, Display},
    fs::{create_dir, create_dir_all},
    path::{Path, PathBuf},
    thread,
};
use serde::{Serialize,Deserialize};
use rayon::prelude::*;

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
    pub sep: char,
    pub save_obj: bool,
    pub save_opt: bool,
    pub save_out: bool,
    pub checkpoint: usize,
    path_evals: PathBuf,
    path_pobj: Option<PathBuf>,
    path_popt: Option<PathBuf>,
    path_codom: PathBuf,
    path_out: Option<PathBuf>,
    path_check: Option<PathBuf>,
    _header_init_partial: bool,
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
        sep: char,
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
                Some(true_path.join(Path::new("check")))
            } else {
                None
            };
            let path_codom = path_evals.join(Path::new("codom.csv"));

            CSVSaver {
                path: true_path,
                sep,
                save_obj,
                save_opt,
                save_out,
                checkpoint,
                path_evals,
                path_pobj,
                path_popt,
                path_codom,
                path_out,
                path_check,
                _header_init_partial: false,
                _header_init_codom: false,
                _header_init_out: false,
            }
        }
    }
}

impl<'de, SolId, Optim, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info>
    Saver<SolId, Optim, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info> for CSVSaver
where
    Optim: Optimizer<SolId, PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info> + Serialize + Deserialize<'de>,
    PObj: Partial<SolId, Obj, SInfo> + Send + Sync + CSVWritable<PObj>,
    CObj: Computed<PObj, SolId, Obj, SInfo, Cod, Out>,
    POpt: Partial<SolId, Opt, SInfo> + Send + Sync + CSVWritable<()>,
    COpt: Computed<POpt, SolId, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo + CSVWritable<()>,
    Cod: Codomain<Out>,
    Cod::TypeCodom: CSVWritable<()>,
    Out: Outcome + CSVWritable<()>,
    Scp: Searchspace<SolId, PObj, POpt, Obj, Opt, SInfo> + CSVLeftRight<Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>> + Send + Sync,
    Info: OptInfo + CSVWritable<()> + Send + Sync,
    SolId: Id + PartialEq + Copy + Clone,
{
    fn init(&self) {
        create_dir_all(self.path.as_path()).unwrap();
        create_dir(self.path_evals.as_path()).unwrap();

        if let Some(ppobj) = &self.path_pobj {
            Writer::from_path(ppobj.as_path()).unwrap();
        }
        if let Some(ppopt) = &self.path_popt {
            Writer::from_path(ppopt.as_path()).unwrap();
        }
        if let Some(ppout) = &self.path_out {
            Writer::from_path(ppout.as_path()).unwrap();
        }
        if let Some(ppcheck) = &self.path_check {
            create_dir(ppcheck.as_path()).unwrap();
        }

        Writer::from_path(self.path_codom.as_path()).unwrap();
    }
    
    fn save_partial(&self, obj: ArcVecArc<PObj>, opt: ArcVecArc<POpt>, sp: Arc<Scp>, info: Arc<Info>) {
        if self.save_obj{
            let sol: Vec<Vec<String>> = obj.par_iter().map(
                |op|
                {   
                    let id = op.get_id();
                    let sinfo = op.get_info();
                    let mut pstr = sp.write_left(&op.get_x().clone());
                    pstr.extend(sinfo.write(&()));
                    pstr.extend(info.write(&()));
                    pstr
                }
            ).collect();
        }
        if self.save_opt{
            let sol: Vec<Vec<String>> = opt.par_iter().map(
                |op|
                {   
                    let id = op.get_id();
                    let sinfo = op.get_info();
                    let mut pstr = sp.write_right(&op.get_x().clone());
                    pstr.extend(sinfo.write(&()));
                    pstr.extend(info.write(&()));
                    pstr
                }
            ).collect();
        }
    }
    
    fn save_codom(&self, obj: ArcVecArc<CObj>, sp: Arc<Scp>, info: Arc<Info>) {
        todo!()
    }
    
    fn save_out(&self, id: (u32, usize), out: Out, sp: Arc<Scp>, info: Arc<Info>) {
        todo!()
    }
    
    fn save_state(&self, state: &Optim) {
        todo!()
    }
    
}
