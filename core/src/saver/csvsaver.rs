use crate::{
    domain::Domain, objective::{Codomain, Outcome}, optimizer::{OptInfo, Optimizer}, saver::Saver, searchspace::Searchspace, solution::{Computed, Partial, SolInfo}
};
use std::{fmt::{Debug, Display}, fs::{create_dir, create_dir_all}, path::{Path, PathBuf}};
use csv::Writer;

/// A CSV [`Saver`] taking a path of where the save folder should be created.
/// 
/// # Attribute
/// 
/// * `path` : `&'static` [`str`]  - The path to where the folder should be created.
/// Creates all parents folder that might not  exist yet.
/// * `sep` : [`char] - The separator between columns of the CSV files.
/// 
pub struct CSVSaver{
    pub path: PathBuf,
    pub sep: char,
    pub save_opt : bool,
    path_evals : PathBuf,
    path_pobj : PathBuf,
    path_popt : Option<PathBuf>,
    path_codom : PathBuf,
    path_out : PathBuf,
    path_check: PathBuf,
}

/// A [`CSVWritable`] is an object for wich a CSV header can be given,
/// and where its components can be written as a [`Vec`] of `&`[`str`].
pub trait CSVWritable{
    type Component;
    fn header(&self)->Vec<String>;
    fn write(&self, comp : Self::Component)->Vec<String>;
}

impl  CSVSaver{
    pub fn new(path : &str, sep : char, save_opt:bool) ->CSVSaver{
        let true_path = PathBuf::from(path);
        let does_exist =  true_path.try_exists().unwrap();
        if does_exist{
            panic!("The folder path already exists, {}.",path)
        }
        else if true_path.is_file(){
            panic!("The given path cannot be pointing at a file, {}.",path)
        }
        else{

            let path_evals = true_path.join(Path::new("evals"));
            let path_check = true_path.join(Path::new("check"));

            let path_pobj = path_evals.join(Path::new("obj.csv"));
            let path_codom = path_evals.join(Path::new("codom.csv"));
            let path_out = path_evals.join(Path::new("out.csv"));
            let path_popt = match save_opt{
                true => Some(path_evals.join(Path::new("opt.csv"))),
                false => None,
            };

            CSVSaver{
                path: true_path,
                sep,
                save_opt,
                path_evals,
                path_pobj,
                path_popt,
                path_codom,
                path_out,
                path_check,
            }
        }
    }
}

impl <Optim,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info> Saver<Optim,PObj, CObj, POpt, COpt, Obj, Opt, SInfo, Cod, Out, Scp, Info> for CSVSaver
where
    Optim: Optimizer<PObj,CObj,POpt,COpt,Obj,Opt,SInfo,Cod,Out,Scp,Info>,
    PObj: Partial<Obj, SInfo>,
    CObj: Computed<PObj, Obj, SInfo, Cod, Out>,
    POpt: Partial<Opt, SInfo>,
    COpt: Computed<POpt, Opt, SInfo, Cod, Out>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<PObj, POpt, Obj, Opt, SInfo> + CSVWritable,
    Info: OptInfo
{
    fn init(&self, scp: &Scp) {
        create_dir_all(self.path.as_path()).unwrap();
        create_dir(self.path_evals.as_path()).unwrap();
        create_dir(self.path_check.as_path()).unwrap();

        let mut wrt_obj = Writer::from_path(self.path_pobj.as_path()).unwrap();
        if let Some(ppopt) = &self.path_popt{
            let mut wrt_opt = Writer::from_path(ppopt.as_path()).unwrap();
        }
        Writer::from_path(self.path_codom.as_path()).unwrap();
    }

    fn save_partial(&self, obj: &PObj, opt: &POpt, sp: Scp, info: Info) {
        todo!()
    }

    fn save_codom(&self, obj: &CObj, sp: Scp, info: Info) {
        todo!()
    }

    fn save_out(&self, id: (u32, usize), out: Out, sp: Scp, info: Info) {
        todo!()
    }

    fn save_state(&self, state: &Optim) {
        todo!()
    }
}