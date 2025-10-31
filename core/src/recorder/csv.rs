use crate::{
    FolderConfig, Onto, OptInfo, Partial, SaverConfig, SolInfo,
    domain::{Domain, onto::OntoDom},
    experiment::Evaluate,
    objective::{Codomain, Outcome},
    optimizer::{Optimizer, opt::{CBType, OBType}},
    recorder::Recorder,
    searchspace::Searchspace,
    solution::{Batch, BatchType, Id, Solution, partial::BasePartial},
    stop::Stop
};

use bincode::config::Config;
#[cfg(feature = "mpi")]
use mpi::Rank;
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions, create_dir, create_dir_all},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex}
};

#[cfg(feature = "mpi")]
use crate::{config::DistSaverConfig, recorder::DistRecorder};

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

/// A [`CSVWrite`] describes a [`BatchType`] and its associated types that can be written within a CSV file.
pub trait BatchCSVWrite<PSol, Scp, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    Self: BatchType<SolId, Obj, Opt, SInfo, Info>,
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo> + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Info: OptInfo + CSVWritable<(), ()> + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
{
    fn write_partial_obj(cbatch: &Self::Comp<Cod, Out>,wrt: csv::Writer<File>,scp: Arc<Scp>);
    fn write_partial_opt(cbatch: &Self::Comp<Cod, Out>,wrt: csv::Writer<File>,scp: Arc<Scp>);
    fn write_info(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>);
    fn write_codom(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, cod: Arc<Cod>);
    fn write_out(obatch: &Self::Outc<Out>, wrt: csv::Writer<File>);
}

impl<Scp, SolId, Obj, Opt, SInfo, Info, Cod, Out>
    BatchCSVWrite<BasePartial<SolId, Obj, SInfo>, Scp, SolId, Obj, Opt, SInfo, Info, Cod, Out>
    for Batch<BasePartial<SolId, Obj, SInfo>, SolId, Obj, Opt, SInfo, Info>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom> + Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<BasePartial<SolId, Obj, SInfo>, SolId, Obj, Opt, SInfo> + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Info: OptInfo + CSVWritable<(), ()> + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn write_partial_obj(
        cbatch: &Self::Comp<Cod, Out>,
        wrt: csv::Writer<File>,
        scp: Arc<Scp>,
    ) {
        let amwrt = Arc::new(Mutex::new(wrt));
        cbatch.cobj.par_iter().for_each(|op| {
            let id = op.get_id();
            let mut idstr = id.write(&());
            idstr.extend(scp.write_left(&op.get_x().clone()));
            {
                let mut wrt_local = amwrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn write_partial_opt(
        cbatch: &Self::Comp<Cod, Out>,
        wrt: csv::Writer<File>,
        scp: Arc<Scp>,
    ) {
        let amwrt = Arc::new(Mutex::new(wrt));
        cbatch.copt.par_iter().for_each(|op| {
            let id = op.get_id();
            let mut idstr = id.write(&());
            idstr.extend(scp.write_right(&op.get_x().clone()));
            {
                let mut wrt_local = amwrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn write_info(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>) {
        let amwrt = Arc::new(Mutex::new(wrt));
        cbatch.cobj.par_iter().for_each(|op| {
            let id = op.get_id();
            let sinfo = op.get_info();
            let mut idstr = id.write(&());
            idstr.extend(sinfo.write(&()));
            idstr.extend(cbatch.info.write(&()));
            {
                let mut wrt_local = amwrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn write_codom(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, cod: Arc<Cod>) {
        let wrt = Arc::new(Mutex::new(wrt));
        cbatch.cobj.par_iter().for_each(|op| {
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

    fn write_out(obatch: &Self::Outc<Out>, wrt: csv::Writer<File>) {
        let wrt = Arc::new(Mutex::new(wrt));
        obatch.robj.par_iter().for_each(|s| {
            let id = s.get_id();
            let output = s.get_out();
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

/// A [`CSVSaver`] taking a path of where the save folder should be created.
/// The computed [`Codomain`] are always saved by default.
///
/// # Attribute
///
/// * `path` : `&'static` [`str`]  - The path to where the files should be created.
/// * `obj` : bool - If `true` computed `Obj` [`Solution`] will be saved.
/// * `opt` : bool - If `true` computed `Opt` [`Solution`] will be saved.
/// * `info` : bool - If `true` [`SolInfo`] and [`OptInfo`] from computed [`Solution`] will be saved.
/// * `out` : bool - If `true` computed [`Outcome`] will be saved.
/// 
/// # Notes on File hierarchy
///
/// The 4 csv files information are linked by the unique [`Id`] of computed [`Solution`].
///
/// * `path`
///  * recorder
///   * obj.csv             (points from the [`Objective`] view)
///   * opt.csv             (points from the [`Optimizer`] view)
///   * info.csv            ([`SolInfo`] and [`OptInfo`])
///   * out.csv             ([`Outcome`])
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "SolId: Serialize",
    deserialize = "SolId: for<'a> Deserialize<'a>",
))]
pub struct CSVRecorder<Config,SolId,Obj,Opt,Out,Scp,Op>
where
    Config: SaverConfig<Op::Sol,Scp,SolId,Obj,Opt,Op::SInfo,Op::Cod,Out>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out:Outcome,
    Op: Optimizer<SolId,Obj,Opt,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Obj,Opt,Op::SInfo>,
{
    pub config: Arc<Config>,
    pub obj: bool,
    pub opt: bool,
    pub info: bool,
    pub out: bool,
    path_pobj: Option<PathBuf>,
    path_popt: Option<PathBuf>,
    path_info: Option<PathBuf>,
    path_codom: PathBuf,
    path_out: Option<PathBuf>,
    p_solid: PhantomData<SolId>,
    p_obj: PhantomData<Obj>,
    p_opt: PhantomData<Opt>,
    p_out: PhantomData<Out>,
    p_scp: PhantomData<Scp>,
    p_op: PhantomData<Op>,
}

impl<SolId,Obj,Opt,Out,Scp,Op> CSVRecorder<FolderConfig<Op::Sol,Scp,SolId,Obj,Opt,Op::SInfo,Op::Cod,Out>,SolId,Obj,Opt,Out,Scp,Op>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out:Outcome,
    Op: Optimizer<SolId,Obj,Opt,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Obj,Opt,Op::SInfo>
{
    pub fn new(config: Arc<FolderConfig<Op::Sol,Scp,SolId,Obj,Opt,Op::SInfo,Op::Cod,Out>>,obj: bool,opt: bool,info: bool,out: bool) -> Self
    {
        let path_pobj = match obj {
            true => Some(config.path_rec.join(Path::new("obj.csv"))),
            false => None,
        };
        let path_popt = match opt {
            true => Some(config.path_rec.join(Path::new("opt.csv"))),
            false => None,
        };
        let path_info = match info {
            true => Some(config.path_rec.join(Path::new("info.csv"))),
            false => None,
        };
        let path_out = match out {
            true => Some(config.path_rec.join(Path::new("out.csv"))),
            false => None,
        };

        let path_codom = config.path_rec.join(Path::new("cod.csv"));

        CSVRecorder {
            config,
            obj,
            opt,
            info,
            out,
            path_pobj,
            path_popt,
            path_info,
            path_codom,
            path_out,
            p_solid: PhantomData,
            p_obj: PhantomData,
            p_opt: PhantomData,
            p_out: PhantomData,
            p_scp: PhantomData,
            p_op: PhantomData,
        }
    }
}

impl<SolId, Obj, Opt, Out, Scp, Op> Recorder<SolId, Obj, Opt, Out, Scp, Op>
    for CSVRecorder<FolderConfig<Op::Sol, Scp, SolId, Obj, Opt, Op::SInfo, Op::Cod, Out>,SolId,Obj,Opt,Out,Scp,Op>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo> + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    Op::Cod: Codomain<Out> + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::BType: BatchCSVWrite<Op::Sol, Scp, SolId, Obj, Opt, Op::SInfo, Op::Info, Op::Cod, Out>,
{
    type Config = FolderConfig<Op::Sol, Scp, SolId, Obj, Opt, Op::SInfo, Op::Cod, Out>;

    fn get_config(&self)->Arc<Self::Config> {
        self.config.clone()
    }

    fn init(&mut self) {
        let does_exist = self.config.path_rec.try_exists().unwrap();

        let sp = self.config.scp;
        let cod = self.config.cod;

        if does_exist {
            panic!("The recorder folder path already exists, {}.", self.config.path_rec.display())
        } else if self.path.is_file() {
            panic!("The recorder path cant point to a file, {}.",self.config.path_rec.display())
        } else {
            create_dir_all(self.config.path_rec).unwrap();

            if let Some(ppobj) = &self.path_pobj {
                let mut wrt = csv::Writer::from_path(ppobj.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(sp));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }

            if let Some(ppopt) = &self.path_popt {
                let mut wrt = csv::Writer::from_path(ppopt.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(sp));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }

            if let Some(ppinfo) = &self.path_info {
                let mut wrt = csv::Writer::from_path(ppinfo.as_path()).unwrap();
                let mut idstr = SolId::header(&());
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
                idstr.extend(Op::Cod::header(cod));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }
        }
    }

    fn after_load(&mut self) {
        // Check if all folder and files exist
        if self.config.path_rec.try_exists().unwrap() {
            if let Some(ppobj) = &self.path_pobj {
                if !ppobj.try_exists().unwrap() {
                    panic!(
                        "The `Objective` recorder file does not exists, {}.",
                        ppobj.display()
                    )
                }
            }

            if let Some(ppopt) = &self.path_popt {
                if !ppopt.try_exists().unwrap() {
                    panic!(
                        "The `Optimizer` recorder file  not exists, {}.",
                        ppopt.display()
                    )
                }
            }

            if let Some(ppinfo) = &self.path_info {
                if !ppinfo.try_exists().unwrap() {
                    panic!(
                        "The `Info` file does not exists, {}.",
                        ppinfo.display()
                    )
                }
            }

            if let Some(ppout) = &self.path_out {
                if !ppout.try_exists().unwrap() {
                    panic!(
                        "The `Output` recorder file does not exists, {}.",
                        ppout.display()
                    )
                }
            }

            if !self.path_codom.try_exists().unwrap() {
                panic!(
                    "The `Codomain` recorder file does not exists, {}.",
                    self.path_codom.display()
                )
            }
        } else {
            panic!("The recorder folder does not exists, {}.", self.config.path_rec.display());
        }
    }

    fn save_partial(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>) {
        if let Some(ppobj) = &self.path_pobj {
            let file = OpenOptions::new()
                .append(true)
                .open(ppobj.as_path())
                .unwrap();
            Op::BType::write_partial_obj_with_comp(
                batch,
                csv::Writer::from_writer(file),
                self.config.sp,
            );
        }
        if let Some(ppopt) = &self.path_popt {
            let file = OpenOptions::new()
                .append(true)
                .open(ppopt.as_path())
                .unwrap();
            Op::BType::write_partial_opt_with_comp(
                batch,
                csv::Writer::from_writer(file),
                self.config.sp,
            );
        }
    }

    fn save_info(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>) {
        if let Some(ppinfo) = &self.path_info {
            let file = OpenOptions::new()
                .append(true)
                .open(ppinfo.as_path())
                .unwrap();
            Op::BType::write_info_with_comp(batch, csv::Writer::from_writer(file));
        }
    }

    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>) {
        let file = OpenOptions::new()
            .append(true)
            .open(self.path_codom.as_path())
            .unwrap();
        Op::BType::write_codom(batch, csv::Writer::from_writer(file), self.config.cod.clone());
    }

    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>) {
        if let Some(ppout) = &self.path_out {
            let file = OpenOptions::new()
                .append(true)
                .open(ppout.as_path())
                .unwrap();
            Op::BType::write_out(batch, csv::Writer::from_writer(file));
        }
    }
}

/// Version of [`CSVSaver`] for MPI-distributed algorithms.
/// The computed [`Codomain`] are always saved by default.
///
/// # Attribute
///
/// * `path` : `&'static` [`str`]  - The path to where the folder should be created.
///   Creates all parents folder that might not  exist yet.
/// * `save_obj` : bool - If `true` computed `Obj` [`Solution`] will be saved.
/// * `save_opt` : bool - If `true` computed `Opt` [`Solution`] will be saved.
/// * `save_info` : bool - If `true` [`SolInfo`] and [`OptInfo`] from computed [`Solution`] will be saved.
/// * `save_out` : bool - If `true` computed [`Outcome`] will be saved.
/// * `checkpoint` : usize - If `>0`, a checkpoint will be created every `checkpoint` call to [`step`](Optimizer::step).
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
///   * state_param.mp    (Various global parameters as the [`Id`] or experiment identifier.)
///
#[cfg(feature = "mpi")]
impl<SolId, Obj, Opt, Out, Scp, Op> DistRecorder<SolId, Obj, Opt, Out, Scp, Op>
    for CSVRecorder<FolderConfig<Op::Sol, Scp, SolId, Obj, Opt, Op::SInfo, Op::Cod, Out>,SolId,Obj,Opt,Out,Scp,Op>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo> + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp>,
    Op::Cod: Codomain<Out> + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::BType: BatchCSVWrite<Op::Sol, Scp, SolId, Obj, Opt, Op::SInfo, Op::Info, Op::Cod, Out>,
{   
    fn get_config(&self, rank:Rank)->Arc<Self::Config> {
        self.config.clone()
    }

    fn init(&mut self, rank:Rank) {
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
                idstr.extend(Scp::header(self.config.sp));
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
                idstr.extend(Scp::header(self.config.sp));
                idstr.extend(Op::SInfo::header(&()));
                idstr.extend(Op::Info::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
                self.path_popt.replace(nppopt);
            }

            if self.path_info.is_some() {
                let nppout = path_evals.join(Path::new("info.csv"));
                let mut wrt = csv::Writer::from_path(nppout.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Out::header(&()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
                self.path_out.replace(nppout);
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
                idstr.extend(Op::Cod::header(self.config.cod));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
                self.path_codom = nppcod;
            }

            if self.path_check.is_some() {
                let path = self.path.join(Path::new(&format!("checkpoint_rk{}", rank)));
                let pste = path.join(Path::new("state_opt.mp"));
                let pstp = path.join(Path::new("state_stp.mp"));
                let peva = path.join(Path::new("state_eval.mp"));
                let ppar = path.join(Path::new("state_param.mp"));
                self.path_check.replace((pste, pstp, peva, ppar));
                create_dir(path.as_path()).unwrap();
            }
        }
    }

    fn after_load(&mut self, rank:Rank) {
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

            if self.path_info.is_some() {
                let nppinfo = path_evals.join(Path::new("info.csv"));
                if !nppinfo.try_exists().unwrap() {
                    panic!(
                        "The file path for Output does not exists, {}.",
                        nppinfo.display()
                    )
                }
                self.path_out = Some(nppinfo);
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
                let pste = path.join(Path::new("state_opt.mp"));
                let pstp = path.join(Path::new("state_stp.mp"));
                let peva = path.join(Path::new("state_eval.mp"));
                let ppar = path.join(Path::new("state_param.mp"));
                self.path_check.replace((pste, pstp, peva, ppar));
            }
        } else {
            panic!("The folder path does not exists, {}.", self.path.display())
        }
    }

    fn save_partial(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank) {
        if let Some(ppobj) = &self.path_pobj {
            let file = OpenOptions::new()
                .append(true)
                .open(ppobj.as_path())
                .unwrap();
            Op::BType::write_partial_obj_with_comp(
                batch,
                csv::Writer::from_writer(file),
                self.config.sp,
            );
        }
        if let Some(ppopt) = &self.path_popt {
            let file = OpenOptions::new()
                .append(true)
                .open(ppopt.as_path())
                .unwrap();
            Op::BType::write_partial_opt_with_comp(
                batch,
                csv::Writer::from_writer(file),
                self.config.sp,
            );
        }
    }

    fn save_info(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank) {
        if let Some(ppinfo) = &self.path_info {
            let file = OpenOptions::new()
                .append(true)
                .open(ppinfo.as_path())
                .unwrap();
            Op::BType::write_info_with_comp(batch, csv::Writer::from_writer(file));
        }
    }

    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank) {
        let file = OpenOptions::new()
            .append(true)
            .open(self.path_codom.as_path())
            .unwrap();
        Op::BType::write_codom(batch, csv::Writer::from_writer(file), self.config.cod);
    }

    fn save_out(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>, rank:Rank) {
        if let Some(ppout) = &self.path_out {
            let file = OpenOptions::new()
                .append(true)
                .open(ppout.as_path())
                .unwrap();
            Op::BType::write_out(batch, csv::Writer::from_writer(file));
        }
    }
}
