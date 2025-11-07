use crate::{
    FolderConfig, OptInfo, Partial, SolInfo,
    domain::onto::OntoDom,
    objective::{Codomain, Outcome},
    optimizer::{Optimizer, opt::{CBType, OBType}},
    recorder::Recorder,
    searchspace::Searchspace,
    solution::{Batch, BatchType, Id, Solution, partial::BasePartial},
};

use rayon::prelude::*;
use std::{
    fs::{File, OpenOptions, create_dir_all},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{Arc, Mutex}
};

#[cfg(feature = "mpi")]
use crate::{experiment::mpi::tools::MPIProcess, recorder::DistRecorder};

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
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom> + Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<BasePartial<SolId, Obj, SInfo>, SolId, Obj, Opt, SInfo> + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>> + Send + Sync,
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
pub struct CSVRecorder<SolId,Obj,Opt,Cod,Out,Scp,Op>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out:Outcome,
    Op: Optimizer<SolId,Obj,Opt,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Obj,Opt,Op::SInfo>,
{
    pub config: Arc<FolderConfig>,
    pub scp: Arc<Scp>,
    pub cod: Arc<Cod>,
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

impl<SolId,Obj,Opt,Cod,Out,Scp,Op> CSVRecorder<SolId,Obj,Opt,Cod,Out,Scp,Op>
where
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out:Outcome,
    Op: Optimizer<SolId,Obj,Opt,Out,Scp>,
    Scp: Searchspace<Op::Sol,SolId,Obj,Opt,Op::SInfo>,
{
    pub fn new(config: Arc<FolderConfig>, scp: Arc<Scp>, cod: Arc<Cod>, obj: bool, opt: bool,info: bool,out: bool) -> Self
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
            scp,
            cod,
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
    for CSVRecorder<SolId,Obj,Opt,Op::Cod,Out,Scp,Op>
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
    fn init(&mut self) {
        let does_exist = self.config.path_rec.try_exists().unwrap();
        if does_exist {
            panic!("The recorder folder path already exists, {}.", self.config.path_rec.display())
        } else if self.config.path.is_file() {
            panic!("The recorder path cant point to a file, {}.",self.config.path_rec.display())
        } else {
            create_dir_all(&self.config.path_rec).unwrap();

            if let Some(ppobj) = &self.path_pobj {
                let mut wrt = csv::Writer::from_path(ppobj.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(self.scp.as_ref()));
                wrt.write_record(idstr).unwrap();
                wrt.flush().unwrap();
            }

            if let Some(ppopt) = &self.path_popt {
                let mut wrt = csv::Writer::from_path(ppopt.as_path()).unwrap();
                let mut idstr = SolId::header(&());
                idstr.extend(Scp::header(self.scp.as_ref()));
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
                idstr.extend(Op::Cod::header(self.cod.as_ref()));
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
            Op::BType::write_partial_obj(
                batch,
                csv::Writer::from_writer(file),
                self.scp.clone(),
            );
        }
        if let Some(ppopt) = &self.path_popt {
            let file = OpenOptions::new()
                .append(true)
                .open(ppopt.as_path())
                .unwrap();
            Op::BType::write_partial_opt(
                batch,
                csv::Writer::from_writer(file),
                self.scp.clone(),
            );
        }
    }

    fn save_info(&self, batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>) {
        if let Some(ppinfo) = &self.path_info {
            let file = OpenOptions::new()
                .append(true)
                .open(ppinfo.as_path())
                .unwrap();
            Op::BType::write_info(batch, csv::Writer::from_writer(file));
        }
    }

    fn save_codom(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>) {
        let file = OpenOptions::new()
            .append(true)
            .open(self.path_codom.as_path())
            .unwrap();
        Op::BType::write_codom(batch, csv::Writer::from_writer(file), self.cod.clone());
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
    for CSVRecorder<SolId,Obj,Opt,Op::Cod,Out,Scp,Op>
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
    fn init_dist(&mut self, _proc:&MPIProcess) {
        if self.config.is_dist{
            let does_exist = self.config.path_rec.try_exists().unwrap();
            if does_exist {
                panic!("The recorder folder path already exists, {}.", self.config.path_rec.display())
            } else if self.config.path.is_file() {
                panic!("The recorder path cant point to a file, {}.",self.config.path_rec.display())
            } else {
                create_dir_all(&self.config.path_rec).unwrap();
    
                if let Some(ppobj) = &self.path_pobj {
                    let mut wrt = csv::Writer::from_path(ppobj.as_path()).unwrap();
                    let mut idstr = SolId::header(&());
                    idstr.extend(Scp::header(self.scp.as_ref()));
                    wrt.write_record(idstr).unwrap();
                    wrt.flush().unwrap();
                }
    
                if let Some(ppopt) = &self.path_popt {
                    let mut wrt = csv::Writer::from_path(ppopt.as_path()).unwrap();
                    let mut idstr = SolId::header(&());
                    idstr.extend(Scp::header(self.scp.as_ref()));
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
                    idstr.extend(Op::Cod::header(self.cod.as_ref()));
                    wrt.write_record(idstr).unwrap();
                    wrt.flush().unwrap();
                }
            }
        } else {panic!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.")}
    }

    fn after_load_dist(&mut self, _proc:&MPIProcess) {
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

    fn save_partial_dist(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>) {
        if let Some(ppobj) = &self.path_pobj {
            let file = OpenOptions::new()
                .append(true)
                .open(ppobj.as_path())
                .unwrap();
            Op::BType::write_partial_obj(
                batch,
                csv::Writer::from_writer(file),
                self.scp.clone(),
            );
        }
        if let Some(ppopt) = &self.path_popt {
            let file = OpenOptions::new()
                .append(true)
                .open(ppopt.as_path())
                .unwrap();
            Op::BType::write_partial_opt(
                batch,
                csv::Writer::from_writer(file),
                self.scp.clone(),
            );
        }
    }

    fn save_info_dist(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>) {
        if let Some(ppinfo) = &self.path_info {
            let file = OpenOptions::new()
                .append(true)
                .open(ppinfo.as_path())
                .unwrap();
            Op::BType::write_info(batch, csv::Writer::from_writer(file));
        }
    }

    fn save_codom_dist(&self,batch: &CBType<Op, SolId, Obj, Opt, Out, Scp>) {
        let file = OpenOptions::new()
            .append(true)
            .open(self.path_codom.as_path())
            .unwrap();
        Op::BType::write_codom(batch, csv::Writer::from_writer(file), self.cod.clone());
    }

    fn save_out_dist(&self, batch: &OBType<Op, SolId, Obj, Opt, Out, Scp>) {
        if let Some(ppout) = &self.path_out {
            let file = OpenOptions::new()
                .append(true)
                .open(ppout.as_path())
                .unwrap();
            Op::BType::write_out(batch, csv::Writer::from_writer(file));
        }
    }
}
