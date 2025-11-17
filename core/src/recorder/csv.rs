use crate::{
    domain::onto::OntoDom,
    objective::{Codomain, FuncWrapper, Outcome},
    optimizer::Optimizer,
    recorder::Recorder,
    searchspace::Searchspace,
    solution::{partial::BasePartial, Batch, BatchType, Id, Solution},
    FidBasePartial, FolderConfig, OptInfo, Partial, SolInfo,
};

use rayon::prelude::*;
use std::{
    fs::{create_dir_all, File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
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

pub struct CSVFiles {
    pub obj: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub opt: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub cod: Arc<Mutex<csv::Writer<File>>>,
    pub info: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub out: Option<Arc<Mutex<csv::Writer<File>>>>,
}

impl CSVFiles {
    pub fn new(
        obj: &Option<PathBuf>,
        opt: &Option<PathBuf>,
        cod: &PathBuf,
        info: &Option<PathBuf>,
        out: &Option<PathBuf>,
    ) -> Self {
        let obj = obj.as_ref().map(|path| {
            Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(path).unwrap(),
            )))
        });
        let opt = opt.as_ref().map(|path| {
            Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(path).unwrap(),
            )))
        });
        let cod = Arc::new(Mutex::new(csv::Writer::from_writer(
            OpenOptions::new().append(true).open(cod).unwrap(),
        )));
        let info = info.as_ref().map(|path| {
            Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(path).unwrap(),
            )))
        });
        let out = out.as_ref().map(|path| {
            Arc::new(Mutex::new(csv::Writer::from_writer(
                OpenOptions::new().append(true).open(path).unwrap(),
            )))
        });
        CSVFiles {
            obj,
            opt,
            cod,
            info,
            out,
        }
    }
}

/// A [`CSVWrite`] describes a [`BatchType`] and its associated types that can be written within a CSV file.
pub trait BatchCSVWrite<PSol, Scp, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    Self: BatchType<SolId, Obj, Opt, SInfo, PSol, Info>,
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>
        + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Info: OptInfo + CSVWritable<(), ()> + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
{
    fn header_partial_obj(wrt: csv::Writer<File>, scp: &Scp);
    fn header_partial_opt(wrt: csv::Writer<File>, scp: &Scp);
    fn header_codom(wrt: csv::Writer<File>, cod: &Cod);
    fn header_info(wrt: csv::Writer<File>);
    fn header_out(wrt: csv::Writer<File>);
    fn write(
        cbatch: &Self::Comp<Cod, Out>,
        obatch: &Self::Outc<Out>,
        wrts: CSVFiles,
        scp: &Scp,
        cod: &Cod,
    );
    fn write_partial_obj(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, scp: &Scp);
    fn write_partial_opt(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, scp: &Scp);
    fn write_codom(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, cod: &Cod);
    fn write_info(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>);
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
    Scp: Searchspace<BasePartial<SolId, Obj, SInfo>, SolId, Obj, Opt, SInfo>
        + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>
        + Send
        + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Info: OptInfo + CSVWritable<(), ()> + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn header_partial_obj(mut wrt: csv::Writer<File>, scp: &Scp) {
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn header_partial_opt(mut wrt: csv::Writer<File>, scp: &Scp) {
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn header_codom(mut wrt: csv::Writer<File>, cod: &Cod) {
        let mut idstr = SolId::header(&());
        idstr.extend(Cod::header(cod));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn header_info(mut wrt: csv::Writer<File>) {
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn header_out(mut wrt: csv::Writer<File>) {
        let mut idstr = SolId::header(&());
        idstr.extend(Out::header(&()));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn write_partial_obj(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, scp: &Scp) {
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

    fn write_partial_opt(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, scp: &Scp) {
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

    fn write_codom(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, cod: &Cod) {
        let wrt = Arc::new(Mutex::new(wrt));
        cbatch.cobj.par_iter().for_each(|op| {
            let id = op.get_id();
            let codom = op.get_y();
            let mut idstr = id.write(&());
            idstr.extend(cod.write(codom));
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn write_out(obatch: &Self::Outc<Out>, wrt: csv::Writer<File>) {
        let wrt = Arc::new(Mutex::new(wrt));
        obatch.into_par_iter().for_each(|(id, out)| {
            let mut idstr = id.write(&());
            idstr.extend(out.write(&()));
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn write(
        cbatch: &Self::Comp<Cod, Out>,
        obatch: &Self::Outc<Out>,
        wrts: CSVFiles,
        scp: &Scp,
        cod: &Cod,
    ) {
        cbatch
            .into_par_iter()
            .zip_eq(obatch)
            .for_each(|((cobj, copt), (oid, out))| {
                let id = cobj.get_id();
                let idstr = id.write(&());

                // CODOM
                let codstr = cod.write(cobj.get_y());
                let fstr: Vec<&String> = idstr.iter().chain(codstr.iter()).collect();
                {
                    let mut wrt_local = wrts.cod.lock().unwrap();
                    wrt_local.write_record(&fstr).unwrap();
                    wrt_local.flush().unwrap();
                }
                // OBJ
                if let Some(f) = wrts.obj.clone() {
                    let xstr = scp.write_left(&cobj.get_x());
                    let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
                    {
                        let mut wrt_local = f.lock().unwrap();
                        wrt_local.write_record(&fstr).unwrap();
                        wrt_local.flush().unwrap();
                    }
                }
                // OPT
                if let Some(f) = wrts.opt.clone() {
                    let xstr = scp.write_right(&copt.get_x());
                    let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
                    {
                        let mut wrt_local = f.lock().unwrap();
                        wrt_local.write_record(&fstr).unwrap();
                        wrt_local.flush().unwrap();
                    }
                }
                // INFO
                if let Some(f) = wrts.info.clone() {
                    let sinfstr = cobj.get_info().write(&());
                    let infstr = cbatch.info.write(&());
                    let fstr: Vec<&String> = idstr
                        .iter()
                        .chain(sinfstr.iter())
                        .chain(infstr.iter())
                        .collect();
                    {
                        let mut wrt_local = f.lock().unwrap();
                        wrt_local.write_record(&fstr).unwrap();
                        wrt_local.flush().unwrap();
                    }
                }
                // OUT
                if let Some(f) = wrts.out.clone() {
                    let mut idstr = oid.write(&());
                    idstr.extend(out.write(&()));
                    {
                        let mut wrt_local = f.lock().unwrap();
                        wrt_local.write_record(&idstr).unwrap();
                        wrt_local.flush().unwrap();
                    }
                }
            });
    }
}

impl<Scp, SolId, Obj, Opt, SInfo, Info, Cod, Out>
    BatchCSVWrite<FidBasePartial<SolId, Obj, SInfo>, Scp, SolId, Obj, Opt, SInfo, Info, Cod, Out>
    for Batch<FidBasePartial<SolId, Obj, SInfo>, SolId, Obj, Opt, SInfo, Info>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom> + Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<FidBasePartial<SolId, Obj, SInfo>, SolId, Obj, Opt, SInfo>
        + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>
        + Send
        + Sync,
    SInfo: SolInfo + CSVWritable<(), ()> + Send + Sync,
    Info: OptInfo + CSVWritable<(), ()> + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Cod::TypeCodom: Send + Sync,
{
    fn header_partial_obj(mut wrt: csv::Writer<File>, scp: &Scp) {
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn header_partial_opt(mut wrt: csv::Writer<File>, scp: &Scp) {
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(scp));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn header_codom(mut wrt: csv::Writer<File>, cod: &Cod) {
        let mut idstr = SolId::header(&());
        idstr.extend(Cod::header(cod));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn header_info(mut wrt: csv::Writer<File>) {
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn header_out(mut wrt: csv::Writer<File>) {
        let mut idstr = SolId::header(&());
        idstr.extend(Out::header(&()));
        wrt.write_record(idstr).unwrap();
        wrt.flush().unwrap();
    }

    fn write_partial_obj(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, scp: &Scp) {
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

    fn write_partial_opt(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, scp: &Scp) {
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
            idstr.extend(op.get_sol().fid.write(&()));
            {
                let mut wrt_local = amwrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn write_codom(cbatch: &Self::Comp<Cod, Out>, wrt: csv::Writer<File>, cod: &Cod) {
        let wrt = Arc::new(Mutex::new(wrt));
        cbatch.cobj.par_iter().for_each(|op| {
            let id = op.get_id();
            let codom = op.get_y();
            let mut idstr = id.write(&());
            idstr.extend(cod.write(codom));
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn write_out(obatch: &Self::Outc<Out>, wrt: csv::Writer<File>) {
        let wrt = Arc::new(Mutex::new(wrt));
        obatch.into_par_iter().for_each(|(id, out)| {
            let mut idstr = id.write(&());
            idstr.extend(out.write(&()));
            {
                let mut wrt_local = wrt.lock().unwrap();
                wrt_local.write_record(&idstr).unwrap();
                wrt_local.flush().unwrap();
            }
        });
    }

    fn write(
        cbatch: &Self::Comp<Cod, Out>,
        obatch: &Self::Outc<Out>,
        wrts: CSVFiles,
        scp: &Scp,
        cod: &Cod,
    ) {
        cbatch
            .into_par_iter()
            .zip_eq(obatch)
            .for_each(|((cobj, copt), (oid, out))| {
                let id = cobj.get_id();
                let idstr = id.write(&());

                // CODOM
                let codstr = cod.write(cobj.get_y());
                let fstr: Vec<&String> = idstr.iter().chain(codstr.iter()).collect();
                {
                    let mut wrt_local = wrts.cod.lock().unwrap();
                    wrt_local.write_record(&fstr).unwrap();
                    wrt_local.flush().unwrap();
                }
                // OBJ
                if let Some(f) = wrts.obj.clone() {
                    let xstr = scp.write_left(&cobj.get_x());
                    let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
                    {
                        let mut wrt_local = f.lock().unwrap();
                        wrt_local.write_record(&fstr).unwrap();
                        wrt_local.flush().unwrap();
                    }
                }
                // OPT
                if let Some(f) = wrts.opt.clone() {
                    let xstr = scp.write_right(&copt.get_x());
                    let fstr: Vec<&String> = idstr.iter().chain(xstr.iter()).collect();
                    {
                        let mut wrt_local = f.lock().unwrap();
                        wrt_local.write_record(&fstr).unwrap();
                        wrt_local.flush().unwrap();
                    }
                }
                // INFO
                if let Some(f) = wrts.info.clone() {
                    let sinfstr = cobj.get_info().write(&());
                    let infstr = cbatch.info.write(&());
                    let fstr: Vec<&String> = idstr
                        .iter()
                        .chain(sinfstr.iter())
                        .chain(infstr.iter())
                        .collect();
                    {
                        let mut wrt_local = f.lock().unwrap();
                        wrt_local.write_record(&fstr).unwrap();
                        wrt_local.flush().unwrap();
                    }
                }
                // OUT
                if let Some(f) = wrts.out.clone() {
                    let mut idstr = oid.write(&());
                    idstr.extend(out.write(&()));
                    {
                        let mut wrt_local = f.lock().unwrap();
                        wrt_local.write_record(&idstr).unwrap();
                        wrt_local.flush().unwrap();
                    }
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
pub struct CSVRecorder {
    pub config: Arc<FolderConfig>,
    pub obj: bool,
    pub opt: bool,
    pub info: bool,
    pub out: bool,
    path_pobj: Option<PathBuf>,
    path_popt: Option<PathBuf>,
    path_info: Option<PathBuf>,
    path_codom: PathBuf,
    path_out: Option<PathBuf>,
}

impl CSVRecorder {
    pub fn new(
        config: Arc<FolderConfig>,
        obj: bool,
        opt: bool,
        info: bool,
        out: bool,
    ) -> Option<Self> {
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

        Some(CSVRecorder {
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
        })
    }
}

impl<SolId, Obj, Opt, Out, Scp, Op, Fn, BType> Recorder<SolId, Obj, Opt, Out, Scp, Op, Fn, BType>
    for CSVRecorder
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>
        + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn, BType = BType>,
    Fn: FuncWrapper,
    BType: BatchType<SolId, Obj, Opt, Op::SInfo, Op::Sol, Op::Info>
        + BatchCSVWrite<Op::Sol, Scp, SolId, Obj, Opt, Op::SInfo, Op::Info, Op::Cod, Out>,
    Op::Cod: Codomain<Out> + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
{
    fn init(&mut self, scp: &Scp, cod: &Op::Cod) {
        let does_exist = self.config.path_rec.try_exists().unwrap();
        if does_exist {
            panic!(
                "The recorder folder path already exists, {}.",
                self.config.path_rec.display()
            )
        } else if self.config.path.is_file() {
            panic!(
                "The recorder path cant point to a file, {}.",
                self.config.path_rec.display()
            )
        } else {
            create_dir_all(&self.config.path_rec).unwrap();

            if let Some(ppobj) = &self.path_pobj {
                BType::header_partial_obj(csv::Writer::from_path(ppobj.as_path()).unwrap(), scp);
            }

            if let Some(ppopt) = &self.path_popt {
                BType::header_partial_opt(csv::Writer::from_path(ppopt.as_path()).unwrap(), scp);
            }

            if let Some(ppinfo) = &self.path_info {
                BType::header_info(csv::Writer::from_path(ppinfo.as_path()).unwrap());
            }

            if let Some(ppout) = &self.path_out {
                BType::header_out(csv::Writer::from_path(ppout.as_path()).unwrap());
            }

            BType::header_codom(
                csv::Writer::from_path(self.path_codom.as_path()).unwrap(),
                cod,
            );
        }
    }

    fn after_load(&mut self, _scp: &Scp, _cod: &Op::Cod) {
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
                    panic!("The `Info` file does not exists, {}.", ppinfo.display())
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
            panic!(
                "The recorder folder does not exists, {}.",
                self.config.path_rec.display()
            );
        }
    }

    fn save_partial(&self, batch: &BType::Comp<Op::Cod, Out>, scp: &Scp, _cod: &Op::Cod) {
        if let Some(ppobj) = &self.path_pobj {
            let file = OpenOptions::new()
                .append(true)
                .open(ppobj.as_path())
                .unwrap();
            BType::write_partial_obj(batch, csv::Writer::from_writer(file), scp);
        }
        if let Some(ppopt) = &self.path_popt {
            let file = OpenOptions::new()
                .append(true)
                .open(ppopt.as_path())
                .unwrap();
            BType::write_partial_opt(batch, csv::Writer::from_writer(file), scp);
        }
    }

    fn save_info(&self, batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, _cod: &Op::Cod) {
        if let Some(ppinfo) = &self.path_info {
            let file = OpenOptions::new()
                .append(true)
                .open(ppinfo.as_path())
                .unwrap();
            BType::write_info(batch, csv::Writer::from_writer(file));
        }
    }

    fn save_codom(&self, batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, cod: &Op::Cod) {
        let file = OpenOptions::new()
            .append(true)
            .open(self.path_codom.as_path())
            .unwrap();
        BType::write_codom(batch, csv::Writer::from_writer(file), cod);
    }

    fn save_out(&self, batch: &BType::Outc<Out>, _scp: &Scp, _cod: &Op::Cod) {
        if let Some(ppout) = &self.path_out {
            let file = OpenOptions::new()
                .append(true)
                .open(ppout.as_path())
                .unwrap();
            BType::write_out(batch, csv::Writer::from_writer(file));
        }
    }

    fn save(
        &self,
        cbatch: &BType::Comp<Op::Cod, Out>,
        obatch: &BType::Outc<Out>,
        scp: &Scp,
        cod: &Op::Cod,
    ) {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );
        BType::write(cbatch, obatch, files, scp, cod);
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
impl<SolId, Obj, Opt, Out, Scp, Op, Fn, BType>
    DistRecorder<SolId, Obj, Opt, Out, Scp, Op, Fn, BType> for CSVRecorder
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Scp: Searchspace<Op::Sol, SolId, Obj, Opt, Op::SInfo>
        + CSVLeftRight<Scp, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>,
    Op: Optimizer<SolId, Obj, Opt, Out, Scp, Fn, BType = BType>,
    Fn: FuncWrapper,
    Op::Cod: Codomain<Out> + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>,
    <Op::Cod as Codomain<Out>>::TypeCodom: Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    BType: BatchType<SolId, Obj, Opt, Op::SInfo, Op::Sol, Op::Info>
        + BatchCSVWrite<Op::Sol, Scp, SolId, Obj, Opt, Op::SInfo, Op::Info, Op::Cod, Out>,
{
    fn init_dist(&mut self, _proc: &MPIProcess, scp: &Scp, cod: &Op::Cod) {
        if self.config.is_dist {
            let does_exist = self.config.path_rec.try_exists().unwrap();
            if does_exist {
                panic!(
                    "The recorder folder path already exists, {}.",
                    self.config.path_rec.display()
                )
            } else if self.config.path.is_file() {
                panic!(
                    "The recorder path cant point to a file, {}.",
                    self.config.path_rec.display()
                )
            } else {
                create_dir_all(&self.config.path_rec).unwrap();

                if let Some(ppobj) = &self.path_pobj {
                    let mut wrt = csv::Writer::from_path(ppobj.as_path()).unwrap();
                    let mut idstr = SolId::header(&());
                    idstr.extend(Scp::header(scp));
                    wrt.write_record(idstr).unwrap();
                    wrt.flush().unwrap();
                }

                if let Some(ppopt) = &self.path_popt {
                    let mut wrt = csv::Writer::from_path(ppopt.as_path()).unwrap();
                    let mut idstr = SolId::header(&());
                    idstr.extend(Scp::header(scp));
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
        } else {
            panic!("The FolderConfig should be set for Distribued environment. Use the `.to_dist()` method.")
        }
    }

    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {
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
                    panic!("The `Info` file does not exists, {}.", ppinfo.display())
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
            panic!(
                "The recorder folder does not exists, {}.",
                self.config.path_rec.display()
            );
        }
    }

    fn save_partial_dist(&self, batch: &BType::Comp<Op::Cod, Out>, scp: &Scp, _cod: &Op::Cod) {
        if let Some(ppobj) = &self.path_pobj {
            let file = OpenOptions::new()
                .append(true)
                .open(ppobj.as_path())
                .unwrap();
            BType::write_partial_obj(batch, csv::Writer::from_writer(file), scp);
        }
        if let Some(ppopt) = &self.path_popt {
            let file = OpenOptions::new()
                .append(true)
                .open(ppopt.as_path())
                .unwrap();
            BType::write_partial_opt(batch, csv::Writer::from_writer(file), scp);
        }
    }

    fn save_info_dist(&self, batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, _cod: &Op::Cod) {
        if let Some(ppinfo) = &self.path_info {
            let file = OpenOptions::new()
                .append(true)
                .open(ppinfo.as_path())
                .unwrap();
            BType::write_info(batch, csv::Writer::from_writer(file));
        }
    }

    fn save_codom_dist(&self, batch: &BType::Comp<Op::Cod, Out>, _scp: &Scp, cod: &Op::Cod) {
        let file = OpenOptions::new()
            .append(true)
            .open(self.path_codom.as_path())
            .unwrap();
        BType::write_codom(batch, csv::Writer::from_writer(file), cod);
    }

    fn save_out_dist(&self, batch: &BType::Outc<Out>, _scp: &Scp, _cod: &Op::Cod) {
        if let Some(ppout) = &self.path_out {
            let file = OpenOptions::new()
                .append(true)
                .open(ppout.as_path())
                .unwrap();
            BType::write_out(batch, csv::Writer::from_writer(file));
        }
    }

    fn save_dist(
        &self,
        cbatch: &BType::Comp<Op::Cod, Out>,
        obatch: &BType::Outc<Out>,
        scp: &Scp,
        cod: &Op::Cod,
    ) {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );
        BType::write(cbatch, obatch, files, scp, cod);
    }
}
