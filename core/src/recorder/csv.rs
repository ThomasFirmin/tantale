use crate::{
    domain::{onto::LinkOpt, Codomain, TypeDom},
    objective::{Outcome, Step},
    optimizer::Optimizer,
    recorder::Recorder,
    searchspace::{CompShape, Searchspace},
    solution::{
        shape::{SolObj, SolOpt},
        Batch, HasFidelity, HasId, HasInfo, HasSolInfo, HasStep, HasUncomputed, HasY, Id, OutBatch,
        Solution, SolutionShape, Uncomputed,
    },
    BasePartial, FidBasePartial, Fidelity, FolderConfig, OptInfo, SolInfo,
};

use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::{
    fs::{create_dir_all, File, OpenOptions},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// A structure containing all [`Writer`](csv::Writer) for a [`CSVRecorder`].
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

#[cfg(feature = "mpi")]
use crate::{experiment::mpi::utils::MPIProcess, recorder::DistRecorder};

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

pub trait SolCSVWrite<PartOpt, SolId, SInfo>: Searchspace<PartOpt, SolId, SInfo>
where
    PartOpt: Uncomputed<SolId, Self::Opt, SInfo>,
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
{
    fn header_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>);
    fn header_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>);
    fn write_partial_obj(
        &self,
        id: &[String],
        sol: &SolObj<Self::SolShape, SolId, SInfo>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
    fn write_partial_opt(
        &self,
        id: &[String],
        sol: &SolOpt<Self::SolShape, SolId, SInfo>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
}

pub trait CodCSVWrite<SolId, Out>: Codomain<Out>
where
    Self: Sized + CSVWritable<Self, Self::TypeCodom>,
    SolId: Id + CSVWritable<(), ()>,
    Out: Outcome,
{
    fn header_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>);
    fn write_codom(
        &self,
        id: &[String],
        codom: Arc<Self::TypeCodom>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
}

pub trait InfoCSVWrite<SolId, SInfo, Info>: SolutionShape<SolId, SInfo>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo + CSVWritable<(), ()>,
{
    fn header_info(wrt: Arc<Mutex<csv::Writer<File>>>);
    fn write_info(&self, id: &[String], info: Arc<Info>, wrt: Arc<Mutex<csv::Writer<File>>>);
}

pub trait OutCSVWrite<SolId: Id>: Outcome {
    fn header_out(wrt: Arc<Mutex<csv::Writer<File>>>);
    fn write_out(&self, id: &[String], wrt: Arc<Mutex<csv::Writer<File>>>);
}

/// Describes how to write a [`SolutionShape`] within a csv file.
pub trait ScpCSVWrite<PartOpt, SolId, SInfo, Info, Cod, Out>:
    Searchspace<PartOpt, SolId, SInfo>
where
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
    Info: OptInfo + CSVWritable<(), ()>,
    Out: Outcome + CSVWritable<(), ()>,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    PartOpt: Uncomputed<SolId, Self::Opt, SInfo>,
    CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
    SolObj<CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Self::Obj, SInfo, Uncomputed = SolObj<Self::SolShape, SolId, SInfo>>,
    SolOpt<CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Self::Opt, SInfo, Uncomputed = SolOpt<Self::SolShape, SolId, SInfo>>,
{
    fn write(
        &self,
        pair: &CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>,
        opair: &(SolId, Out),
        info: Arc<Info>,
        cod: &Cod,
        wrts: &CSVFiles,
    );
}

impl<Scp, SolId, SInfo> SolCSVWrite<BasePartial<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo> for Scp
where
    Scp: Searchspace<BasePartial<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo>
        + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
{
    fn header_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(self));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(self));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_partial_obj(
        &self,
        id: &[String],
        sol: &SolObj<Self::SolShape, SolId, SInfo>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    ) {
        let solstr = self.write_left(&sol.get_x());
        let idstr = id.iter().chain(solstr.iter());
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }
    fn write_partial_opt(
        &self,
        id: &[String],
        sol: &SolOpt<Self::SolShape, SolId, SInfo>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    ) {
        let solstr = self.write_right(&sol.get_x());
        let idstr = id.iter().chain(solstr.iter());
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }
}

impl<Scp, SolId, SInfo> SolCSVWrite<FidBasePartial<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo>
    for Scp
where
    Scp: Searchspace<FidBasePartial<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo>
        + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
{
    fn header_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        let stepstr = Step::header(&());
        let fidstr = Fidelity::header(&());
        idstr.extend(Scp::header(self));
        idstr.extend(stepstr);
        idstr.extend(fidstr);
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn header_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        let stepstr = Step::header(&());
        let fidstr = Fidelity::header(&());
        idstr.extend(Scp::header(self));
        idstr.extend(stepstr);
        idstr.extend(fidstr);
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_partial_obj(
        &self,
        id: &[String],
        sol: &SolObj<Self::SolShape, SolId, SInfo>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    ) {
        let solstr = self.write_left(&sol.get_x());
        let stepstr = &sol.step().write(&());
        let fidstr = &sol.fidelity().write(&());
        let idstr = id.iter().chain(solstr.iter()).chain(stepstr).chain(fidstr);
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }
    fn write_partial_opt(
        &self,
        id: &[String],
        sol: &SolOpt<Self::SolShape, SolId, SInfo>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    ) {
        let solstr = self.write_right(&sol.get_x());
        let stepstr = &sol.step().write(&());
        let fidstr = &sol.fidelity().write(&());
        let idstr = id.iter().chain(solstr.iter()).chain(stepstr).chain(fidstr);
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }
}

impl<SolId, Cod, Out> CodCSVWrite<SolId, Out> for Cod
where
    Self: Sized + CSVWritable<Self, Self::TypeCodom>,
    SolId: Id + CSVWritable<(), ()>,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    fn header_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Cod::header(self));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }
    fn write_codom(
        &self,
        id: &[String],
        codom: Arc<Self::TypeCodom>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    ) {
        let codstr = self.write(&codom);
        let idstr = id.iter().chain(codstr.iter());
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }
}

impl<Shape, SolId, SInfo, Info> InfoCSVWrite<SolId, SInfo, Info> for Shape
where
    Shape: SolutionShape<SolId, SInfo> + HasSolInfo<SInfo>,
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
    Info: OptInfo + CSVWritable<(), ()>,
{
    fn header_info(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_info(&self, id: &[String], info: Arc<Info>, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let sinfostr = self.get_sinfo().write(&());
        let infostr = info.write(&());
        let idstr = id.iter().chain(sinfostr.iter()).chain(infostr.iter());
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }
}

impl<Out, SolId> OutCSVWrite<SolId> for Out
where
    SolId: Id + CSVWritable<(), ()>,
    Out: Outcome + CSVWritable<(), ()>,
{
    fn header_out(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Out::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    fn write_out(&self, id: &[String], wrt: Arc<Mutex<csv::Writer<File>>>) {
        let outstr = self.write(&());
        let idstr = id.iter().chain(outstr.iter());
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }
}

impl<Scp, PartOpt, SolId, SInfo, Info, Cod, Out> ScpCSVWrite<PartOpt, SolId, SInfo, Info, Cod, Out>
    for Scp
where
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
    Info: OptInfo + CSVWritable<(), ()>,
    Out: Outcome + OutCSVWrite<SolId> + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Cod::TypeCodom: Send + Sync,
    PartOpt: Uncomputed<SolId, Self::Opt, SInfo>,
    Scp: SolCSVWrite<PartOpt, SolId, SInfo>,
    CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>: InfoCSVWrite<SolId, SInfo, Info>,
    SolObj<CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Self::Obj, SInfo, Uncomputed = SolObj<Self::SolShape, SolId, SInfo>>,
    SolOpt<CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Self::Opt, SInfo, Uncomputed = SolOpt<Self::SolShape, SolId, SInfo>>,
{
    fn write(
        &self,
        pair: &CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>,
        opair: &(SolId, Out),
        info: Arc<Info>,
        cod: &Cod,
        wrts: &CSVFiles,
    ) {
        let id = pair.get_id();
        let idstr = id.write(&());

        // CODOM
        <Cod as CodCSVWrite<SolId, Out>>::write_codom(cod, &idstr, pair.get_y(), wrts.cod.clone());
        // OBJ
        if let Some(f) = wrts.obj.clone() {
            self.write_partial_obj(&idstr, pair.get_sobj().get_uncomputed(), f);
        }
        // OPT
        if let Some(f) = wrts.opt.clone() {
            self.write_partial_opt(&idstr, pair.get_sopt().get_uncomputed(), f);
        }
        // INFO
        if let Some(f) = wrts.info.clone() {
            pair.write_info(&idstr, info, f);
        }
        // OUT
        if let Some(f) = wrts.out.clone() {
            opair.1.write_out(&idstr, f);
        }
    }
}

/// Describes how to write a [`Batch`] within a csv file.
pub trait BatchCSVWrite<PartOpt, Scp, SolId, SInfo, Cod, Out, Info>
where
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
    Info: OptInfo + CSVWritable<(), ()>,
    Out: Outcome + CSVWritable<(), ()>,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    PartOpt: Uncomputed<SolId, Scp::Opt, SInfo>,
    Scp: Searchspace<PartOpt, SolId, SInfo>,
    CompShape<Scp, PartOpt, SolId, SInfo, Cod, Out>: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
    SolObj<CompShape<Scp, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Scp::Obj, SInfo, Uncomputed = SolObj<Scp::SolShape, SolId, SInfo>>,
    SolOpt<CompShape<Scp, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Scp::Opt, SInfo, Uncomputed = SolOpt<Scp::SolShape, SolId, SInfo>>,
{
    fn write(&self, obatch: &OutBatch<SolId, Info, Out>, scp: &Scp, cod: &Cod, wrts: Arc<CSVFiles>);
}

impl<PartOpt, Scp, SolId, SInfo, Cod, Out, Info>
    BatchCSVWrite<PartOpt, Scp, SolId, SInfo, Cod, Out, Info>
    for Batch<SolId, SInfo, Info, CompShape<Scp, PartOpt, SolId, SInfo, Cod, Out>>
where
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    SInfo: SolInfo + CSVWritable<(), ()>,
    Info: OptInfo + CSVWritable<(), ()> + Send + Sync,
    Out: Outcome + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom> + Send + Sync,
    PartOpt: Uncomputed<SolId, Scp::Opt, SInfo>,
    Scp: Searchspace<PartOpt, SolId, SInfo>
        + ScpCSVWrite<PartOpt, SolId, SInfo, Info, Cod, Out>
        + Send
        + Sync,
    CompShape<Scp, PartOpt, SolId, SInfo, Cod, Out>:
        SolutionShape<SolId, SInfo> + HasY<Cod, Out> + Send + Sync,
    SolObj<CompShape<Scp, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Scp::Obj, SInfo, Uncomputed = SolObj<Scp::SolShape, SolId, SInfo>>,
    SolOpt<CompShape<Scp, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Scp::Opt, SInfo, Uncomputed = SolOpt<Scp::SolShape, SolId, SInfo>>,
{
    fn write(
        &self,
        obatch: &OutBatch<SolId, Info, Out>,
        scp: &Scp,
        cod: &Cod,
        wrts: Arc<CSVFiles>,
    ) {
        let info = self.get_info();
        self.into_par_iter()
            .zip(obatch)
            .for_each(|(cpair, opair)| scp.write(cpair, opair, info.clone(), cod, &wrts.clone()));
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

impl<PSol, SolId, Out, Scp, Op> Recorder<PSol, SolId, Out, Scp, Op> for CSVRecorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: OutCSVWrite<SolId> + CSVWritable<(), ()> + Send + Sync,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::Cod: CodCSVWrite<SolId, Out>
        + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>
        + Send
        + Sync,
    Scp: SolCSVWrite<PSol, SolId, Op::SInfo>
        + ScpCSVWrite<PSol, SolId, Op::SInfo, Op::Info, Op::Cod, Out>
        + Send
        + Sync,
    Scp::SolShape: InfoCSVWrite<SolId, Op::SInfo, Op::Info>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        InfoCSVWrite<SolId, Op::SInfo, Op::Info> + HasY<Op::Cod, Out> + Send + Sync,
    SolObj<CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>, SolId, Op::SInfo>: HasUncomputed<
        SolId,
        Scp::Obj,
        Op::SInfo,
        Uncomputed = SolObj<Scp::SolShape, SolId, Op::SInfo>,
    >,
    SolOpt<CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>, SolId, Op::SInfo>: HasUncomputed<
        SolId,
        Scp::Opt,
        Op::SInfo,
        Uncomputed = SolOpt<Scp::SolShape, SolId, Op::SInfo>,
    >,
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
            if let Some(f) = &self.path_pobj {
                File::create_new(f).unwrap();
            }

            if let Some(f) = &self.path_popt {
                File::create_new(f).unwrap();
            }

            if let Some(f) = &self.path_info {
                File::create_new(f).unwrap();
            }

            if let Some(f) = &self.path_out {
                File::create_new(f).unwrap();
            }
            File::create_new(&self.path_codom).unwrap();

            let files = CSVFiles::new(
                &self.path_pobj,
                &self.path_popt,
                &self.path_codom,
                &self.path_info,
                &self.path_out,
            );

            if let Some(f) = files.obj.clone() {
                scp.header_partial_obj(f);
            }

            if let Some(f) = files.opt.clone() {
                scp.header_partial_opt(f);
            }

            if let Some(f) = files.info.clone() {
                Scp::SolShape::header_info(f);
            }

            if let Some(f) = files.out.clone() {
                Out::header_out(f);
            }

            cod.header_codom(files.cod.clone());
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

    fn save_batch(
        &self,
        computed: &Batch<
            SolId,
            Op::SInfo,
            Op::Info,
            CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        >,
        outputed: &OutBatch<SolId, Op::Info, Out>,
        scp: &Scp,
        cod: &Op::Cod,
    ) {
        let files = Arc::new(CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        ));
        BatchCSVWrite::<PSol, Scp, SolId, Op::SInfo, Op::Cod, Out, Op::Info>::write(
            computed, outputed, scp, cod, files,
        );
    }

    fn save_pair(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
        info: Arc<Op::Info>,
    ) {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );
        scp.write(computed, outputed, info, cod, &files);
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
impl<PSol, SolId, Out, Scp, Op> DistRecorder<PSol, SolId, Out, Scp, Op> for CSVRecorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: OutCSVWrite<SolId> + CSVWritable<(), ()> + Send + Sync,
    Op: Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::Cod: CodCSVWrite<SolId, Out>
        + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>
        + Send
        + Sync,
    Scp: SolCSVWrite<PSol, SolId, Op::SInfo>
        + ScpCSVWrite<PSol, SolId, Op::SInfo, Op::Info, Op::Cod, Out>
        + Send
        + Sync,
    Scp::SolShape: InfoCSVWrite<SolId, Op::SInfo, Op::Info>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        InfoCSVWrite<SolId, Op::SInfo, Op::Info> + HasY<Op::Cod, Out> + Send + Sync,
    SolObj<CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>, SolId, Op::SInfo>: HasUncomputed<
        SolId,
        Scp::Obj,
        Op::SInfo,
        Uncomputed = SolObj<Scp::SolShape, SolId, Op::SInfo>,
    >,
    SolOpt<CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>, SolId, Op::SInfo>: HasUncomputed<
        SolId,
        Scp::Opt,
        Op::SInfo,
        Uncomputed = SolOpt<Scp::SolShape, SolId, Op::SInfo>,
    >,
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
                if let Some(f) = &self.path_pobj {
                    File::create_new(f).unwrap();
                }

                if let Some(f) = &self.path_popt {
                    File::create_new(f).unwrap();
                }

                if let Some(f) = &self.path_info {
                    File::create_new(f).unwrap();
                }

                if let Some(f) = &self.path_out {
                    File::create_new(f).unwrap();
                }

                File::create_new(&self.path_codom).unwrap();

                let files = CSVFiles::new(
                    &self.path_pobj,
                    &self.path_popt,
                    &self.path_codom,
                    &self.path_info,
                    &self.path_out,
                );

                if let Some(f) = files.obj.clone() {
                    scp.header_partial_obj(f);
                }

                if let Some(f) = files.opt.clone() {
                    scp.header_partial_opt(f);
                }

                if let Some(f) = files.info.clone() {
                    Scp::SolShape::header_info(f);
                }

                if let Some(f) = files.out.clone() {
                    Out::header_out(f);
                }

                cod.header_codom(files.cod.clone());
            }
        } else {
            panic!("The FolderConfig should be set for Distribued environment.")
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

    fn save_batch_dist(
        &self,
        computed: &Batch<
            SolId,
            Op::SInfo,
            Op::Info,
            CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        >,
        outputed: &OutBatch<SolId, Op::Info, Out>,
        scp: &Scp,
        cod: &Op::Cod,
    ) {
        let files = Arc::new(CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        ));
        BatchCSVWrite::<PSol, Scp, SolId, Op::SInfo, Op::Cod, Out, Op::Info>::write(
            computed, outputed, scp, cod, files,
        );
    }

    fn save_pair_dist(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
        info: Arc<Op::Info>,
    ) {
        let files = CSVFiles::new(
            &self.path_pobj,
            &self.path_popt,
            &self.path_codom,
            &self.path_info,
            &self.path_out,
        );
        scp.write(computed, outputed, info, cod, &files);
    }
}
