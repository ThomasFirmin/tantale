//! CSV recording utilities and recorder implementation.
//!
//! This module defines the traits required to serialize solutions, codomain
//! values, optimizer metadata, and outcomes into CSV files, plus the
//! [`CSVRecorder`] that writes them to disk.
//!
//! # File layout
//! By default, CSV files are created under the recorder folder (see
//! [`FolderConfig`](crate::FolderConfig)). Each row is linked by the solution
//! [`Id`](crate::solution::Id).
//!
//! * recorder/
//!   * obj.csv  — Objective-side inputs
//!   * opt.csv  — Optimizer-side inputs
//!   * cod.csv  — Codomain values
//!   * info.csv — [`SolInfo`](crate::solution::SolInfo) and [`OptInfo`]
//!   * out.csv  — Raw [`Outcome`](crate::objective::Outcome)
//!
//! In MPI mode, recorder folders are suffixed with the rank.

use crate::{
    BasePartial, FidBasePartial, Fidelity, FolderConfig, OptInfo, SolInfo,
    domain::{Codomain, TypeDom, onto::LinkOpt},
    objective::{Outcome, Step},
    optimizer::Optimizer,
    recorder::Recorder,
    searchspace::{CompShape, Searchspace},
    solution::{
        Batch, HasFidelity, HasId, HasInfo, HasSolInfo, HasStep, HasUncomputed, HasY, Id, OutBatch,
        Solution, SolutionShape, Uncomputed,
        shape::{SolObj, SolOpt},
    },
};

use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::{
    fs::{File, OpenOptions, create_dir_all},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Container holding the [`csv::Writer`] handles used by [`CSVRecorder`].
pub struct CSVFiles {
    pub obj: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub opt: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub cod: Arc<Mutex<csv::Writer<File>>>,
    pub info: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub out: Option<Arc<Mutex<csv::Writer<File>>>>,
}

impl CSVFiles {
    /// Open all configured CSV files and wrap them in thread-safe writers.
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

/// Trait describing how a type is written to CSV.
///
/// Implementers provide a header and a row representation for a given value.
/// `H` is the type of an element used to generate the header
/// used when the header does not depend on the value to write.
/// While `C` is the type of the value to write.
/// 
/// For instance, [`Codomain`](crate::Codomain) implements this trait
/// by using itself as `H` (to generate the header columns depending on
/// the codomain definition), and write its associated [`TypeCodom`](crate::Codomain::TypeCodom) as `C`.
pub trait CSVWritable<H, C> {
    /// Header columns for this type.
    fn header(elem: &H) -> Vec<String>;
    /// Row columns for this value.
    fn write(&self, comp: &C) -> Vec<String>;
}

/// CSV writer for types that have left/right components (e.g. `Obj` and `Opt`).
/// For instance, a [`Var`](crate::Var), use itself for the header generation,
/// and it writes two [`TypeDom`](crate::domain::TypeDom)s for `Obj` [`Domain`](crate::Domain)
/// and `Opt` [`Domain`](crate::Domain).
pub trait CSVLeftRight<H, L, R> {
    /// Header columns for this pair.
    fn header(elem: &H) -> Vec<String>;
    /// Row columns for the left component.
    fn write_left(&self, comp: &L) -> Vec<String>;
    /// Row columns for the right component.
    fn write_right(&self, comp: &R) -> Vec<String>;
}

/// CSV writer for [`SolutionShape`]s, describing how to write the solution components within a CSV file.
/// A [`SolutionShape`] is decomposed into an objective-side part and an optimizer-side part, which are written separately.
/// The [`Searchspace`], defining the structure of a [`Solution`],
///  is used to generate the header and row columns for both parts, while the [`Solution`] are used 
/// to generate the row columns for their respective part.
pub trait SolCSVWrite<PartOpt, SolId, SInfo>: Searchspace<PartOpt, SolId, SInfo>
where
    PartOpt: Uncomputed<SolId, Self::Opt, SInfo>,
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
{
    /// Write the header for objective-side inputs.
    fn header_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>);
    /// Write the header for optimizer-side inputs.
    fn header_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>);
    /// Write a single objective-side row.
    fn write_partial_obj(
        &self,
        id: &[String],
        sol: &SolObj<Self::SolShape, SolId, SInfo>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
    /// Write a single optimizer-side row.
    fn write_partial_opt(
        &self,
        id: &[String],
        sol: &SolOpt<Self::SolShape, SolId, SInfo>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
}

/// CSV writer for [`Codomain`]s, describing how to write a [`TypeCodom`](Codomain::TypeCodom)
/// components within a CSV file.
/// It uses a [`Codomain`], defining the structure of a [`TypeCodom`](Codomain::TypeCodom),
/// to generate the header and row columns, while the codomain values are used to generate the row columns.
pub trait CodCSVWrite<SolId, Out>: Codomain<Out>
where
    Self: Sized + CSVWritable<Self, Self::TypeCodom>,
    SolId: Id + CSVWritable<(), ()>,
    Out: Outcome,
{
    /// Write the codomain header.
    fn header_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>);
    /// Write one codomain row for a solution id.
    fn write_codom(
        &self,
        id: &[String],
        codom: Arc<Self::TypeCodom>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
}

/// CSV writer for solution and optimizer info ([`SolInfo`],[`OptInfo`]),
/// describing how to write the metadata associated with a [`Computed`](crate::Computed) solution within a CSV file.
pub trait InfoCSVWrite<SolId, SInfo, Info>: SolutionShape<SolId, SInfo>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo + CSVWritable<(), ()>,
{
    /// Write the info header (solution info + optimizer info).
    fn header_info(wrt: Arc<Mutex<csv::Writer<File>>>);
    /// Write one info row for a solution id.
    fn write_info(
        &self,
        id: &[String],
        info: Option<Arc<Info>>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
}

/// CSV writer for raw [`Outcome`] values.
pub trait OutCSVWrite<SolId: Id>: Outcome {
    /// Write the outcome header.
    fn header_out(wrt: Arc<Mutex<csv::Writer<File>>>);
    /// Write one outcome row for a solution id.
    fn write_out(&self, id: &[String], wrt: Arc<Mutex<csv::Writer<File>>>);
}

/// Describes how to write ouputs of an evaluation,
/// [`CompShape`]s, raw [`Outcome`]s, and associated metadata within CSV files.
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
    /// Write a fully computed solution and its associated records.
    fn write(
        &self,
        pair: &CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>,
        opair: &(SolId, Out),
        info: Option<Arc<Info>>,
        cod: &Cod,
        wrts: &CSVFiles,
    );
}

/// Implementation for [`BasePartial`] [`Solution`]s, which writes the solution components within the CSV files.
impl<Scp, SolId, SInfo> SolCSVWrite<BasePartial<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo> for Scp
where
    Scp: Searchspace<BasePartial<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo>
        + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
{
    /// Header row columns: `SolId` fields followed by the searchspace columns.
    /// 
    /// # Example
    /// For a [`SId`](crate::SId), a searchspace with 3 [`Variable`](crate::Var)s, the header will be:
    /// ```text
    /// id, var1_name, var2_name, var3_name
    /// ```
    fn header_partial_obj(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(self));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    /// Header row columns: `SolId` fields followed by the searchspace columns.
    /// 
    /// # Example
    /// For a [`SId`](crate::SId), a searchspace with 3 [`Variable`](crate::Var)s, the header will be:
    /// ```text
    /// id, var1_name, var2_name, var3_name
    /// ```
    fn header_partial_opt(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Scp::header(self));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    /// Row columns: `SolId` fields followed by objective-side searchspace values.
    /// 
    /// # Example
    /// For a [`SId`](crate::SId), a searchspace with 3 [`Variable`](crate::Var)s of *
    /// [`Real`](crate::Real) [`Onto`](crate::Onto) [`Int`](crate::Int) types, respectively,
    /// the row could be:
    /// ```text
    /// 3, 10.5,20.4, 30.1
    /// ```
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
    /// Row columns: `SolId` fields followed by optimizer-side searchspace values.
    /// 
    /// # Example
    /// For a [`SId`](crate::SId), a searchspace with 3 [`Variable`](crate::Var)s of *
    /// [`Real`](crate::Real) [`Onto`](crate::Onto) [`Int`](crate::Int) types, respectively,
    /// the row could be:
    /// ```text
    /// 3, 1, 2, 3
    /// ```
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

/// Implementation for [`FidBasePartial`] [`Solution`]s, which adds [`Fidelity`] and [`Step`] columns to the CSV files.
impl<Scp, SolId, SInfo> SolCSVWrite<FidBasePartial<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo>
    for Scp
where
    Scp: Searchspace<FidBasePartial<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo>
        + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
{
    /// Header row columns: `SolId` fields, searchspace columns, then [`Step`] and [`Fidelity`].
    /// 
    /// # Example
    /// For a [`SId`](crate::SId), a searchspace with 3 [`Variable`](crate::Var)s, the header will be:
    /// ```text
    /// id, var1_name, var2_name, var3_name, step, fidelity
    /// ```
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

    /// Header row columns: `SolId` fields, searchspace columns, then [`Step`] and [`Fidelity`].
    /// 
    /// # Example
    /// For a [`SId`](crate::SId), a searchspace with 3 [`Variable`](crate::Var)s, the header will be:
    /// ```text
    /// id, var1_name, var2_name, var3_name, step, fidelity
    /// ```
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

    /// Row columns: `SolId` fields, objective-side searchspace values, then [`Step`] and [`Fidelity`].
    /// 
    /// # Example
    /// For a [`SId`](crate::SId), a searchspace with 3 [`Variable`](crate::Var)s of [`Real`](crate::Real) [`Onto`](crate::Onto) [`Int`](crate::Int) types, respectively,
    /// the row could be:
    /// ```text
    /// 3, 10.5,20.4, 30.1, 2, 0.5
    /// ```
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
    /// Row columns: `SolId` fields, optimizer-side searchspace values, then [`Step`] and [`Fidelity`].
    /// 
    /// # Example
    /// For a [`SId`](crate::SId), a searchspace with 3 [`Variable`](crate::Var)s of [`Real`](crate::Real) [`Onto`](crate::Onto) [`Int`](crate::Int) types, respectively,
    /// the row could be:
    /// ```text
    /// 3, 1,2, 3, 2, 0.5
    /// ```
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
    /// Header row columns: `SolId` fields followed by codomain columns.
    ///
    /// # Example
    /// With a [`ConstCodomain`](crate::ConstCodomain) defining `y` and two constraints, the
    /// header will be:
    /// ```text
    /// id, y, c0, c1
    /// ```
    fn header_codom(&self, wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Cod::header(self));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }
    /// Row columns: `SolId` fields followed by codomain values.
    ///
    /// # Example
    /// For a [`ConstCodomain`](crate::ConstCodomain) with `y=0.42` and two constraints
    /// `[-1.0, 0.0]`, the row could be:
    /// ```text
    /// 3, 0.42, -1.0, 0.0
    /// ```
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
    /// Header row columns: `SolId` fields, then [`SolInfo`] columns, then [`OptInfo`] columns.
    ///
    /// # Example
    /// With a mock `SolInfo` that exposes `age` and `seed`, and a mock `OptInfo` that exposes
    /// `iter` and `batch`, the header could be:
    /// ```text
    /// id, age, seed, iter, batch
    /// ```
    fn header_info(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    /// Row columns: `SolId` fields, [`SolInfo`] values, then [`OptInfo`] values if present.
    ///
    /// # Example
    /// With the mock headers above and `id=3`, `age=12`, `seed=42`, `iter=7`, `batch=2`, the row
    /// could be:
    /// ```text
    /// 3, 12, 42, 7, 2
    /// ```
    fn write_info(
        &self,
        id: &[String],
        info: Option<Arc<Info>>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    ) {
        let mut sinfostr = self.get_sinfo().write(&());
        let idstr = match info {
            Some(i) => {
                sinfostr.append(&mut i.write(&()));
                id.iter().chain(sinfostr.iter())
            }
            None => id.iter().chain(sinfostr.iter()),
        };
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
    /// Header row columns: `SolId` fields followed by outcome columns.
    ///
    /// # Example
    /// With a mock `Outcome` that exposes `loss` and `acc`, the header could be:
    /// ```text
    /// id, loss, acc
    /// ```
    fn header_out(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(Out::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    /// Row columns: `SolId` fields followed by outcome values.
    ///
    /// # Example
    /// With the mock headers above and `id=3`, `loss=0.12`, `acc=0.97`, the row could be:
    /// ```text
    /// 3, 0.12, 0.97
    /// ```
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
        info: Option<Arc<Info>>,
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

/// Describes how to write a [`Batch`] of [`Computed`](crate::Computed) [`SolShape`](crate::SolShape), 
/// and associated metadata within CSV files.
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
    /// Write a [`Batch`] (`Self`) of computed solutions and their associated raw [`Outcome`].
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
        self.into_par_iter().zip(obatch).for_each(|(cpair, opair)| {
            scp.write(cpair, opair, Some(info.clone()), cod, &wrts.clone())
        });
    }
}

/// Recorder that saves computed solutions and outputs as CSV.
///
/// The computed [`Codomain`] is always saved by default.
///
/// # Attributes
///
/// * `path` : `&'static` [`str`] - The path to where the files should be created.
/// * `obj` : bool - If `true` computed `Obj` [`Solution`] will be saved.
/// * `opt` : bool - If `true` computed `Opt` [`Solution`] will be saved.
/// * `info` : bool - If `true` [`SolInfo`] and [`OptInfo`] from computed [`Solution`] will be saved.
/// * `out` : bool - If `true` computed [`Outcome`] will be saved.
///
/// # Notes on file hierarchy
///
/// The 4 csv files information are linked by the unique [`Id`] of computed [`Solution`].
///
/// * `path`
///  * recorder
///   * obj.csv             ([`Objective`](crate::objective::Objective) points)
///   * opt.csv             ([`Optimizer`] points)
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
    /// Initialize the recorder folder and CSV files, and write CSV headers.
    ///
    /// # File hierarchy
    /// ```text
    /// recorder/
    ///   obj.csv
    ///   opt.csv
    ///   cod.csv
    ///   info.csv
    ///   out.csv
    /// ```
    /// Files are created based on the `obj/opt/info/out` flags, while `cod.csv` is always created.
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

    /// Validate that the recorder folder and configured CSV files exist after a [`load!`](crate::load) macro call.
    fn after_load(&mut self, _scp: &Scp, _cod: &Op::Cod) {
        // Check if all folder and files exist
        if self.config.path_rec.try_exists().unwrap() {
            if let Some(ppobj) = &self.path_pobj
                && !ppobj.try_exists().unwrap()
            {
                panic!(
                    "The `Objective` recorder file does not exist, {}.",
                    ppobj.display()
                )
            }

            if let Some(ppopt) = &self.path_popt
                && !ppopt.try_exists().unwrap()
            {
                panic!(
                    "The `Optimizer` recorder file does not exist, {}.",
                    ppopt.display()
                )
            }

            if let Some(ppinfo) = &self.path_info
                && !ppinfo.try_exists().unwrap()
            {
                panic!("The `Info` file does not exist, {}.", ppinfo.display())
            }

            if let Some(ppout) = &self.path_out
                && !ppout.try_exists().unwrap()
            {
                panic!(
                    "The `Output` recorder file does not exist, {}.",
                    ppout.display()
                )
            }

            if !self.path_codom.try_exists().unwrap() {
                panic!(
                    "The `Codomain` recorder file does not exist, {}.",
                    self.path_codom.display()
                )
            }
        } else {
            panic!(
                "The recorder folder does not exist, {}.",
                self.config.path_rec.display()
            );
        }
    }

    /// Write a full batch of computed solutions and their associated outputs.
    ///
    /// # Example
    /// During batched optimization, one call to `save_batch` writes *one row per solution* to each
    /// enabled CSV file (`obj.csv`, `opt.csv`, `info.csv`, `out.csv`) plus `cod.csv`.
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

    /// Write a single computed pair (`CompShape`) and its raw outcome.
    ///
    /// # Example
    /// In sequential optimization, `save_pair` is called per evaluation, appending exactly one row
    /// to each enabled CSV file (`obj.csv`, `opt.csv`, `info.csv`, `out.csv`) plus `cod.csv`.
    fn save_pair(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
        info: Option<Arc<Op::Info>>,
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

/// Version of [`CSVRecorder`] for MPI-distributed algorithms.
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
///
/// # Notes on file hierarchy
///
/// The 4 csv files information are linked by the unique [`Id`] of computed [`Solution`].
///
/// * `path`
///  * recorder_rank`{rank}`
///   * obj.csv             ([`Objective`](crate::objective::Objective) points)
///   * opt.csv             ([`Optimizer`] points)
///   * info.csv            ([`SolInfo`] and [`OptInfo`])
///   * out.csv             ([`Outcome`])
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
    /// Initialize the distributed recorder folder and CSV files, and write CSV headers.
    ///
    /// # File hierarchy
    /// ```text
    /// recorder_rank{rank}/
    ///   obj.csv
    ///   opt.csv
    ///   cod.csv
    ///   info.csv
    ///   out.csv
    /// ```
    /// Files are created based on the `obj/opt/info/out` flags, while `cod.csv` is always created.
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

    /// Validate that the distributed recorder folder and configured CSV files exist after a load.
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {
        // Check if all folder and files exist
        if self.config.path_rec.try_exists().unwrap() {
            if let Some(ppobj) = &self.path_pobj
                && !ppobj.try_exists().unwrap()
            {
                panic!(
                    "The `Objective` recorder file does not exists, {}.",
                    ppobj.display()
                )
            }

            if let Some(ppopt) = &self.path_popt
                && !ppopt.try_exists().unwrap()
            {
                panic!(
                    "The `Optimizer` recorder file  not exists, {}.",
                    ppopt.display()
                )
            }

            if let Some(ppinfo) = &self.path_info
                && !ppinfo.try_exists().unwrap()
            {
                panic!("The `Info` file does not exists, {}.", ppinfo.display())
            }

            if let Some(ppout) = &self.path_out
                && !ppout.try_exists().unwrap()
            {
                panic!(
                    "The `Output` recorder file does not exists, {}.",
                    ppout.display()
                )
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

    /// Write a full [`Batch`] of [`Computed`](crate::Computed)
    /// solutions and their associated [`Outcome`] (per MPI rank where the initial [`Solution`] was generated).
    ///
    /// # Example
    /// In multi-instance runs ([`MultiInstanceOptimizer`](crate::optimizer::opt::MultiInstanceOptimizer)),
    /// each rank writes *one row per solution* to its dedicated CSV files
    /// (`obj.csv`, `opt.csv`, `info.csv`, `out.csv`) plus `cod.csv` under `recorder_rank{rank}/`.
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

    /// Write a single computed pair (`CompShape`) and its raw outcome (per MPI rank).
    ///
    /// # Example
    /// In multi-instance runs ([`MultiInstanceOptimizer`](crate::optimizer::opt::MultiInstanceOptimizer)),
    /// each rank writes *one row per solution* to its dedicated CSV files
    /// (`obj.csv`, `opt.csv`, `info.csv`, `out.csv`) plus `cod.csv` under `recorder_rank{rank}/`.
    fn save_pair_dist(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
        scp: &Scp,
        cod: &Op::Cod,
        info: Option<Arc<Op::Info>>,
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
