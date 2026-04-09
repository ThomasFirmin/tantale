//! CSV recording utilities and recorder implementation.
//!
//! This module defines the traits required to serialize solutions, codomain
//! values, optimizer metadata, and outcomes into CSV files, plus the
//! [`CSVRecorder`] that writes them to disk.
//!
//! ## Trait Hierarchy
//!
//! The CSV recording system is built on several trait layers:
//!
//! - [`CSVWritable`] - Base trait for writing individual types to CSV columns
//! - [`CSVLeftRight`] - For types with Obj/Opt components (used by [`Searchspace`](crate::Searchspace))
//! - [`SolCSVWrite`] - Writing [`Solution`](crate::Solution) components (obj.csv, opt.csv)
//! - [`CodCSVWrite`] - Writing [`Codomain`](crate::Codomain) values (cod.csv)
//! - [`InfoCSVWrite`] - Writing [`SolInfo`](crate::SolInfo) and [`OptInfo`](crate::OptInfo) metadata (info.csv)
//! - [`OutCSVWrite`] - Writing raw [`Outcome`](crate::objective::Outcome) values (out.csv)
//! - [`ScpCSVWrite`] - Orchestrates writing all components of a computed solution
//! - [`BatchCSVWrite`] - Writes batches of solutions in parallel
//!
//! ## File Layout
//!
//! By default, CSV files are created under the recorder folder (see
//! [`FolderConfig`](crate::FolderConfig)). Each row is linked by the solution
//! [`Id`](crate::solution::Id).
//!
//! ```text
//! recorder/
//!   obj.csv  - Objective-side inputs
//!   opt.csv  - Optimizer-side inputs
//!   cod.csv  - Codomain values (always created)
//!   info.csv - SolInfo and OptInfo metadata
//!   out.csv  - Raw Outcome values
//! ```
//!
//! In MPI mode, recorder folders are suffixed with the rank (e.g., `recorder_rank0`).

use crate::{
    BaseSol, BatchOptimizer, Fidelity, FidelitySol, FolderConfig, FuncWrapper, OptInfo, RawObj, SequentialOptimizer, SolInfo, StepId, domain::{Codomain, TypeDom, onto::LinkOpt}, objective::{Outcome, Step}, recorder::{BatchRecorder, Recorder, SeqRecorder}, searchspace::{CompShape, Searchspace}, solution::{
        Batch, HasFidelity, HasId, HasInfo, HasSolInfo, HasStep, HasUncomputed, HasY, Id, OutBatch,
        Solution, SolutionShape, Uncomputed,
        shape::{SolObj, SolOpt},
    }
};

use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use std::{
    fs::{File, OpenOptions, create_dir_all},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Container holding thread-safe CSV writer handles for all output files.
///
/// Each writer is wrapped in `Arc<Mutex<_>>` to enable parallel writes from multiple threads.
/// Only `cod` is required; other files are optional based on [`CSVRecorder`] configuration.
pub struct CSVFiles {
    pub obj: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub opt: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub cod: Arc<Mutex<csv::Writer<File>>>,
    pub info: Option<Arc<Mutex<csv::Writer<File>>>>,
    pub out: Option<Arc<Mutex<csv::Writer<File>>>>,
}

impl CSVFiles {
    /// Open all configured CSV files and wrap them in thread-safe writers.
    ///
    /// Files are opened in append mode to support resuming experiments.
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
use crate::{
    experiment::mpi::utils::MPIProcess,
    recorder::{DistBatchRecorder, DistSeqRecorder},
};

/// Base trait for writing a type to CSV with headers and values.
///
/// This trait separates header generation from value serialization, allowing
/// headers to be defined once and values to be written many times.
///
/// # Type Parameters
///
/// - `H` - The type used to generate column headers (often `Self` or `()`)
/// - `C` - The type of the value to write as a CSV row
///
/// # Examples
///
/// A [`Codomain`](crate::Codomain) uses itself as `H` to generate headers based on its
/// structure, and writes its associated [`TypeCodom`](crate::Codomain::TypeCodom) as `C`.
///
/// An [`Id`](crate::solution::Id) typically uses `()` for both `H` and `C`, as headers
/// and values are self-contained.
pub trait CSVWritable<H, C> {
    /// Generate header column names.
    ///
    /// # Parameters
    ///
    /// * `elem` - Element providing header structure information
    fn header(elem: &H) -> Vec<String>;

    /// Write value as CSV row columns.
    ///
    /// # Parameters
    ///
    /// * `comp` - Component to serialize
    fn write(&self, comp: &C) -> Vec<String>;
}

/// CSV writer for types with separate Obj and Opt components.
///
/// This trait is used by [`Searchspace`](crate::Searchspace) implementations to write
/// objective-side and optimizer-side values separately to obj.csv and opt.csv.
///
/// # Type Parameters
///
/// - `H` - The type used to generate column headers (typically the searchspace itself)
/// - `L` - The left (Obj) component type to write
/// - `R` - The right (Opt) component type to write
///
/// # Example
///
/// A [`Var`](crate::Var) uses itself for header generation and writes
/// [`TypeDom`](crate::domain::TypeDom) values for both Obj and Opt [`Domain`](crate::Domain)s.
pub trait CSVLeftRight<H, L, R> {
    /// Generate header columns (shared by both Obj and Opt).
    fn header(elem: &H) -> Vec<String>;

    /// Write Obj (left) component as CSV columns.
    fn write_left(&self, comp: &L) -> Vec<String>;

    /// Write Opt (right) component as CSV columns.
    fn write_right(&self, comp: &R) -> Vec<String>;
}

/// CSV writer for [`Solution`](crate::Solution) components (obj.csv and opt.csv).
///
/// This trait enables a [`Searchspace`](crate::Searchspace) to write solution components
/// by decomposing them into objective-side (obj.csv) and optimizer-side (opt.csv) parts.
/// The searchspace generates headers and serializes values for both parts.
///
/// # CSV Files
///
/// - **obj.csv**: Contains objective-side inputs that are passed to the objective function
/// - **opt.csv**: Contains optimizer-side representations used internally by the optimizer
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

/// CSV writer for [`Codomain`](crate::Codomain) values (cod.csv).
///
/// This trait enables a [`Codomain`](crate::Codomain) to serialize its
/// [`TypeCodom`](crate::Codomain::TypeCodom) values. The codomain structure
/// defines the headers, while computed values are written as rows.
///
/// # CSV File
///
/// - **cod.csv**: Contains computed objective values, constraints, and other codomain elements
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

/// CSV writer for solution metadata (info.csv).
///
/// This trait enables writing [`SolInfo`](crate::SolInfo) and optionally
/// [`OptInfo`](crate::OptInfo) metadata associated with computed solutions.
///
/// # CSV File
///
/// - **info.csv**: Contains metadata like age, seed, iteration number, batch number, etc.
pub trait InfoCSVWrite<SolId, SInfo>: SolutionShape<SolId, SInfo>
where
    SolId: Id,
    SInfo: SolInfo,
{
    /// Write the info header (solution info only).
    fn header_info(wrt: Arc<Mutex<csv::Writer<File>>>);

    /// Write the info header (solution info + optimizer info).
    fn header_info_with_optinfo<Info: OptInfo + CSVWritable<(), ()>>(
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
    /// Write one info row (solution info only).
    fn write_info(&self, id: &[String], wrt: Arc<Mutex<csv::Writer<File>>>);

    /// Write one info row (solution info + optimizer info).
    fn write_info_with_optinfo<Info: OptInfo + CSVWritable<(), ()>>(
        &self,
        id: &[String],
        info: Arc<Info>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    );
}

/// CSV writer for raw [`Outcome`](crate::objective::Outcome) values (out.csv).
///
/// This trait enables writing raw outcome values returned by the objective function.
///
/// # CSV File
///
/// - **out.csv**: Contains raw outcomes before interpretation into the codomain
pub trait OutCSVWrite<SolId: Id>: Outcome {
    /// Write the outcome header.
    fn header_out(wrt: Arc<Mutex<csv::Writer<File>>>);
    /// Write one outcome row for a solution id.
    fn write_out(&self, id: &[String], wrt: Arc<Mutex<csv::Writer<File>>>);
}

/// Orchestrates writing all components of a computed solution to CSV files.
///
/// This trait combines [`SolCSVWrite`], [`CodCSVWrite`], [`InfoCSVWrite`], and [`OutCSVWrite`]
/// to write a complete [`CompShape`](crate::searchspace::CompShape) and its raw [`Outcome`](crate::objective::Outcome)
/// across all CSV files in a coordinated manner.
pub trait ScpCSVWrite<PartOpt, SolId, SInfo, Cod, Out>: Searchspace<PartOpt, SolId, SInfo>
where
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
    Out: Outcome + CSVWritable<(), ()>,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    PartOpt: Uncomputed<SolId, Self::Opt, SInfo>,
    CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>: SolutionShape<SolId, SInfo> + HasY<Cod, Out>,
    SolObj<CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Self::Obj, SInfo, Uncomputed = SolObj<Self::SolShape, SolId, SInfo>>,
    SolOpt<CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Self::Opt, SInfo, Uncomputed = SolOpt<Self::SolShape, SolId, SInfo>>,
{
    /// Write a fully computed solution without optimizer info.
    ///
    /// Used by sequential optimizers that don't provide batch-level metadata.
    fn write(
        &self,
        pair: &CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>,
        opair: &(SolId, Out),
        cod: &Cod,
        wrts: &CSVFiles,
    );

    /// Write a fully computed solution with optimizer info.
    ///
    /// Used by batch optimizers that provide iteration/batch metadata via [`OptInfo`](crate::OptInfo).
    fn write_with_opt_info<Info: OptInfo + CSVWritable<(), ()>>(
        &self,
        pair: &CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>,
        opair: &(SolId, Out),
        info: Arc<Info>,
        cod: &Cod,
        wrts: &CSVFiles,
    );
}

/// Implementation for [`BasePartial`] [`Solution`]s, which writes the solution components within the CSV files.
impl<Scp, SolId, SInfo> SolCSVWrite<BaseSol<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo> for Scp
where
    Scp: Searchspace<BaseSol<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo>
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
        let solstr = self.write_left(sol.get_x());
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
        let solstr = self.write_right(sol.get_x());
        let idstr = id.iter().chain(solstr.iter());
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }
}

/// Implementation for [`FidBasePartial`] [`Solution`]s, which adds [`Fidelity`] and [`Step`] columns to the CSV files.
impl<Scp, SolId, SInfo> SolCSVWrite<FidelitySol<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo> for Scp
where
    Scp: Searchspace<FidelitySol<SolId, LinkOpt<Scp>, SInfo>, SolId, SInfo>
        + CSVLeftRight<Scp, Arc<[TypeDom<Scp::Obj>]>, Arc<[TypeDom<Scp::Opt>]>>,
    SolId: StepId + CSVWritable<(), ()>,
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
        let solstr = self.write_left(sol.get_x());
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
        let solstr = self.write_right(sol.get_x());
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

impl<Shape, SolId, SInfo> InfoCSVWrite<SolId, SInfo> for Shape
where
    Shape: SolutionShape<SolId, SInfo> + HasSolInfo<SInfo>,
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
{
    /// Header row columns: `SolId` fields, then [`SolInfo`] columns.
    ///
    /// # Example
    /// With a mock `SolInfo` that exposes `age` and `seed` the header could be:
    /// ```text
    /// id, age, seed
    /// ```
    fn header_info(wrt: Arc<Mutex<csv::Writer<File>>>) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    /// Header row columns: `SolId` fields, then [`SolInfo`] columns.
    ///
    /// # Example
    /// With a mock `SolInfo` that exposes `age` and `seed`, and a mock `OptInfo` that exposes
    /// `iter` and `batch`, the header could be:
    /// ```text
    /// id, age, seed, iter, batch
    /// ```
    fn header_info_with_optinfo<Info: OptInfo + CSVWritable<(), ()>>(
        wrt: Arc<Mutex<csv::Writer<File>>>,
    ) {
        let mut lwrt = wrt.lock().unwrap();
        let mut idstr = SolId::header(&());
        idstr.extend(SInfo::header(&()));
        idstr.extend(Info::header(&()));
        lwrt.write_record(idstr).unwrap();
        lwrt.flush().unwrap();
    }

    /// Row columns: `SolId` fields, [`SolInfo`] values.
    ///
    /// # Example
    /// With the mock headers above and `id=3`, `age=12`, `seed=42`, the row
    /// could be:
    /// ```text
    /// 3, 12, 42
    /// ```
    fn write_info(&self, id: &[String], wrt: Arc<Mutex<csv::Writer<File>>>) {
        let sinfostr = self.sinfo().write(&());
        let idstr = id.iter().chain(sinfostr.iter());
        let mut wrt_local = wrt.lock().unwrap();
        wrt_local.write_record(idstr).unwrap();
        wrt_local.flush().unwrap();
    }

    /// Row columns: `SolId` fields, [`SolInfo`] and [`OptInfo`] values.
    ///
    /// # Example
    /// With the mock headers above and `id=3`, `age=12`, `seed=42`, `iter=7`, `batch=2`, the row
    /// could be:
    /// ```text
    /// 3, 12, 42, 7, 2
    /// ```
    fn write_info_with_optinfo<Info: OptInfo + CSVWritable<(), ()>>(
        &self,
        id: &[String],
        info: Arc<Info>,
        wrt: Arc<Mutex<csv::Writer<File>>>,
    ) {
        let sinfostr = self.sinfo().write(&());
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

impl<Scp, PartOpt, SolId, SInfo, Cod, Out> ScpCSVWrite<PartOpt, SolId, SInfo, Cod, Out> for Scp
where
    SolId: Id + CSVWritable<(), ()>,
    SInfo: SolInfo + CSVWritable<(), ()>,
    Out: Outcome + OutCSVWrite<SolId> + CSVWritable<(), ()> + Send + Sync,
    Cod: Codomain<Out> + CSVWritable<Cod, Cod::TypeCodom>,
    Cod::TypeCodom: Send + Sync,
    PartOpt: Uncomputed<SolId, Self::Opt, SInfo>,
    Scp: SolCSVWrite<PartOpt, SolId, SInfo>,
    CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>: InfoCSVWrite<SolId, SInfo>,
    SolObj<CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Self::Obj, SInfo, Uncomputed = SolObj<Self::SolShape, SolId, SInfo>>,
    SolOpt<CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>, SolId, SInfo>:
        HasUncomputed<SolId, Self::Opt, SInfo, Uncomputed = SolOpt<Self::SolShape, SolId, SInfo>>,
{
    fn write(
        &self,
        pair: &CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>,
        opair: &(SolId, Out),
        cod: &Cod,
        wrts: &CSVFiles,
    ) {
        let id = pair.id();
        let idstr = id.write(&());

        // CODOM
        <Cod as CodCSVWrite<SolId, Out>>::write_codom(cod, &idstr, pair.y(), wrts.cod.clone());
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
            pair.write_info(&idstr, f);
        }
        // OUT
        if let Some(f) = wrts.out.clone() {
            opair.1.write_out(&idstr, f);
        }
    }

    fn write_with_opt_info<Info: OptInfo + CSVWritable<(), ()>>(
        &self,
        pair: &CompShape<Self, PartOpt, SolId, SInfo, Cod, Out>,
        opair: &(SolId, Out),
        info: Arc<Info>,
        cod: &Cod,
        wrts: &CSVFiles,
    ) {
        let id = pair.id();
        let idstr = id.write(&());

        // CODOM
        <Cod as CodCSVWrite<SolId, Out>>::write_codom(cod, &idstr, pair.y(), wrts.cod.clone());
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
            pair.write_info_with_optinfo(&idstr, info, f);
        }
        // OUT
        if let Some(f) = wrts.out.clone() {
            opair.1.write_out(&idstr, f);
        }
    }
}

/// Writes a [`Batch`](crate::Batch) of computed solutions to CSV files in parallel.
///
/// This trait enables efficient parallel writing of multiple solutions using Rayon.
/// Each solution in the batch is written to all configured CSV files simultaneously.
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
    /// Write a batch of computed solutions and raw outcomes in parallel.
    ///
    /// # Parameters
    ///
    /// * `obatch` - Batch of raw outcomes corresponding to the solutions
    /// * `scp` - Searchspace for interpreting solutions
    /// * `cod` - Codomain for interpreting outcomes
    /// * `wrts` - Thread-safe CSV file handles
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
        + ScpCSVWrite<PartOpt, SolId, SInfo, Cod, Out>
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
        let info = self.info();
        self.into_par_iter().zip(obatch).for_each(|(cpair, opair)| {
            scp.write_with_opt_info(cpair, opair, info.clone(), cod, &wrts.clone())
        });
    }
}

/// CSV-based recorder implementation for optimization experiments.
///
/// The [`CSVRecorder`] saves computed solutions and their outcomes as CSV files.
/// The codomain (cod.csv) is always saved; other files are optional.
///
/// # Configuration
///
/// Create a recorder using [`CSVRecorder::new`] with a [`FolderConfig`](crate::FolderConfig)
/// and boolean flags for each optional file:
///
/// * `obj` - If `true`, saves objective-side inputs (obj.csv)
/// * `opt` - If `true`, saves optimizer-side representations (opt.csv)
/// * `info` - If `true`, saves metadata (info.csv)
/// * `out` - If `true`, saves raw outcomes (out.csv)
///
/// # File Structure
///
/// All CSV files are linked by the solution [`Id`](crate::solution::Id):
///
/// ```text
/// <config.path>/
///   recorder/
///     obj.csv   - Objective-side inputs
///     opt.csv   - Optimizer-side representations
///     cod.csv   - Codomain values (always created)
///     info.csv  - SolInfo and OptInfo metadata
///     out.csv   - Raw Outcome values
/// ```
///
/// In MPI mode, folders are suffixed: `recorder_rank0/`, `recorder_rank1/`, etc.
///
/// # See Also
///
/// - [`NoSaver`](crate::NoSaver) - No-op recorder for experiments without persistence
/// - [`SeqRecorder`](crate::SeqRecorder) - Trait for sequential recorders
/// - [`BatchRecorder`](crate::BatchRecorder) - Trait for batch recorders
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
    /// Create a new CSV recorder with optional file outputs.
    ///
    /// # Parameters
    ///
    /// * `config` - Folder configuration specifying the recorder directory
    /// * `obj` - Enable obj.csv (objective-side inputs)
    /// * `opt` - Enable opt.csv (optimizer-side representations)
    /// * `info` - Enable info.csv (metadata)
    /// * `out` - Enable out.csv (raw outcomes)
    ///
    /// # Returns
    ///
    /// `Some(CSVRecorder)` with file paths configured.
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

impl Recorder for CSVRecorder {}

impl<PSol, SolId, Out, Scp, Op, FnWrap> SeqRecorder<PSol, SolId, Out, Scp, Op, FnWrap>
    for CSVRecorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: OutCSVWrite<SolId> + CSVWritable<(), ()> + Send + Sync,
    Op: SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::Cod: CodCSVWrite<SolId, Out>
        + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>
        + Send
        + Sync,
    Scp: SolCSVWrite<PSol, SolId, Op::SInfo>
        + ScpCSVWrite<PSol, SolId, Op::SInfo, Op::Cod, Out>
        + Send
        + Sync,
    Scp::SolShape: InfoCSVWrite<SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        InfoCSVWrite<SolId, Op::SInfo> + HasY<Op::Cod, Out> + Send + Sync,
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
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
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

    /// Write a single computed pair (`CompShape`) and its raw outcome.
    ///
    /// # Example
    /// In sequential optimization, `save_pair` is called per evaluation, appending exactly one row
    /// to each enabled CSV file (`obj.csv`, `opt.csv`, `info.csv`, `out.csv`) plus `cod.csv`.
    fn save(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
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
        scp.write(computed, outputed, cod, &files);
    }
}

impl<PSol, SolId, Out, Scp, Op, FnWrap> BatchRecorder<PSol, SolId, Out, Scp, Op, FnWrap>
    for CSVRecorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: OutCSVWrite<SolId> + CSVWritable<(), ()> + Send + Sync,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::Cod: CodCSVWrite<SolId, Out>
        + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>
        + Send
        + Sync,
    Scp: SolCSVWrite<PSol, SolId, Op::SInfo>
        + ScpCSVWrite<PSol, SolId, Op::SInfo, Op::Cod, Out>
        + Send
        + Sync,
    Scp::SolShape: InfoCSVWrite<SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        InfoCSVWrite<SolId, Op::SInfo> + HasY<Op::Cod, Out> + Send + Sync,
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
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
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
                Scp::SolShape::header_info_with_optinfo::<Op::Info>(f);
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
    fn save(
        &self,
        computed: &crate::CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
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
}

/// Implementation of [`DistSeqRecorder`](crate::DistSeqRecorder) for MPI-distributed sequential experiments.
///
/// The CSV recorder creates rank-specific folders (e.g., `recorder_rank0/`) to avoid
/// write conflicts between MPI processes. The codomain (cod.csv) is always saved.
///
/// # File Structure
///
/// ```text
/// <config.path>/
///   recorder_rank0/
///     obj.csv, opt.csv, cod.csv, info.csv, out.csv
///   recorder_rank1/
///     obj.csv, opt.csv, cod.csv, info.csv, out.csv
///   ...
/// ```
///
/// Each rank maintains independent CSV files linked by solution [`Id`](crate::solution::Id).
#[cfg(feature = "mpi")]
impl<PSol, SolId, Out, Scp, Op, FnWrap> DistSeqRecorder<PSol, SolId, Out, Scp, Op, FnWrap>
    for CSVRecorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: OutCSVWrite<SolId> + CSVWritable<(), ()> + Send + Sync,
    Op: SequentialOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::Cod: CodCSVWrite<SolId, Out>
        + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>
        + Send
        + Sync,
    Scp: SolCSVWrite<PSol, SolId, Op::SInfo>
        + ScpCSVWrite<PSol, SolId, Op::SInfo, Op::Cod, Out>
        + Send
        + Sync,
    Scp::SolShape: InfoCSVWrite<SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        InfoCSVWrite<SolId, Op::SInfo> + HasY<Op::Cod, Out> + Send + Sync,
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
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
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
    fn init_dist(&mut self, _proc: &MPIProcess, scp: &Scp, cod: &Op::Cod) {
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
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {
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

    /// Write a single computed pair (`CompShape`) and its raw outcome.
    ///
    /// # Example
    /// In sequential optimization, `save_pair` is called per evaluation, appending exactly one row
    /// to each enabled CSV file (`obj.csv`, `opt.csv`, `info.csv`, `out.csv`) plus `cod.csv`.
    fn save_dist(
        &self,
        computed: &CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>,
        outputed: &(SolId, Out),
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
        scp.write(computed, outputed, cod, &files);
    }
}

/// Implementation of [`DistBatchRecorder`](crate::DistBatchRecorder) for MPI-distributed batch experiments.
///
/// The CSV recorder creates rank-specific folders (e.g., `recorder_rank0/`) to avoid
/// write conflicts between MPI processes. The codomain (cod.csv) is always saved.
///
/// # File Structure
///
/// ```text
/// <config.path>/
///   recorder_rank0/
///     obj.csv, opt.csv, cod.csv, info.csv, out.csv
///   recorder_rank1/
///     obj.csv, opt.csv, cod.csv, info.csv, out.csv
///   ...
/// ```
///
/// Each rank maintains independent CSV files linked by solution [`Id`](crate::solution::Id).
#[cfg(feature = "mpi")]
impl<PSol, SolId, Out, Scp, Op, FnWrap> DistBatchRecorder<PSol, SolId, Out, Scp, Op, FnWrap>
    for CSVRecorder
where
    PSol: Uncomputed<SolId, Scp::Opt, Op::SInfo>,
    SolId: Id + CSVWritable<(), ()> + Send + Sync,
    Out: OutCSVWrite<SolId> + CSVWritable<(), ()> + Send + Sync,
    Op: BatchOptimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp, FnWrap>,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::Cod: CodCSVWrite<SolId, Out>
        + CSVWritable<Op::Cod, <Op::Cod as Codomain<Out>>::TypeCodom>
        + Send
        + Sync,
    Scp: SolCSVWrite<PSol, SolId, Op::SInfo>
        + ScpCSVWrite<PSol, SolId, Op::SInfo, Op::Cod, Out>
        + Send
        + Sync,
    Scp::SolShape: InfoCSVWrite<SolId, Op::SInfo>,
    CompShape<Scp, PSol, SolId, Op::SInfo, Op::Cod, Out>:
        InfoCSVWrite<SolId, Op::SInfo> + HasY<Op::Cod, Out> + Send + Sync,
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
    FnWrap: FuncWrapper<RawObj<Scp::SolShape, SolId, Op::SInfo>>,
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
    fn init_dist(&mut self, _proc: &MPIProcess, scp: &Scp, cod: &Op::Cod) {
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
                Scp::SolShape::header_info_with_optinfo::<Op::Info>(f);
            }

            if let Some(f) = files.out.clone() {
                Out::header_out(f);
            }

            cod.header_codom(files.cod.clone());
        }
    }

    /// Validate that the recorder folder and configured CSV files exist after a [`load!`](crate::load) macro call.
    fn after_load_dist(&mut self, _proc: &MPIProcess, _scp: &Scp, _cod: &Op::Cod) {
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
    fn save_dist(
        &self,
        computed: &crate::CompBatch<SolId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, Out>,
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
}
