//! CSV persistence for solution inputs and evaluation results.
//!
//! [`CSVRecorder`] writes selected record targets to files under a
//! [`FolderConfig`]'s recorder path. The codomain output is recorded by the
//! experiment, while objective inputs, optimizer inputs, metadata, and raw
//! outcomes are controlled by the recorder configuration. See the crate-level
//! documentation for setup and usage examples.

use crate::{
    Computed, FolderConfig, HasVariables, Problem, Var,
    aliases::{Obj, Out, SObj, SOpt, TrueOpt},
    errors::RecorderError,
    objective::outcome::IdOut,
    recorder::{Recorder, RecorderTargets},
    utils::record::{CompParts, Recordable, Schema, SolParts},
};

use std::{
    fs::{File, OpenOptions, create_dir_all},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// CSV-based recorder implementation for optimization experiments.
///
/// The [`CSVRecorder`] writes solution inputs and evaluation results to
/// separate CSV files under a [`FolderConfig`]. Use [`deactivate`](CSVRecorder::deactivate)
/// to disable a target.
///
/// # File Structure
///
/// Rows for a candidate are associated through its solution identifier.
///
/// - `obj.csv`: objective-facing inputs
/// - `opt.csv`: optimizer-facing representations
/// - `cod.csv`: codomain values
/// - `info.csv`: solution metadata
/// - `out.csv`: raw outcomes
///
/// # See Also
///
/// - [`Recorder`] - Recording interface
pub struct CSVRecorder {
    pub config: FolderConfig,
    path_pobj: Option<PathBuf>,
    path_popt: Option<PathBuf>,
    path_info: Option<PathBuf>,
    path_codom: Option<PathBuf>,
    path_out: Option<PathBuf>,
}

impl CSVRecorder {
    /// Create a CSV recorder with paths for each record target.
    ///
    /// # Parameters
    ///
    /// * `config` - Folder configuration specifying the recorder directory
    ///
    /// # Returns
    ///
    /// Returns `Some(CSVRecorder)` with file paths configured.
    pub fn new(config: FolderConfig) -> Option<Self> {
        let path_pobj = Some(config.path_rec.join(Path::new("obj.csv")));
        let path_popt = Some(config.path_rec.join(Path::new("opt.csv")));
        let path_info = Some(config.path_rec.join(Path::new("info.csv")));
        let path_out = Some(config.path_rec.join(Path::new("out.csv")));
        let path_codom = Some(config.path_rec.join(Path::new("cod.csv")));

        Some(CSVRecorder {
            config,
            path_pobj,
            path_popt,
            path_info,
            path_codom,
            path_out,
        })
    }

    /// Retrieve the file path for a given recorder target.
    pub fn retrieve(&self, target: &RecorderTargets) -> &Option<PathBuf> {
        match target {
            RecorderTargets::XObjective => &self.path_pobj,
            RecorderTargets::XOptimizer => &self.path_popt,
            RecorderTargets::Y => &self.path_codom,
            RecorderTargets::Outcome => &self.path_out,
            RecorderTargets::Metadata => &self.path_info,
            RecorderTargets::Other(_) => &None,
        }
    }

    /// Create the CSV file for a specific target if it does not exist.
    pub fn create(&self, target: &RecorderTargets) -> Result<(), RecorderError> {
        match self.retrieve(target) {
            Some(path) => {
                File::create_new(path)
                    .map_err(|e| RecorderError::FileCreationError(e.to_string()))?;
                Ok(())
            }
            None => Err(RecorderError::FileNotFound(format!(
                "No file configured for target {:?}",
                target
            ))),
        }
    }

    /// Open a CSV file for appending, returning a thread-safe writer.
    pub fn open(
        &self,
        target: &RecorderTargets,
    ) -> Result<Arc<Mutex<csv::Writer<File>>>, RecorderError> {
        match self.retrieve(target) {
            Some(path) => {
                let writer = csv::Writer::from_writer(
                    OpenOptions::new()
                        .append(true)
                        .open(path)
                        .map_err(|e| RecorderError::FileCreationError(e.to_string()))?,
                );
                Ok(Arc::new(Mutex::new(writer)))
            }
            None => Err(RecorderError::FileNotFound(format!(
                "No file configured for target {:?}",
                target
            ))),
        }
    }

    /// Deactivate recording for a specific target.
    pub fn deactivate(&mut self, target: &RecorderTargets) {
        match target {
            RecorderTargets::XObjective => self.path_pobj = None,
            RecorderTargets::XOptimizer => self.path_popt = None,
            RecorderTargets::Y => self.path_codom = None,
            RecorderTargets::Outcome => self.path_out = None,
            RecorderTargets::Metadata => self.path_info = None,
            RecorderTargets::Other(_) => {}
        }
    }
}

fn write_to(writer: Arc<Mutex<csv::Writer<File>>>, row: Vec<String>) -> Result<(), RecorderError> {
    let mut lwrt = writer.lock().unwrap();
    lwrt.write_record(row)
        .map_err(|e| RecorderError::WriteError(e.to_string()))?;
    lwrt.flush()
        .map_err(|e| RecorderError::WriteError(e.to_string()))?;
    Ok(())
}

impl<P> Recorder<P, ()> for CSVRecorder
where
    P: Problem,
    P::Scp: HasVariables,
    SObj<P>: for<'v> Schema<&'v [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>
        + Recordable<(), Parts = SolParts>,
    SOpt<P>: for<'v> Schema<&'v [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts>
        + Recordable<(), Parts = SolParts>,
{
    fn init(&mut self, problem: &P, _context: &()) -> Result<(), RecorderError> {
        let does_exist = self.config.path_rec.try_exists().unwrap();
        if does_exist {
            Err(RecorderError::FolderCreationError(format!(
                "This folder already exists: {}",
                self.config.path_rec.display()
            )))
        } else if self.config.path.is_file() {
            Err(RecorderError::FolderCreationError(format!(
                "This path points to a file: {}",
                self.config.path_rec.display()
            )))
        } else {
            create_dir_all(&self.config.path_rec)
                .map_err(|e| RecorderError::FolderCreationError(e.to_string()))?;
            self.create(&RecorderTargets::XObjective)?;
            self.create(&RecorderTargets::XOptimizer)?;
            self.create(&RecorderTargets::Metadata)?;
            self.create(&RecorderTargets::Outcome)?;
            self.create(&RecorderTargets::Y)?;

            // Header of objective-side X of solution
            if let Ok(f) = self.open(&RecorderTargets::XObjective) {
                let header = SObj::<P>::columns(&problem.get_searchspace().get_vars(), SolParts::X);
                write_to(f, header)?;
            }
            // Header of optimizer-side X of solution
            if let Ok(f) = self.open(&RecorderTargets::XOptimizer) {
                let header = SOpt::<P>::columns(&problem.get_searchspace().get_vars(), SolParts::X);
                write_to(f, header)?;
            }
            // Header of solution metadata (SolInfo)
            if let Ok(f) = self.open(&RecorderTargets::Metadata) {
                let header =
                    SObj::<P>::columns(&problem.get_searchspace().get_vars(), SolParts::Info);
                write_to(f, header)?;
            }
            // Header of the output file (out.csv)
            if let Ok(f) = self.open(&RecorderTargets::Outcome) {
                let header = IdOut::<P::Id, Out<P>>::columns(&(), ());
                write_to(f, header)?;
            }

            if let Ok(f) = self.open(&RecorderTargets::Y) {
                let header = Computed::<SObj<P>, Out<P>>::columns(
                    &(problem.get_searchspace().get_vars(), problem.get_codomain()),
                    CompParts::Y,
                );
                write_to(f, header)?;
            }
            Result::Ok(())
        }
    }

    fn before_load(&mut self, _problem: &P, _context: &()) -> Result<(), RecorderError> {
        if self
            .config
            .path_rec
            .try_exists()
            .map_err(|e| RecorderError::Other(e.to_string()))?
        {
            if let Some(ppobj) = &self.path_pobj
                && !ppobj
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The `Objective`-side solution recorder file does not exist, {}.",
                    ppobj.display()
                )));
            }

            if let Some(ppopt) = &self.path_popt
                && !ppopt
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The `Optimizer`-side solution recorder file does not exist, {}.",
                    ppopt.display()
                )));
            }

            if let Some(ppinfo) = &self.path_info
                && !ppinfo
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The solution metadata (`Info`) file does not exist, {}.",
                    ppinfo.display()
                )));
            }

            if let Some(ppout) = &self.path_out
                && !ppout
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The `Output` recorder file does not exist, {}.",
                    ppout.display()
                )));
            }

            if let Some(ppcodom) = &self.path_codom
                && !ppcodom
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The `Codomain` recorder file does not exist, {}.",
                    ppcodom.display()
                )));
            }
        } else {
            return Err(RecorderError::FileNotFound(format!(
                "The recorder folder does not exist, {}.",
                self.config.path_rec.display()
            )));
        }
        Ok(())
    }

    fn after_load(&mut self, _problem: &P, _context: &()) -> Result<(), RecorderError> {
        if self
            .config
            .path_rec
            .try_exists()
            .map_err(|e| RecorderError::Other(e.to_string()))?
        {
            if let Some(ppobj) = &self.path_pobj
                && !ppobj
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The `Objective`-side solution recorder file does not exist, {}.",
                    ppobj.display()
                )));
            }

            if let Some(ppopt) = &self.path_popt
                && !ppopt
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The `Optimizer`-side solution recorder file does not exist, {}.",
                    ppopt.display()
                )));
            }

            if let Some(ppinfo) = &self.path_info
                && !ppinfo
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The solution metadata (`Info`) file does not exist, {}.",
                    ppinfo.display()
                )));
            }

            if let Some(ppout) = &self.path_out
                && !ppout
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The `Output` recorder file does not exist, {}.",
                    ppout.display()
                )));
            }

            if let Some(ppcodom) = &self.path_codom
                && !ppcodom
                    .try_exists()
                    .map_err(|e| RecorderError::Other(e.to_string()))?
            {
                return Err(RecorderError::FileNotFound(format!(
                    "The `Codomain` recorder file does not exist, {}.",
                    ppcodom.display()
                )));
            }
        } else {
            return Err(RecorderError::FileNotFound(format!(
                "The recorder folder does not exist, {}.",
                self.config.path_rec.display()
            )));
        }
        Ok(())
    }

    fn record<T, CR, CS>(
        &self,
        target: &RecorderTargets,
        object: &T,
        contexts: (&CR, &CS),
        part: <T as Recordable<CR>>::Parts,
    ) -> Result<(), RecorderError>
    where
        T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>,
    {
        if let Ok(f) = self.open(target) {
            let row = object.record(contexts.0, part);
            return write_to(f, row);
        }
        Ok(())
    }

    fn vec_record<I, T, CR, CS>(
        &self,
        target: &RecorderTargets,
        object: I,
        contexts: (&CR, &CS),
        part: <T as Recordable<CR>>::Parts,
    ) -> Result<(), RecorderError>
    where
        T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>,
        I: IntoIterator<Item = T>,
    {
        if let Ok(f) = self.open(target) {
            object.into_iter().try_for_each(|obj| {
                let row = obj.record(contexts.0, part.clone());
                write_to(f.clone(), row)
            })?
        }
        Ok(())
    }
}

// impl<P, O, S, C, E> Recorder<ThrExperiment<P, O, S, CSVRecorder, C, E>> for CSVRecorder
// where
//     P: Problem<Info = O::Info>,
//     P::Scp: HasVariables,
//     SObj<P>: for <'v> Schema<&'v [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts> + Recordable<(), Parts = SolParts>,
//     SOpt<P>: for <'v> Schema<&'v [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts> + Recordable<(), Parts = SolParts>,
//     O: Optimizer<P>,
//     S: Stop,
//     C: Checkpointer<ThrExperiment<P, O, S, CSVRecorder, C, E>>,
// {
//    fn init(&mut self, runable: &ThrExperiment<P, O, S, CSVRecorder, C, E>) -> Result<(), RecorderError> {
//         let does_exist = self.config.path_rec.try_exists().unwrap();
//         if does_exist {
//             return Err(RecorderError::FolderCreationError(format!("This folder already exists: {}", self.config.path_rec.display())));
//         } else if self.config.path.is_file() {
//             return Err(RecorderError::FolderCreationError(format!("This path points to a file: {}", self.config.path_rec.display())));
//         } else {
//             create_dir_all(&self.config.path_rec).map_err(|e| RecorderError::FolderCreationError(e.to_string()))?;
//             self.create(&RecorderTargets::XObjective);
//             self.create(&RecorderTargets::XOptimizer);
//             self.create(&RecorderTargets::Metadata);
//             self.create(&RecorderTargets::Outcome);
//             self.create(&RecorderTargets::Y);

//             // Header of objective-side X of solution
//             if let Ok(f) = self.open(&RecorderTargets::XObjective) {
//                 let header = SObj::<P>::columns(&runable.searchspace.get_vars(), SolParts::X);
//                 write_to(f, header)?;

//             }
//             // Header of optimizer-side X of solution
//             if let Ok(f) = self.open(&RecorderTargets::XOptimizer) {
//                 let header = SOpt::<P>::columns(&runable.searchspace.get_vars(), SolParts::X);
//                 write_to(f, header)?;
//             }
//             // Header of solution metadata (SolInfo)
//             if let Ok(f) = self.open(&RecorderTargets::Metadata) {
//                 let header = SObj::<P>::columns(&runable.searchspace.get_vars(), SolParts::Info);
//                 write_to(f, header)?;
//             }
//             // Header of the output file (out.csv)
//             if let Ok(f) = self.open(&RecorderTargets::Outcome) {
//                 let header = Out::<P>::columns(&(), ());
//                 write_to(f, header)?;
//             }

//             if let Ok(f) = self.open(&RecorderTargets::Y) {
//                 let header = Computed::<SObj<P>, Out<P>>::columns(&(&runable.searchspace.get_vars(), &runable.codomain), CompParts::Y);
//                 write_to(f, header)?;
//             }
//             Result::Ok(())
//         }
//     }

//     fn after_load(&mut self, _runable: &ThrExperiment<P, O, S, CSVRecorder, C, E>) -> Result<(), RecorderError> {
//         if self.config.path_rec.try_exists().map_err(|e| RecorderError::Other(e.to_string()))? {
//             if let Some(ppobj) = &self.path_pobj
//                 && !ppobj.try_exists().map_err(|e| RecorderError::Other(e.to_string()))?
//             {
//                 return Err(RecorderError::FileNotFound(format!("The `Objective`-side solution recorder file does not exist, {}.", ppobj.display())));
//             }

//             if let Some(ppopt) = &self.path_popt
//                 && !ppopt.try_exists().map_err(|e| RecorderError::Other(e.to_string()))?
//             {
//                 return Err(RecorderError::FileNotFound(format!("The `Optimizer`-side solution recorder file does not exist, {}.", ppopt.display())));
//             }

//             if let Some(ppinfo) = &self.path_info
//                 && !ppinfo.try_exists().map_err(|e| RecorderError::Other(e.to_string()))?
//             {
//                 return Err(RecorderError::FileNotFound(format!("The solution metadata (`Info`) file does not exist, {}.", ppinfo.display())));
//             }

//             if let Some(ppout) = &self.path_out
//                 && !ppout.try_exists().map_err(|e| RecorderError::Other(e.to_string()))?
//             {
//                 return Err(RecorderError::FileNotFound(format!("The `Output` recorder file does not exist, {}.", ppout.display())));
//             }

//             if let Some(ppcodom) = &self.path_codom && !ppcodom.try_exists().map_err(|e| RecorderError::Other(e.to_string()))? {
//                 return Err(RecorderError::FileNotFound(format!("The `Codomain` recorder file does not exist, {}.", ppcodom.display())));
//             }
//         } else {
//             return Err(RecorderError::FileNotFound(format!("The recorder folder does not exist, {}.", self.config.path_rec.display())));
//         }
//         Ok(())
//     }

//     fn record<T, CR, CS>(&self, target: &RecorderTargets, object: &T, contexts: (&CR, &CS), part: <T as Recordable<CR>>::Parts) -> Result<(), RecorderError>
//     where
//         T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>
//     {
//         if let Ok(f) = self.open(target) {
//             let row = object.record(contexts.0, part);
//             return write_to(f, row)
//         }
//         Ok(())
//     }

//     fn vec_record<I, T, CR, CS>(&self, target: &RecorderTargets, object: I, contexts: (&CR, &CS), part: <T as Recordable<CR>>::Parts) -> Result<(), RecorderError>
//     where
//         T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>,
//         I: IntoIterator<Item = T>
//     {
//         if let Ok(f) = self.open(target) {
//             object.into_iter().try_for_each(|obj| {
//                 let row = obj.record(contexts.0, part.clone());
//                 write_to(f.clone(), row)
//             })?
//         }
//         Ok(())
//     }
// }

// #[cfg(feature = "mpi")]
// impl<'a, P, O, S, C, E> Recorder<MPIExperiment<'a, P, O, S, CSVRecorder, C, E>> for CSVRecorder
// where
//     P: Problem,
//     P::Scp: HasVariables,
//     SObj<P>: for <'v> Schema<&'v [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts> + Recordable<(), Parts = SolParts>,
//     SOpt<P>: for <'v> Schema<&'v [Var<Obj<P>, TrueOpt<P>>], Parts = SolParts> + Recordable<(), Parts = SolParts>,
//     O: Optimizer<P>,
//     S: Stop,
//     C: Checkpointer<MPIExperiment<'a, P, O, S, CSVRecorder, C, E>>,
// {
//     fn init(&mut self, runable: &MPIExperiment<'a, P, O, S, CSVRecorder, C, E>) -> Result<(), RecorderError> {
//         let does_exist = self.config.path_rec.try_exists().unwrap();
//         if does_exist {
//             return Err(RecorderError::FolderCreationError(format!("This folder already exists: {}", self.config.path_rec.display())));
//         } else if self.config.path.is_file() {
//             return Err(RecorderError::FolderCreationError(format!("This path points to a file: {}", self.config.path_rec.display())));
//         } else {
//             create_dir_all(&self.config.path_rec).map_err(|e| RecorderError::FolderCreationError(e.to_string()))?;
//             self.create(&RecorderTargets::XObjective);
//             self.create(&RecorderTargets::XOptimizer);
//             self.create(&RecorderTargets::Metadata);
//             self.create(&RecorderTargets::Outcome);
//             self.create(&RecorderTargets::Y);

//             // Header of objective-side X of solution
//             if let Ok(f) = self.open(&RecorderTargets::XObjective) {
//                 let header = SObj::<P>::columns(&runable.searchspace.get_vars(), SolParts::X);
//                 write_to(f, header)?;

//             }
//             // Header of optimizer-side X of solution
//             if let Ok(f) = self.open(&RecorderTargets::XOptimizer) {
//                 let header = SOpt::<P>::columns(&runable.searchspace.get_vars(), SolParts::X);
//                 write_to(f, header)?;
//             }
//             // Header of solution metadata (SolInfo)
//             if let Ok(f) = self.open(&RecorderTargets::Metadata) {
//                 let header = SObj::<P>::columns(&runable.searchspace.get_vars(), SolParts::Info);
//                 write_to(f, header)?;
//             }
//             // Header of the output file (out.csv)
//             if let Ok(f) = self.open(&RecorderTargets::Outcome) {
//                 let header = Out::<P>::columns(&(), ());
//                 write_to(f, header)?;
//             }

//             if let Ok(f) = self.open(&RecorderTargets::Y) {
//                 let header = Computed::<SObj<P>, Out<P>>::columns(&(&runable.searchspace.get_vars(), &runable.codomain), CompParts::Y);
//                 write_to(f, header)?;
//             }
//             Result::Ok(())
//         }
//     }

//     fn after_load(&mut self, _runable: &MPIExperiment<'a, P, O, S, CSVRecorder, C, E>) -> Result<(), RecorderError> {
//         if self.config.path_rec.try_exists().map_err(|e| RecorderError::Other(e.to_string()))? {
//             if let Some(ppobj) = &self.path_pobj
//                 && !ppobj.try_exists().map_err(|e| RecorderError::Other(e.to_string()))?
//             {
//                 return Err(RecorderError::FileNotFound(format!("The `Objective`-side solution recorder file does not exist, {}.", ppobj.display())));
//             }

//             if let Some(ppopt) = &self.path_popt
//                 && !ppopt.try_exists().map_err(|e| RecorderError::Other(e.to_string()))?
//             {
//                 return Err(RecorderError::FileNotFound(format!("The `Optimizer`-side solution recorder file does not exist, {}.", ppopt.display())));
//             }

//             if let Some(ppinfo) = &self.path_info
//                 && !ppinfo.try_exists().map_err(|e| RecorderError::Other(e.to_string()))?
//             {
//                 return Err(RecorderError::FileNotFound(format!("The solution metadata (`Info`) file does not exist, {}.", ppinfo.display())));
//             }

//             if let Some(ppout) = &self.path_out
//                 && !ppout.try_exists().map_err(|e| RecorderError::Other(e.to_string()))?
//             {
//                 return Err(RecorderError::FileNotFound(format!("The `Output` recorder file does not exist, {}.", ppout.display())));
//             }

//             if let Some(ppcodom) = &self.path_codom && !ppcodom.try_exists().map_err(|e| RecorderError::Other(e.to_string()))? {
//                 return Err(RecorderError::FileNotFound(format!("The `Codomain` recorder file does not exist, {}.", ppcodom.display())));
//             }
//         } else {
//             return Err(RecorderError::FileNotFound(format!("The recorder folder does not exist, {}.", self.config.path_rec.display())));
//         }
//         Ok(())
//     }

//     fn record<T, CR, CS>(&self, target: &RecorderTargets, object: &T, contexts: (&CR, &CS), part: <T as Recordable<CR>>::Parts) -> Result<(), RecorderError>
//     where
//         T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>
//     {
//         if let Ok(f) = self.open(target) {
//             let row = object.record(contexts.0, part);
//             return write_to(f, row)
//         }
//         Ok(())
//     }

//     fn vec_record<I, T, CR, CS>(&self, target: &RecorderTargets, object: I, contexts: (&CR, &CS), part: <T as Recordable<CR>>::Parts) -> Result<(), RecorderError>
//     where
//         T: Recordable<CR> + Schema<CS, Parts = <T as Recordable<CR>>::Parts>,
//         I: IntoIterator<Item = T>
//     {
//         if let Ok(f) = self.open(target) {
//             object.into_iter().try_for_each(|obj| {
//                 let row = obj.record(contexts.0, part.clone());
//                 write_to(f.clone(), row)
//             })?
//         }
//         Ok(())
//     }
// }
