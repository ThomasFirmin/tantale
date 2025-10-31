use crate::domain::onto::OntoDom;
use crate::{Codomain, Id, Outcome, Partial, Searchspace, SolInfo};

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature="mpi")]
use crate::experiment::mpi::tools::MPIProcess;

pub trait SaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
{
    fn init(&mut self);
    fn after_load(&mut self);
    fn get_sp(&self)->Arc<Scp>;
    fn get_cod(&self)->Arc<Cod>;
}

#[cfg(feature="mpi")]
pub trait DistSaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>: SaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
{
    fn init(&mut self, proc:&MPIProcess);
    fn after_load(&mut self, proc:&MPIProcess);
}

/// Describes a folders and files hierarchy for file-based [`Recorder`] and [`Checkpointer`].
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
///   * state_param.mp    (Various global parameters such as the [`Id`] or experiment identifier.)
pub struct FolderConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
{
    pub path: PathBuf,
    pub scp: Arc<Scp>,
    pub cod: Arc<Cod>,
    pub path_rec: PathBuf,
    pub path_check: PathBuf,
    pub is_dist:bool,
    psol: PhantomData<PSol>,
    sid: PhantomData<SolId>,
    obj: PhantomData<Obj>,
    opt: PhantomData<Opt>,
    sinfo: PhantomData<SInfo>,
    out: PhantomData<Out>,
}

impl<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out> FolderConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo
{
    pub fn new(
        path: &str,
        scp: Arc<Scp>,
        cod: Arc<Cod>,
    ) -> Arc<Self> {
        let path = PathBuf::from(path);
        let path_rec = path.join(Path::new("recorder"));
        let path_check = path.join(Path::new("checkpointer"));
        Arc::new(
            FolderConfig {
                path,
                scp,
                cod,
                path_rec,
                path_check,
                is_dist: false,
                psol: PhantomData,
                sid: PhantomData,
                obj: PhantomData,
                opt: PhantomData,
                sinfo: PhantomData,
                out: PhantomData
        })
    }

    #[cfg(feature="mpi")]
    pub fn new(
        path: &str,
        scp: Arc<Scp>,
        cod: Arc<Cod>,
        proc:&MPIProcess,
    ) -> Arc<Self> {
        let path = PathBuf::from(path);
        let path_rec = path.join(Path::new(format!("recorder_rank{}",proc.rank)));
        let path_check = path.join(Path::new(format!("checkpointer_rank{}",proc.rank)));
        Arc::new(
            FolderConfig {
                path,
                scp,
                cod,
                path_rec,
                path_check,
                is_dist: true,
                psol: PhantomData,
                sid: PhantomData,
                obj: PhantomData,
                opt: PhantomData,
                sinfo: PhantomData,
                out: PhantomData
        })
    }
}
impl<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out> SaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out> for FolderConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo
{
    fn init(&mut self) {}
    
    fn after_load(&mut self) {}

    fn get_sp(&self)->Arc<Scp> {
        self.scp.clone()
    }

    fn get_cod(&self)->Arc<Cod> {
        self.cod.clone()
    }
}

impl<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out> DistSaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out> for FolderConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
{
    fn init(&mut self, proc:&MPIProcess) {
        todo!()
    }

    fn after_load(&mut self, proc:&MPIProcess) {
        todo!()
    }
}


pub struct NoConfig;
impl<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out> SaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out>for NoConfig
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo
{
    fn init(&mut self) {}
    
    fn after_load(&mut self) {}

    fn get_sp(&self)->Arc<Scp> {
        unreachable!("The Searchspace should not be accessed with NoConfig")
    }

    fn get_cod(&self)->Arc<Cod> {
        unreachable!("The Codomain should not be accessed with NoConfig")
    }
    
}

impl<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out> DistSaverConfig<PSol, Scp, SolId, Obj, Opt, SInfo, Cod, Out> for NoConfig
where
    PSol: Partial<SolId,Obj,SInfo>,
    PSol::Twin<Opt> : Partial<SolId,Opt,SInfo, Twin<Obj> = PSol>,
    Scp: Searchspace<PSol, SolId, Obj, Opt, SInfo>,
    SolId: Id,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
{
    fn init(&mut self, proc:&MPIProcess) {
        todo!()
    }

    fn after_load(&mut self, proc:&MPIProcess) {
        todo!()
    }
}
