use crate::{
    Codomain, Computed, Domain, Id, OptInfo, Outcome, Partial, SolInfo,
    solution::{CompBatch, RawBatch, RawSol}
};
use std::{fmt::Debug,sync::Arc};

pub type PartPair<PSolA,PSolB> = (Arc<PSolA>,Arc<PSolB>);

#[derive(Debug)]
pub struct BatchResults<PSolA,PSolB,SolId, Obj, Opt, Cod, Out, SInfo, Info>
where
    PSolA: Partial<SolId,Obj,SInfo,Twin<Opt>=PSolB>,
    PSolB: Partial<SolId,Opt,SInfo,Twin<Obj>=PSolA>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub rbatch: RawBatch<PSolA,PSolB,SolId,Obj,Opt,SInfo,Info,Out>,
    pub cbatch: CompBatch<PSolA,PSolB,SolId,Obj,Opt,SInfo,Info,Cod,Out>,
}

impl<PSolA,PSolB,SolId, Obj, Opt, Cod, Out, SInfo, Info> BatchResults<PSolA,PSolB,SolId, Obj, Opt, Cod, Out, SInfo, Info>
where
    PSolA: Partial<SolId,Obj,SInfo,Twin<Opt>=PSolB>,
    PSolB: Partial<SolId,Opt,SInfo,Twin<Obj>=PSolA>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub fn new(info:Arc<Info>) -> Self {
        BatchResults {
            rbatch: RawBatch::empty(info.clone()),
            cbatch: CompBatch::empty(info),
        }
    }

    pub fn add(&mut self, pair:PartPair<PSolA,PSolB>,out:Arc<Out>,y:Arc<Cod::TypeCodom>){
        self.rbatch.add(
                Arc::new(RawSol::new(pair.0.clone(), out.clone())),
                Arc::new(RawSol::new(pair.1.clone(), out))
            );
        self.cbatch.add(
            Arc::new(Computed::new(pair.0, y.clone())),
            Arc::new(Computed::new(pair.1, y))
        );
    }
}