use crate::{
    Codomain, Computed, Domain, Id, OptInfo, Outcome, Partial, SolInfo,
    solution::{CompBatch, RawBatch, RawSol}
};
use std::{fmt::Debug,sync::Arc};

pub type PartPair<SolId,Obj,Opt,SInfo> = (Arc<Partial<SolId, Obj, SInfo>>,Arc<Partial<SolId, Opt, SInfo>>);

#[derive(Debug)]
pub struct BatchResults<SolId, Obj, Opt, Cod, Out, SInfo, Info>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub rbatch: RawBatch<SolId,Obj,Opt,SInfo,Info,Out>,
    pub cbatch: CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>,
}

impl<SolId, Obj, Opt, Cod, Out, SInfo, Info> BatchResults<SolId, Obj, Opt, Cod, Out, SInfo, Info>
where
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

    pub fn add(&mut self, pair:PartPair<SolId,Obj,Opt,SInfo>,out:Arc<Out>,y:Arc<Cod::TypeCodom>){
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