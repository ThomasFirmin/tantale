use crate::{
    solution::{CompBatch, RawBatch, RawSol},
    Codomain, Computed, Domain, Id, OptInfo, Outcome, Partial, SolInfo,
};
use std::sync::Arc;

pub type PartPair<PSolA, PSolB> = (Arc<PSolA>, Arc<PSolB>);

#[derive(Debug)]
pub struct BatchResults<PSol, SolId, Obj, Opt, Cod, Out, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub rbatch: RawBatch<PSol, SolId, Obj, Opt, SInfo, Info, Out>,
    pub cbatch: CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>,
}

impl<PSol, SolId, Obj, Opt, Cod, Out, SInfo, Info>
    BatchResults<PSol, SolId, Obj, Opt, Cod, Out, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub fn new(info: Arc<Info>) -> Self {
        BatchResults {
            rbatch: RawBatch::empty(info.clone()),
            cbatch: CompBatch::empty(info),
        }
    }

    pub fn add(
        &mut self,
        pair: PartPair<PSol, PSol::Twin<Opt>>,
        out: Arc<Out>,
        y: Arc<Cod::TypeCodom>,
    ) {
        self.rbatch.add(
            Arc::new(RawSol::new(pair.0.clone(), out.clone())),
            Arc::new(RawSol::new(pair.1.clone(), out)),
        );
        self.cbatch.add(
            Arc::new(Computed::new(pair.0, y.clone())),
            Arc::new(Computed::new(pair.1, y)),
        );
    }
}
