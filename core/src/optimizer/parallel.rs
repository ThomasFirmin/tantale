use crate::{Codomain, Domain, Optimizer, Outcome, ParSId, Searchspace, optimizer::opt::BatchType};

pub enum AlgoLvl {
    Sequential,
    MultiThread,
    MultiProcess,
    Distributed,
}

pub enum IterLvl {
    Sync,
    Async,
    Hybrid,
    Workstealing,
}

/// A parallel [`Optimizer`] with multi-processing.
pub trait DistributedOptimizer<Obj, Opt, Cod, Out, Scp, BType>:
    Optimizer<ParSId, Obj, Opt, Cod, Out, Scp, BType>
where
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<ParSId, Obj, Opt, Self::SInfo>,
    BType: BatchType<ParSId,Obj,Opt,Self::SInfo,Self::Info>,
{
    fn interact(&self);
    fn update(&self);
}
