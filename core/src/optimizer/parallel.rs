use crate::{Codomain, Domain, Optimizer, Outcome, ParSId, Searchspace};

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
pub trait DistributedOptimizer<Obj, Opt, Cod, Out, Scp>:
    Optimizer<ParSId, Obj, Opt, Cod, Out, Scp>
where
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<ParSId, Obj, Opt, Self::SInfo>,
{
    fn interact(&self);
    fn update(&self);
}
