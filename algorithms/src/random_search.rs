use tantale_core::{
    BasePartial, Criteria, Objective, Stop, domain::onto::OntoDom, experiment::{BatchEvaluator, Runable, SyncExperiment, ThrBatchEvaluator}, objective::{codomain::SingleCodomain, outcome::Outcome}, optimizer::{
        CBType, EmptyInfo, OptInfo, OptState, opt::{MonoOptimizer, Optimizer, ThrOptimizer}
    }, saver::{CSVWritable, Saver}, searchspace::Searchspace, solution::{Batch, SId}
};
#[cfg(feature = "mpi")]
use tantale_core::{
    experiment::DistRunable, optimizer::opt::DistOptimizer, saver::DistributedSaver,
};

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
pub struct RSState {
    pub batch: usize,
    pub iteration: usize,
}
impl OptState for RSState {}

#[derive(Serialize, Deserialize, Debug)]
pub struct RSInfo {
    pub iteration: usize,
}
impl OptInfo for RSInfo {}
impl CSVWritable<(), ()> for RSInfo {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("iteration")])
    }
    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.iteration.to_string()])
    }
}

pub struct RandomSearch(pub RSState, ThreadRng);
impl RandomSearch {
    pub fn new(batch: usize) -> Self {
        let rng = rand::rng();
        RandomSearch(
            RSState {
                batch,
                iteration: 0,
            },
            rng,
        )
    }
    pub fn codomain<Out: Outcome>(extractor: Criteria<Out>) -> SingleCodomain<Out> {
        SingleCodomain {
            y_criteria: extractor,
        }
    }
}

fn rs_iter<Obj, Opt, Scp>(
    opt: &mut RandomSearch,
    sp: Arc<Scp>,
) -> Batch<BasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo, RSInfo>
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Scp: Searchspace<BasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo>,
{
    let samples = sp.vec_sample_obj(Some(&mut opt.1), opt.0.batch, Arc::new(EmptyInfo {}));
    let opt_samples = sp.vec_onto_opt(samples.clone());
    let info = Arc::new(RSInfo {
        iteration: opt.0.iteration,
    });
    opt.0.iteration += 1;
    Batch::new(samples, opt_samples, info)
}

impl<Obj, Opt, Out, Scp> Optimizer<SId, Obj, Opt, Out, Scp> for RandomSearch
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<BasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo>,
{
    type Sol = BasePartial<SId, Obj, EmptyInfo>;
    type BType = Batch<Self::Sol, SId, Obj, Opt, EmptyInfo, RSInfo>;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;
    type State = RSState;
    type FnWrap = Objective<Obj, Self::Cod, Out>;

    fn init(&mut self) {}

    fn first_step(&mut self, sp: Arc<Scp>) -> Self::BType {
        rs_iter::<Obj, Opt, Scp>(self, sp.clone())
    }

    fn step(
        &mut self,
        _x: CBType<Self, SId, Obj, Opt, Out, Scp>,
        sp: Arc<Scp>,
    ) -> Batch<Self::Sol, SId, Obj, Opt, EmptyInfo, RSInfo> {
        rs_iter::<Obj, Opt, Scp>(self, sp.clone())
    }

    fn get_state(&mut self) -> &RSState {
        &self.0
    }

    fn from_state(state: RSState) -> Self {
        RandomSearch(state, rand::rng())
    }
}

impl<Obj, Opt, Out, Scp> MonoOptimizer<SId, Obj, Opt, Out, Scp> for RandomSearch
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<BasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo>,
{
    type Eval<St: Stop> = BatchEvaluator<Self::Sol, SId, Obj, Opt, Self::SInfo, Self::Info>;
    type Exp<St, Sv>
        = SyncExperiment<SId, Self::Eval<St>, Scp, Self, St, Sv, Obj, Opt, Out>
    where
        St: Stop,
        Sv: Saver<SId, St, Obj, Opt, Out, Scp, Self, Self::Eval<St>>;

    fn get_mono<St, Sv>(
        searchspace: Scp,
        objective: Self::FnWrap,
        optimizer: Self,
        stop: St,
        saver: Sv,
    ) -> Self::Exp<St, Sv>
    where
        St: Stop,
        Sv: Saver<SId, St, Obj, Opt, Out, Scp, Self, Self::Eval<St>>,
    {
        SyncExperiment::new(searchspace, objective, optimizer, stop, saver)
    }
}

impl<Obj, Opt, Out, Scp> ThrOptimizer<SId, Obj, Opt, Out, Scp> for RandomSearch
where
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
    Out: Outcome + Send + Sync,
    Scp: Searchspace<BasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo> + Send + Sync,
{
    type Eval<St: Stop + Send + Sync> =
        ThrBatchEvaluator<Self::Sol, SId, Obj, Opt, Self::SInfo, Self::Info>;
    type Exp<St, Sv>
        = SyncExperiment<SId, Self::Eval<St>, Scp, Self, St, Sv, Obj, Opt, Out>
    where
        St: Stop + Send + Sync,
        Sv: Saver<SId, St, Obj, Opt, Out, Scp, Self, Self::Eval<St>> + Send + Sync;

    fn get_threaded<St, Sv>(
        searchspace: Scp,
        objective: Self::FnWrap,
        optimizer: Self,
        stop: St,
        saver: Sv,
    ) -> Self::Exp<St, Sv>
    where
        St: Stop + Send + Sync,
        Sv: Saver<SId, St, Obj, Opt, Out, Scp, Self, Self::Eval<St>> + Send + Sync,
    {
        SyncExperiment::new(searchspace, objective, optimizer, stop, saver)
    }
}

#[cfg(feature = "mpi")]
impl<Obj, Opt, Out, Scp> DistOptimizer<SId, Obj, Opt, Out, Scp> for RandomSearch
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: Outcome,
    Scp: Searchspace<BasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo>,
{
    type Eval<St: Stop> = BatchEvaluator<Self::Sol, SId, Obj, Opt, Self::SInfo, Self::Info>;
    type Exp<St, Sv>
        = SyncExperiment<SId, Self::Eval<St>, Scp, Self, St, Sv, Obj, Opt, Out>
    where
        St: Stop,
        Sv: DistributedSaver<SId, St, Obj, Opt, Out, Scp, Self, Self::Eval<St>>;

    fn get_distributed<St, Sv>(
        proc: &tantale_core::experiment::mpi::tools::MPIProcess,
        searchspace: Scp,
        objective: Self::FnWrap,
        optimizer: Self,
        stop: St,
        saver: Sv,
    ) -> Self::Exp<St, Sv>
    where
        St: Stop,
        Sv: DistributedSaver<SId, St, Obj, Opt, Out, Scp, Self, Self::Eval<St>>,
    {
        SyncExperiment::new_dist(proc, searchspace, objective, optimizer, stop, saver)
    }
}
