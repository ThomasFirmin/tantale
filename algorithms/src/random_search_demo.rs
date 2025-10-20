use rand::prelude::ThreadRng; // Random Number Generator
use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[cfg(feature = "mpi")]
use tantale_core::{experiment::DistRunable, optimizer::DistOptimizer, saver::DistributedSaver};
use tantale_core::{
    experiment::{BatchEvaluator, Runable, SyncExperiment, ThrBatchEvaluator},
    optimizer::{
        opt::{MonoOptimizer, ThrOptimizer},
        OptState,
    },
    saver::{CSVWritable, Saver},
    solution::Batch,
    BasePartial, Domain, Objective, OptInfo, Optimizer, Outcome, Real, SId, Searchspace,
    SingleCodomain, SolInfo, Stop,
}; // Serialization + Deserialization // Multi-Threading

#[derive(Serialize, Deserialize)]
pub struct RSState {
    pub iteration: usize,
}
impl OptState for RSState {}

#[derive(Debug, Serialize, Deserialize)]
pub struct RSSInfo {
    pub predicted_accuracy: f64,
}
impl SolInfo for RSSInfo {}
impl CSVWritable<(), ()> for RSSInfo {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("guessed_accuracy")])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.predicted_accuracy.to_string()])
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RSGlobalInfo {
    pub iteration: usize,
}
impl OptInfo for RSGlobalInfo {}
impl CSVWritable<(), ()> for RSGlobalInfo {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("iteration")])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.iteration.to_string()])
    }
}

pub struct RandomSearch(RSState, ThreadRng);

// Mutator, Selection, Crossover
//Optimizer
impl<Obj, Opt, Out, Scp> Optimizer<SId, Obj, Opt, Out, Scp> for RandomSearch
where
    Obj: Domain,  // Float, Int, Nat, Mixed
    Opt: Domain,  // Float, Int, Nat, Mixed
    Out: Outcome, // f(x) => Out
    Scp: Searchspace<
        BasePartial<SId, Obj, RSSInfo>, // [Obj] => [1.5,5,...]
        SId,
        Obj,
        Opt,
        RSSInfo,
    >,
{
    // Kind of solution
    type Sol = BasePartial<SId, Obj, RSSInfo>; // [Obj] => [1.5,5,...]

    //Kind of Batch
    type BType = Batch<Self::Sol, SId, Obj, Opt, RSSInfo, RSGlobalInfo>;

    type State = RSState;

    // f ? f(x) => Simple / Stepped
    type FnWrap = Objective<Obj, Self::Cod, Out>;

    // f(x) -> SingleCodomain(Out) -> Y, Y -> scalar (99.9...)
    type Cod = SingleCodomain<Out>;

    type SInfo = RSSInfo;

    type Info = RSGlobalInfo;

    fn init(&mut self) {}

    fn first_step(&mut self, sp: Arc<Scp>) -> Self::BType {
        let samples = sp.vec_sample_obj(
            Some(&mut self.1),
            50,
            Arc::new(RSSInfo {
                predicted_accuracy: 50.0,
            }),
        );
        let samples_opt = sp.vec_onto_opt(samples.clone());
        Batch::new(
            samples,
            samples_opt,
            Arc::new(RSGlobalInfo {
                iteration: self.0.iteration,
            }),
        )
    }

    fn step(
        &mut self,
        _x: tantale_core::optimizer::CBType<Self, SId, Obj, Opt, Out, Scp>,
        sp: Arc<Scp>,
    ) -> Self::BType {
        let samples = sp.vec_sample_obj(
            Some(&mut self.1),
            50,
            Arc::new(RSSInfo {
                predicted_accuracy: 50.0,
            }),
        );
        let samples_opt = sp.vec_onto_opt(samples.clone());
        Batch::new(
            samples,
            samples_opt,
            Arc::new(RSGlobalInfo {
                iteration: self.0.iteration,
            }),
        )
    }

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        RandomSearch(state, rand::rng())
    }
}
