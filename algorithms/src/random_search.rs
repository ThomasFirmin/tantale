use tantale_core::{
    domain::Domain,
    objective::{codomain::SingleCodomain, outcome::Outcome},
    optimizer::{opt::{OptOutput, ArcVecArc}, EmptyInfo, OptInfo, OptState, Optimizer},
    saver::CSVWritable,
    searchspace::Searchspace,
    solution::{SId,Computed},
};

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

#[derive(Serialize, Deserialize)]
pub struct RSState {
    pub batch: usize,
    pub iteration: usize,
}
impl OptState for RSState {}

pub struct RSInfo {
    pub iteration: usize,
}
impl OptInfo for RSInfo {}
impl CSVWritable<(), ()> for RSInfo {
    fn header(_elem:&()) -> Vec<String> {
        Vec::from([String::from("iteration")])
    }
    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.iteration.to_string()])
    }
}

pub struct RandomSearch(RSState, ThreadRng);
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
}

fn rs_iter<Obj, Opt, Scp>(
    opt: &mut RandomSearch,
    sp: Arc<Scp>,
) -> OptOutput<SId, Obj, Opt, EmptyInfo, RSInfo>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Scp: Searchspace<SId, Obj, Opt, EmptyInfo>,
{
    let samples = sp.vec_sample_obj(Some(&mut opt.1), opt.0.batch, Arc::new(EmptyInfo {}));
    let opt_samples = sp.vec_onto_opt(samples.clone());
    let info = Arc::new(RSInfo {
        iteration: opt.0.iteration,
    });
    opt.0.iteration += 1;
    (samples, opt_samples, info)
}

impl<Obj, Opt, Out, Scp>
    Optimizer<SId, Obj, Opt, EmptyInfo, SingleCodomain<Out>, Out, Scp, RSInfo, RSState>
    for RandomSearch
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Scp: Searchspace<SId, Obj, Opt, EmptyInfo>,
{
    fn init(&mut self) {}

    fn first_step(&mut self, sp: Arc<Scp>) -> OptOutput<SId, Obj, Opt, EmptyInfo, RSInfo> {
        rs_iter::<Obj, Opt, Scp>(self, sp.clone())
    }

    fn step(
        &mut self,
        _x: ArcVecArc<Computed<SId, Opt, SingleCodomain<Out>, Out, EmptyInfo>>,
        sp: Arc<Scp>,
    ) -> OptOutput<SId, Obj, Opt, EmptyInfo, RSInfo> {
        rs_iter::<Obj, Opt, Scp>(self, sp.clone())
    }

    fn get_state(&mut self) -> &RSState {
        &self.0
    }
}
