use tantale_core::{
    domain::Domain,
    objective::{codomain::SingleCodomain, outcome::Outcome},
    optimizer::{opt::SolPairs, EmptyInfo, OptInfo, OptState, Optimizer},
    saver::CSVWritable,
    searchspace::Searchspace,
    solution::{Partial, SId},
    ArcVecArc,
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
impl CSVWritable<()> for RSInfo {
    fn header(&self) -> Vec<String> {
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

fn rs_iter<PObj, POpt, Obj, Opt, Scp>(
    opt: &mut RandomSearch,
    sp: Arc<Scp>,
) -> (ArcVecArc<PObj>, ArcVecArc<POpt>, RSInfo)
where
    PObj: Partial<SId, Obj, EmptyInfo>,
    POpt: Partial<SId, Opt, EmptyInfo>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Scp: Searchspace<SId, PObj, POpt, Obj, Opt, EmptyInfo>,
{
    let samples = sp.vec_sample_obj(Some(&mut opt.1), opt.0.batch, Arc::new(EmptyInfo {}));
    let opt_samples = sp.vec_onto_opt(samples.clone());
    let info = RSInfo {
        iteration: opt.0.iteration,
    };
    opt.0.iteration += 1;
    (samples, opt_samples, info)
}

impl<PObj, POpt, Obj, Opt, Out, Scp>
    Optimizer<SId, PObj, POpt, Obj, Opt, EmptyInfo, SingleCodomain<Out>, Out, Scp, RSInfo, RSState>
    for RandomSearch
where
    PObj: Partial<SId, Obj, EmptyInfo>,
    POpt: Partial<SId, Opt, EmptyInfo>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Scp: Searchspace<SId, PObj, POpt, Obj, Opt, EmptyInfo>,
{
    fn init(&mut self) {}

    fn first_step(&mut self, sp: Arc<Scp>) -> (ArcVecArc<PObj>, ArcVecArc<POpt>, RSInfo) {
        rs_iter::<PObj, POpt, Obj, Opt, Scp>(self, sp.clone())
    }

    fn step(
        &mut self,
        _x: SolPairs<SId, PObj, Obj, POpt, Opt, SingleCodomain<Out>, Out, EmptyInfo>,
        sp: Arc<Scp>,
    ) -> (ArcVecArc<PObj>, ArcVecArc<POpt>, RSInfo) {
        rs_iter::<PObj, POpt, Obj, Opt, Scp>(self, sp.clone())
    }

    fn get_state(&mut self) -> &RSState {
        &self.0
    }
}
