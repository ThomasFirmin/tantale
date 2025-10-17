use tantale_core::{
    BasePartial, Criteria, Objective, domain::Domain, objective::{codomain::SingleCodomain, outcome::Outcome}, optimizer::{
        CBType, EmptyInfo, OptInfo, OptState, opt::{IterMode, Optimizer}
    }, saver::CSVWritable, searchspace::Searchspace, solution::{Batch, SId}
};

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
pub struct RSState {
    pub batch: usize,
    pub iteration: usize,
    iter_lvl: IterMode,
}
impl OptState for RSState
{
    fn set_iter_lvl(&mut self, mode: IterMode){
        self.iter_lvl = mode;
    }

    fn get_iter_lvl(&self) -> &IterMode{
        &self.iter_lvl
    }
}

#[derive(Serialize,Deserialize,Debug)]
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
                iter_lvl: IterMode::Monothreaded,
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

fn rs_iter<Obj,Opt,Scp>(
    opt: &mut RandomSearch,
    sp: Arc<Scp>,
) -> Batch<BasePartial<SId,Obj,EmptyInfo>,SId,Obj,Opt,EmptyInfo,RSInfo>
where
    Obj: Domain,
    Opt: Domain,
    Scp: Searchspace<BasePartial<SId,Obj,EmptyInfo>,SId,Obj,Opt,EmptyInfo>,
{
    let samples = sp.vec_sample_obj(Some(&mut opt.1), opt.0.batch, Arc::new(EmptyInfo {}));
    let opt_samples = sp.vec_onto_opt(samples.clone());
    let info = Arc::new(RSInfo {
        iteration: opt.0.iteration,
    });
    opt.0.iteration += 1;
    Batch::new(samples, opt_samples, info)
}

impl<Obj, Opt, Out, Scp>
    Optimizer<SId, Obj, Opt, Out, Scp>
    for RandomSearch
where
    Obj: Domain,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<BasePartial<SId,Obj,EmptyInfo>,SId, Obj, Opt, EmptyInfo>,
{
    type Sol = BasePartial<SId,Obj,EmptyInfo>;
    type BType= Batch<Self::Sol,SId,Obj,Opt,EmptyInfo,RSInfo>;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;
    type State = RSState;
    type FnWrap = Objective<Obj,Self::Cod,Out>;

    fn init(&mut self) {}

    fn first_step(&mut self, sp: Arc<Scp>) -> Self::BType {
        rs_iter::<Obj, Opt, Scp>(self, sp.clone())
    }

    fn step(
        &mut self,
        _x: CBType<Self,SId,Obj,Opt,Out,Scp>,
        sp: Arc<Scp>,
    ) -> Batch<Self::Sol,SId, Obj, Opt, EmptyInfo, RSInfo> {
        rs_iter::<Obj, Opt, Scp>(self, sp.clone())
    }

    fn get_state(&mut self) -> &RSState {
        &self.0
    }

    fn from_state(state:RSState) -> Self {
        RandomSearch(state, rand::rng())
    }
}
