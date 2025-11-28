use tantale_core::{
    domain::onto::OntoDom,
    objective::{
        codomain::SingleCodomain,
        outcome::{FuncState, Outcome},
    },
    optimizer::{
        opt::{CBType, Optimizer},
        EmptyInfo, OptInfo, OptState,
    },
    recorder::csv::CSVWritable,
    searchspace::Searchspace,
    solution::{
        partial::{FidBasePartial, FidelityPartial},
        Batch, SId,
    },
    BasePartial, Codomain, Criteria, FidOutcome, Objective, StepCodomain, Stepped,
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

#[derive(Serialize, Deserialize, Debug, Default)]
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

    pub fn codomain<Cod, Out>(extractor: Criteria<Out>) -> Cod
    where
        Cod: Codomain<Out> + From<SingleCodomain<Out>>,
        Out: Outcome,
    {
        let out = SingleCodomain {
            y_criteria: extractor,
        };
        out.into()
    }
}

//-----------------//
//--- OBJECTIVE ---//
//-----------------//

fn rs_iter<Obj, Opt, Scp>(
    opt: &mut RandomSearch,
    sp: &Scp,
) -> Batch<BasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo, RSInfo>
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Scp: Searchspace<BasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo>,
{
    let samples = sp.vec_sample_obj(Some(&mut opt.1), opt.0.batch, Arc::new(EmptyInfo {}));
    let opt_samples = sp.vec_onto_opt(&samples);
    let info = Arc::new(RSInfo {
        iteration: opt.0.iteration,
    });
    opt.0.iteration += 1;
    Batch::new(samples, opt_samples, info)
}

impl<Obj, Opt, Out, Scp> Optimizer<SId, Obj, Opt, Out, Scp, Objective<Obj, Out>> for RandomSearch
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

    fn init(&mut self) {}

    fn first_step(&mut self, scp: &Scp) -> Self::BType {
        rs_iter::<Obj, Opt, Scp>(self, scp)
    }

    fn step(
        &mut self,
        _x: CBType<Self, SId, Obj, Opt, Out, Scp, Objective<Obj, Out>>,
        scp: &Scp,
    ) -> Self::BType {
        rs_iter::<Obj, Opt, Scp>(self, scp)
    }

    fn get_state(&mut self) -> &RSState {
        &self.0
    }

    fn from_state(state: RSState) -> Self {
        RandomSearch(state, rand::rng())
    }
}

//---------------//
//--- STEPPED ---//
//---------------//

fn fid_rs_iter<Obj, Opt, Scp>(
    opt: &mut RandomSearch,
    sp: &Scp,
    batch: &mut Batch<FidBasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo, RSInfo>,
)
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Scp: Searchspace<FidBasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo>,
{
    let info = Arc::new(EmptyInfo);
    (batch.sobj,batch.sopt) = sp.sample_pair(Some(&mut opt.1), opt.0.batch, info.clone()).into_iter().unzip();
}

impl<Obj, Opt, Out, Scp, FnState> Optimizer<SId, Obj, Opt, Out, Scp, Stepped<Obj, Out, FnState>>
    for RandomSearch
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    Out: FidOutcome,
    Scp: Searchspace<FidBasePartial<SId, Obj, EmptyInfo>, SId, Obj, Opt, EmptyInfo>,
    FnState: FuncState,
{
    type Sol = FidBasePartial<SId, Obj, EmptyInfo>;
    type BType = Batch<Self::Sol, SId, Obj, Opt, EmptyInfo, RSInfo>;
    type Cod = StepCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;
    type State = RSState;

    fn init(&mut self) {}

    fn first_step(&mut self, scp: &Scp) -> Self::BType {
        let info = RSInfo{iteration: self.0.iteration};
        let mut batch = Batch::empty(info.into());
        self.0.iteration += 1;
        fid_rs_iter::<Obj, Opt, Scp>(self, scp, &mut batch);
        batch
    }

    fn step(
        &mut self,
        x: CBType<Self, SId, Obj, Opt, Out, Scp, Stepped<Obj, Out, FnState>>,
        scp: &Scp,
    ) -> Self::BType {
        let mut batch: Self::BType = x.into_iter().filter_map(
            |(obj, opt)| 
            {
                let step = obj.get_y().step;
                let mut xobj = obj.sol;
                let mut xopt = opt.sol;
                if step.is_partially(){
                    xobj.resume(&mut xopt, 0.0);
                    Some((xobj, xopt))
                }
                else{None}
            }
        ).collect();
        if batch.size() == 0{
            self.0.iteration += 1;
            fid_rs_iter::<Obj, Opt, Scp>(self, scp, &mut batch)
        }
        let info = RSInfo {iteration: self.0.iteration}.into();
        batch.info = info;
        batch
    }

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        RandomSearch(state, rand::rng())
    }
}
