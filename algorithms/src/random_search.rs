use tantale_core::{
    BasePartial, Codomain, Criteria, FidOutcome, Objective, Solution, Stepped, domain::onto::LinkOpt, objective::{
        Step, codomain::SingleCodomain, outcome::{FuncState, Outcome}
    }, optimizer::{
        EmptyInfo, OptInfo, OptState, opt::{BatchOptimizer, Optimizer}
    }, recorder::csv::CSVWritable, searchspace::{CompShape, Searchspace}, solution::{
        Batch, HasFidelity, HasStep, IntoComputed, SId, SolutionShape, partial::FidBasePartial, shape::RawObj
    }
};

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};

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

fn rs_iter<Scp,PSol>(
    opt: &mut RandomSearch,
    sp: &Scp,
    bsize:usize,
) -> Batch<SId,EmptyInfo,RSInfo,Scp::SolShape>
where
    PSol: Solution<SId,Scp::Opt,EmptyInfo>,
    Scp: Searchspace<PSol,SId,EmptyInfo>,
{
    let info = RSInfo{iteration: opt.0.iteration};
    opt.0.iteration += 1;
    let pairs = sp.sample_pair(
        Some(&mut opt.1),
        bsize,
        EmptyInfo.into());
    Batch::new(pairs, info.into())
}

impl<Out,Scp> Optimizer<BasePartial<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp> for RandomSearch 
where
    Out:Outcome,
    Scp: Searchspace<BasePartial<SId,LinkOpt<Scp>,EmptyInfo>,SId,EmptyInfo>,
{
    type State = RSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state,rand::rng())
    }
}

impl<Out,Scp> BatchOptimizer<BasePartial<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp,Objective<RawObj<Scp::SolShape,SId,EmptyInfo>,Out>> for RandomSearch
where
    Out:Outcome,
    Scp: Searchspace<BasePartial<SId,LinkOpt<Scp>,EmptyInfo>,SId,EmptyInfo>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SId, Self::SInfo>,
{
    fn first_step(&mut self, scp: &Scp) -> Batch<SId,Self::SInfo,Self::Info,Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }

    fn step(&mut self, _x: Batch<SId,Self::SInfo,Self::Info, CompShape<Scp,BasePartial<SId,Scp::Opt,EmptyInfo>,SId,Self::SInfo,Self::Cod,Out>>, scp:&Scp) -> Batch<SId,Self::SInfo,Self::Info,Scp::SolShape> {
        rs_iter(self, scp,self.0.batch)
    }
}

//---------------//
//--- STEPPED ---//
//---------------//


impl<Out,Scp> Optimizer<FidBasePartial<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp> for RandomSearch 
where
    Out:FidOutcome,
    Scp: Searchspace<FidBasePartial<SId,LinkOpt<Scp>,EmptyInfo>,SId,EmptyInfo>,
{
    type State = RSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state,rand::rng())
    }
}

impl<Out,Scp,FnState> BatchOptimizer<FidBasePartial<SId,Scp::Opt,EmptyInfo>,SId,Scp::Opt,Out,Scp,Stepped<RawObj<Scp::SolShape,SId,EmptyInfo>,Out,FnState>> for RandomSearch
where
    Out:FidOutcome,
    Scp: Searchspace<FidBasePartial<SId,LinkOpt<Scp>,EmptyInfo>,SId,EmptyInfo>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState:FuncState,
{
    fn first_step(&mut self, scp: &Scp) -> Batch<SId,Self::SInfo,Self::Info,Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }

    fn step(&mut self, x: Batch<SId,Self::SInfo,Self::Info, CompShape<Scp,FidBasePartial<SId,Scp::Opt,EmptyInfo>,SId,Self::SInfo,Self::Cod,Out>>, scp:&Scp) -> Batch<SId,Self::SInfo,Self::Info,Scp::SolShape> {
        let mut pairs: Vec<_> = x
            .into_iter()
            .filter_map(|p| {
                match p.step(){
                    Step::Pending => Some(IntoComputed::extract(p).0),
                    Step::Partially(_) => Some(IntoComputed::extract(p).0),
                    _ => None,
                }
            })
            .collect();
        let info = RSInfo{iteration: self.0.iteration}.into();
        if pairs.len() < self.0.batch{
            let fill = self.0.batch - pairs.len();
            let mut npairs = scp.sample_pair(Some(&mut self.1),fill,EmptyInfo.into());
            pairs.append(&mut npairs);
        }
        Batch::new(pairs, info)
    }
}