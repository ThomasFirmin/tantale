use std::sync::Arc;

use tantale_core::{
    BasePartial, Codomain, Criteria, FidOutcome, Objective, Solution, Stepped, domain::{
        codomain::{SingleCodomain, TypeCodom},
        onto::LinkOpt,
    }, objective::{Step, outcome::{FuncState, Outcome}}, optimizer::{
        EmptyInfo, OptInfo, OptState, opt::{BatchOptimizer, Optimizer, SequentialOptimizer}
    }, recorder::csv::CSVWritable, searchspace::{CompShape, OptionCompShape, Searchspace}, solution::{
        Batch, HasFidelity, HasStep, IntoComputed, SId, SolutionShape, partial::FidBasePartial, shape::RawObj
    }
};

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};


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


//------------------//
//--- SEQUENTIAL ---//
//------------------//

#[derive(Serialize, Deserialize)]
pub struct SeqRSState;
impl OptState for SeqRSState {}

pub struct RandomSearch(pub SeqRSState, ThreadRng);

impl RandomSearch {
    pub fn new() -> Self {
        let rng = rand::rng();
        RandomSearch(
            SeqRSState,
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

impl Default for RandomSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl<Out, Scp> Optimizer<BasePartial<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for RandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BasePartial<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = SeqRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state, rand::rng())
    }
}

impl<Out, Scp> Optimizer<FidBasePartial<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for RandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidBasePartial<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = SeqRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state, rand::rng())
    }
}

impl<Out, Scp>
    SequentialOptimizer<
        BasePartial<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SId, EmptyInfo>, Out>,
    > for RandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BasePartial<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SId, Self::SInfo>,
{
    fn step(
            &mut self,
            _x: OptionCompShape<Scp, BasePartial<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
            scp: &Scp,
        ) -> Scp::SolShape
    {
        scp.sample_pair(Some(&mut self.1),EmptyInfo.into())    
    }
}

impl<Out, Scp, FnState>
    SequentialOptimizer<
        FidBasePartial<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for RandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidBasePartial<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    fn step(
            &mut self,
            x: OptionCompShape<Scp, FidBasePartial<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
            scp: &Scp,
        ) -> Scp::SolShape
    {
        match x{
            Some(comp) => 
            {
                let (pair, _): (Scp::SolShape, Arc<TypeCodom<SingleCodomain<Out>, Out>>) = IntoComputed::extract(comp);
                match pair.step() {
                    Step::Pending => {
                        unreachable!("A pending SolShape, should not be passed to RandomSearch step.")
                    }
                    Step::Partially(_) => pair,
                    _ => scp.sample_pair(Some(&mut self.1), EmptyInfo.into()),
                }
            },
            None => scp.sample_pair(Some(&mut self.1), EmptyInfo.into()),
        }
    }
}

//------------------//
//--- Batched ---//
//------------------//

#[derive(Serialize, Deserialize)]
pub struct BatchRSState {
    pub batch: usize,
    pub iteration: usize,
    _emptyinfo: Arc<EmptyInfo>,
}

impl BatchRSState{
    pub fn new(batch:usize, iteration:usize) -> Self{
        BatchRSState { batch, iteration, _emptyinfo: Arc::new(EmptyInfo) }
    }
}
impl OptState for BatchRSState {}

pub struct BatchRandomSearch(pub BatchRSState, ThreadRng);

impl BatchRandomSearch {
    pub fn new(batch: usize) -> Self {
        let rng = rand::rng();
        BatchRandomSearch(
            BatchRSState {
                batch,
                iteration: 0,
                _emptyinfo: Arc::new(EmptyInfo),
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

fn rs_iter<Scp, PSol>(
    opt: &mut BatchRandomSearch,
    sp: &Scp,
    bsize: usize,
) -> Batch<SId, EmptyInfo, RSInfo, Scp::SolShape>
where
    PSol: Solution<SId, Scp::Opt, EmptyInfo>,
    Scp: Searchspace<PSol, SId, EmptyInfo>,
{
    let info = RSInfo {
        iteration: opt.0.iteration,
    };
    opt.0.iteration += 1;
    let pairs = sp.vec_sample_pair(Some(&mut opt.1), bsize, opt.0._emptyinfo.clone());
    Batch::new(pairs, info.into())
}

impl<Out, Scp> Optimizer<BasePartial<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for BatchRandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BasePartial<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = BatchRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state, rand::rng())
    }
}

impl<Out, Scp>
    BatchOptimizer<
        BasePartial<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SId, EmptyInfo>, Out>,
    > for BatchRandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BasePartial<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SId, Self::SInfo>,
{
    fn first_step(&mut self, scp: &Scp) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }

    fn step(
        &mut self,
        _x: Batch<
            SId,
            Self::SInfo,
            Self::Info,
            CompShape<Scp, BasePartial<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
        >,
        scp: &Scp,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }
}

//---------------//
//--- STEPPED ---//
//---------------//

impl<Out, Scp> Optimizer<FidBasePartial<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for BatchRandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidBasePartial<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = BatchRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = RSInfo;

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state, rand::rng())
    }
}

impl<Out, Scp, FnState>
    BatchOptimizer<
        FidBasePartial<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for BatchRandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidBasePartial<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    fn first_step(&mut self, scp: &Scp) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }

    fn step(
        &mut self,
        x: Batch<
            SId,
            Self::SInfo,
            Self::Info,
            CompShape<
                Scp,
                FidBasePartial<SId, Scp::Opt, EmptyInfo>,
                SId,
                Self::SInfo,
                Self::Cod,
                Out,
            >,
        >,
        scp: &Scp,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        let pairs: Vec<_> = x.into_iter().map(|p| 
            {
                match p.step() {
                    Step::Evaluated | Step::Discard| Step::Error => scp.sample_pair(Some(&mut self.1), self.0._emptyinfo.clone()),
                    _ => IntoComputed::extract(p).0,
                }
            }
        ).collect();
        self.0.iteration += 1;
        let info = RSInfo {
            iteration: self.0.iteration,
        }
        .into();
        Batch::new(pairs, info)
    }
}