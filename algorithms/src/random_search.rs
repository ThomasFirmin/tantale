use std::sync::Arc;

use tantale_core::{
    domain::{
        codomain::{SingleCodomain, TypeCodom},
        onto::LinkOpt,
    },
    objective::outcome::{FuncState, Outcome},
    optimizer::{
        opt::{BatchOptimizer, Optimizer, SequentialOptimizer},
        EmptyInfo, OptInfo, OptState,
    },
    recorder::csv::CSVWritable,
    searchspace::{CompShape, Searchspace},
    solution::{
        partial::FidBasePartial, shape::RawObj, Batch, HasFidelity, HasStep, IntoComputed, SId,
        SolutionShape,
    },
    BasePartial, Codomain, Criteria, FidOutcome, Objective, Solution, Stepped,
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
pub struct SeqRSState {
    pub iteration: usize,
}
impl OptState for SeqRSState {}

pub struct RandomSearch(pub SeqRSState, ThreadRng);

impl RandomSearch {
    pub fn new() -> Self {
        let rng = rand::rng();
        RandomSearch(
            SeqRSState {
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
    fn first_step(&mut self, scp: &Scp) -> Scp::SolShape {
        scp.sample_pair(Some(&mut self.1), EmptyInfo.into())
    }

    fn step(
        &mut self,
        x: CompShape<
            Scp,
            FidBasePartial<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
    ) -> Scp::SolShape {
        let (pair, _): (Scp::SolShape, Arc<TypeCodom<SingleCodomain<Out>, Out>>) =
            IntoComputed::extract(x);
        match pair.step() {
            tantale_core::objective::Step::Pending => {
                unreachable!("A pending SolShape, should not be passed to RandomSearch step.")
            }
            tantale_core::objective::Step::Partially(_) => pair,
            _ => scp.sample_pair(Some(&mut self.1), EmptyInfo.into()),
        }
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
    fn first_step(&mut self, scp: &Scp) -> Scp::SolShape {
        scp.sample_pair(Some(&mut self.1), EmptyInfo.into())
    }

    fn step(
        &mut self,
        _x: CompShape<Scp, BasePartial<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
        scp: &Scp,
    ) -> Scp::SolShape {
        scp.sample_pair(Some(&mut self.1), EmptyInfo.into())
    }
}

//------------------//
//--- Batched ---//
//------------------//

#[derive(Serialize, Deserialize)]
pub struct BatchRSState {
    pub batch: usize,
    pub iteration: usize,
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
    let pairs = sp.vec_sample_pair(Some(&mut opt.1), bsize, EmptyInfo.into());
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
        let mut pairs: Vec<_> = x.into_iter().map(|p| IntoComputed::extract(p).0).collect();
        if pairs.len() < self.0.batch {
            let fill = self.0.batch - pairs.len();
            let mut npairs = scp.vec_sample_pair(Some(&mut self.1), fill, EmptyInfo.into());
            pairs.append(&mut npairs);
            self.0.iteration += 1;
        }
        let info = RSInfo {
            iteration: self.0.iteration,
        }
        .into();
        Batch::new(pairs, info)
    }
}