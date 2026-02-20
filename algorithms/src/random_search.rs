use tantale_core::{
    BaseSol, Codomain, Criteria, FidOutcome, Objective, Solution, Stepped,
    domain::{
        codomain::{SingleCodomain, TypeCodom},
        onto::LinkOpt,
    },
    objective::{
        Step,
        outcome::{FuncState, Outcome},
    },
    optimizer::{
        EmptyInfo, OptInfo, OptState,
        opt::{BatchOptimizer, Optimizer, SequentialOptimizer},
    },
    recorder::csv::CSVWritable,
    searchspace::{CompShape, OptionCompShape, Searchspace},
    solution::{
        Batch, HasFidelity, HasStep, IntoComputed, SId, SolutionShape, partial::FidelitySol,
        shape::RawObj,
    },
};

use rand::{SeedableRng, prelude::ThreadRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{cell::RefCell, sync::Arc};

thread_local! {
    static THREAD_RNG: RefCell<StdRng> = RefCell::new(StdRng::from_os_rng());
}

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

pub struct RandomSearch(pub SeqRSState);

impl RandomSearch {
    pub fn new() -> Self {
        RandomSearch(SeqRSState)
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

    fn with_rng<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut StdRng) -> T,
    {
        THREAD_RNG.with(|rng| f(&mut rng.borrow_mut()))
    }
}

impl Default for RandomSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl<Out, Scp> Optimizer<BaseSol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for RandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = SeqRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = EmptyInfo;

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state)
    }
}

impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for RandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
{
    type State = SeqRSState;
    type Cod = SingleCodomain<Out>;
    type SInfo = EmptyInfo;
    type Info = EmptyInfo;

    fn get_state(&mut self) -> &Self::State {
        &self.0
    }

    fn from_state(state: Self::State) -> Self {
        Self(state)
    }
}

impl<Out, Scp>
    SequentialOptimizer<
        BaseSol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SId, EmptyInfo>, Out>,
    > for RandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SId, Self::SInfo>,
{
    fn step(
        &mut self,
        _x: OptionCompShape<
            Scp,
            BaseSol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
    ) -> Scp::SolShape {
        self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into()))
    }
}

impl<Out, Scp, FnState>
    SequentialOptimizer<
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for RandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
    Scp::SolShape: HasStep + HasFidelity,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>:
        SolutionShape<SId, Self::SInfo> + HasStep + HasFidelity,
    FnState: FuncState,
{
    fn step(
        &mut self,
        x: OptionCompShape<
            Scp,
            FidelitySol<SId, Scp::Opt, EmptyInfo>,
            SId,
            Self::SInfo,
            Self::Cod,
            Out,
        >,
        scp: &Scp,
    ) -> Scp::SolShape {
        match x {
            Some(comp) => {
                let (pair, _): (Scp::SolShape, Arc<TypeCodom<SingleCodomain<Out>, Out>>) =
                    IntoComputed::extract(comp);
                match pair.step() {
                    Step::Pending => {
                        unreachable!(
                            "A pending SolShape, should not be passed to RandomSearch step."
                        )
                    }
                    Step::Partially(_) => pair,
                    _ => self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into())),
                }
            }
            None => self.with_rng(|rng| scp.sample_pair(rng, EmptyInfo.into())),
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

impl BatchRSState {
    pub fn new(batch: usize, iteration: usize) -> Self {
        BatchRSState {
            batch,
            iteration,
            _emptyinfo: Arc::new(EmptyInfo),
        }
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
    let pairs = sp.vec_sample_pair(&mut opt.1, bsize, opt.0._emptyinfo.clone());
    Batch::new(pairs, info.into())
}

impl<Out, Scp> Optimizer<BaseSol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for BatchRandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
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
        BaseSol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Objective<RawObj<Scp::SolShape, SId, EmptyInfo>, Out>,
    > for BatchRandomSearch
where
    Out: Outcome,
    Scp: Searchspace<BaseSol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
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
            CompShape<Scp, BaseSol<SId, Scp::Opt, EmptyInfo>, SId, Self::SInfo, Self::Cod, Out>,
        >,
        scp: &Scp,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        rs_iter(self, scp, self.0.batch)
    }
}

//---------------//
//--- STEPPED ---//
//---------------//

impl<Out, Scp> Optimizer<FidelitySol<SId, Scp::Opt, EmptyInfo>, SId, Scp::Opt, Out, Scp>
    for BatchRandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
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
        FidelitySol<SId, Scp::Opt, EmptyInfo>,
        SId,
        Scp::Opt,
        Out,
        Scp,
        Stepped<RawObj<Scp::SolShape, SId, EmptyInfo>, Out, FnState>,
    > for BatchRandomSearch
where
    Out: FidOutcome,
    Scp: Searchspace<FidelitySol<SId, LinkOpt<Scp>, EmptyInfo>, SId, EmptyInfo>,
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
                FidelitySol<SId, Scp::Opt, EmptyInfo>,
                SId,
                Self::SInfo,
                Self::Cod,
                Out,
            >,
        >,
        scp: &Scp,
    ) -> Batch<SId, Self::SInfo, Self::Info, Scp::SolShape> {
        let pairs: Vec<_> = x
            .into_iter()
            .map(|p| match p.step() {
                Step::Evaluated | Step::Discard | Step::Error => {
                    scp.sample_pair(&mut self.1, self.0._emptyinfo.clone())
                }
                _ => IntoComputed::extract(p).0,
            })
            .collect();
        self.0.iteration += 1;
        let info = RSInfo {
            iteration: self.0.iteration,
        }
        .into();
        Batch::new(pairs, info)
    }
}
