use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    saver::CSVWritable,
    searchspace::Searchspace,
    solution::{Computed, Id, Partial, SolInfo},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub trait BatchType<SolId,ADom,BDom,SInfo,Info>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome>: CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
}

pub trait CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part: BatchType<SolId,ADom,BDom,SInfo,Info>;
}

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Batch<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo
{
    pub sobj : ArcVecArc<Partial<SolId,ADom,SInfo>>,
    pub sopt : ArcVecArc<Partial<SolId,BDom,SInfo>>,
    pub info : Arc<Info>
}

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct CompBatch<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub cobj : ArcVecArc<Computed<SolId,ADom,Cod,Out,SInfo>>,
    pub copt : ArcVecArc<Computed<SolId,BDom,Cod,Out,SInfo>>,
    pub info : Arc<Info>
}

impl <SolId,ADom,BDom,SInfo,Info> BatchType<SolId,ADom,BDom,SInfo,Info> for Batch<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome> = CompBatch<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
}

impl <SolId,ADom,BDom,SInfo,Info,Cod,Out> CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out> for CompBatch<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part = Batch<SolId,ADom,BDom,SInfo,Info>;
}

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Single<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub sobj : Arc<Partial<SolId,ADom,SInfo>>,
    pub sopt : Arc<Partial<SolId,BDom,SInfo>>,
    pub info : Arc<Info>
}

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct CompSingle<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub cobj : ArcVecArc<Computed<SolId,ADom,Cod,Out,SInfo>>,
    pub copt : ArcVecArc<Computed<SolId,BDom,Cod,Out,SInfo>>,
    pub info : Arc<Info>
}

impl <SolId,ADom,BDom,SInfo,Info> BatchType<SolId,ADom,BDom,SInfo,Info> for Single<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome> = CompSingle<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
}

impl <SolId,ADom,BDom,SInfo,Info,Cod,Out> CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out> for CompSingle<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part = Single<SolId,ADom,BDom,SInfo,Info>;
}

/// Describes information linked to a group of [`Solutions`](Solution)
/// obtained  after each iteration of the [`Optimizer`].
pub trait OptInfo
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}


/// Describes the current state of an [`Optimizer`].
/// It is used to serialize and deserialize the [`Optimizer`].
pub trait OptState
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// An empty [`OptInfo`] or [`SolInfo`].
#[derive(Serialize, Deserialize, std::fmt::Debug)]
pub struct EmptyInfo {}
impl SolInfo for EmptyInfo {}
impl OptInfo for EmptyInfo {}
impl CSVWritable<(), ()> for EmptyInfo {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::new()
    }
}

pub type ArcVecArc<T> = Arc<Vec<Arc<T>>>;
/// Computed [`BatchType`], the associated type of a [`BatchType`] knwowing the optimizer.
pub type CBType<Op:Optimizer<SolId, Obj, Opt, Cod, Out, Scp>, SolId, Obj, Opt, Cod, Out, Scp> = <<Op as Optimizer<SolId, Obj, Opt, Cod, Out, Scp>>::BType as BatchType<SolId,Obj,Opt,Op::SInfo,Op::Info>>::Comp<Cod,Out>;

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<SolId, Obj, Opt, Cod, Out, Scp>
where
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Scp: Searchspace<SolId, Obj, Opt, Self::SInfo>,
{
    type SInfo: SolInfo;
    type Info: OptInfo;
    type State: OptState;
    type BType: BatchType<SolId,Obj,Opt,Self::SInfo,Self::Info>;

    /// Initialize the [`Optimizer`]
    fn init(&mut self);

    /// Executed once at the beginning of the optimization. Does not require previous [`Computed`].
    fn first_step(&mut self, sp: Arc<Scp>) -> Self::BType;

    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    /// Requires previously [`Computed`] `x` [`Solution`].
    fn step(
        &mut self,
        x: CBType<Self,SolId,Obj,Opt,Cod,Out,Scp>,
        sp: Arc<Scp>,
    ) ->Self::BType;

    /// Returns the current [`OptState`] of the [`Optimizer`].
    fn get_state(&mut self) -> &Self::State;

    /// Return an instance of the [`Optimizer`]  from an [`OptState`].
    fn from_state(state: Self::State) -> Self;
}
