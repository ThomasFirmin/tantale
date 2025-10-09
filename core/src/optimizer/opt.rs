use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    saver::CSVWritable,
    searchspace::Searchspace,
    solution::{Id, SolInfo},
    optimizer::BatchType,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
pub type PBType<Op:Optimizer<SolId, Obj, Opt, Cod, Out, Scp>, SolId, Obj, Opt, Cod, Out, Scp> = <Op as Optimizer<SolId, Obj, Opt, Cod, Out, Scp>>::BType;
pub type CBType<Op:Optimizer<SolId, Obj, Opt, Cod, Out, Scp>, SolId, Obj, Opt, Cod, Out, Scp> = <<Op as Optimizer<SolId, Obj, Opt, Cod, Out, Scp>>::BType as BatchType<SolId,Obj,Opt,Op::SInfo,Op::Info>>::Comp<Cod,Out>;
pub type OBType<Op:Optimizer<SolId, Obj, Opt, Cod, Out, Scp>, SolId, Obj, Opt, Cod, Out, Scp> = <<Op as Optimizer<SolId, Obj, Opt, Cod, Out, Scp>>::BType as BatchType<SolId,Obj,Opt,Op::SInfo,Op::Info>>::Outc<Out>;

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
