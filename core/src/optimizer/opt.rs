use crate::{
    objective::{Codomain, Outcome}, solution::ComputedSol, Domain, Searchspace
};
use std::{fmt::{Debug, Display}, sync::Arc};

/// Return type of [`Solution`] from an [`Optimizer`].
pub type ArcSol<Dom, Cod, Out, SInfo, const DIM:usize> = Arc<[ComputedSol<Dom, Cod, Out, SInfo, DIM>]>;

/// Output of an [`Optimizer`].
pub struct OptOutput<Obj, Opt, Cod, Out, Info, SInfo, const DIM:usize>
(
    ArcSol<Obj, Cod, Out, SInfo, DIM>,
    ArcSol<Opt, Cod, Out, SInfo, DIM>,
    Info,
)
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo;

impl <Obj, Opt, Cod, Out, Info, SInfo, const DIM:usize> OptOutput<Obj, Opt, Cod, Out, Info, SInfo, DIM>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo
{
    pub fn new(
        sol_obj:ArcSol<Obj, Cod, Out, SInfo, DIM>,
        sol_opt:ArcSol<Opt, Cod, Out, SInfo, DIM>,
        info:Info,
    )->Self
    {
        OptOutput(sol_obj, sol_opt, info)
    }
}

/// Describes information linked to a group of [`Solutions`](Solution)
/// obtained  after each iteration of the [`Optimizer`].
pub trait OptInfo{}

/// Describes single-[`Solution`] information
/// obtained after each iteration of the [`Optimizer`].
pub trait SolInfo {}

/// An empty [`OptInfo`] or [`SolInfo`].
pub struct EmptyInfo {}
impl SolInfo for EmptyInfo {}
impl OptInfo for EmptyInfo {}

/// Describes the current state of the [`Optimizer`].
/// At each iteration an [`Optimizer`] uses the previous
/// [`OptState`] to update the current one.
/// It is mostly used for checkpointing.
pub trait OptState{}

/// The [`Optimizer`] is one of the elemental software brick of the library.
/// It describes how to sample [`Solutions`](Solution) in order to **maximize**
/// the [`Objective`] function.
pub trait Optimizer<Obj, Opt, Cod, Out, Sp, Info, SInfo, State, const DIM: usize>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Sp: Searchspace<Obj, Opt, Cod, Out, SInfo, DIM>,
    Info: OptInfo,
    SInfo: SolInfo,
    State:OptState,
{
    /// Computes a single iteration of the [`Optimizer`]. It must return a slice of [`Solution`]`<Opt,Cod, Out, SInfo, DIM>`
    /// and some optimizer info [`OptInfo`]. [`Self`] is mutable in order to update the [`Optimizer`]'s state.
    fn step(
        &mut self,
        x: ArcSol<Opt, Cod, Out, SInfo, DIM>,
        sp: &Sp,
        state:&mut State,
        pid:u32,
    ) -> OptOutput<Obj, Opt, Cod, Out, Info, SInfo, DIM>;

}

#[cfg(feature = "par")]
pub trait ParallelOptimizer<
    Obj,
    Opt,
    Cod,
    Out,
    Sp,
    Info,
    SInfo,
    State:OptState,
    const DIM: usize,
>: Optimizer<Obj, Opt, Cod, Out, Sp, Info, SInfo, State, DIM> where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Sp: Searchspace<Obj, Opt, Cod, Out, SInfo, DIM>,
    Info: OptInfo,
    SInfo: SolInfo,
{
    fn interact(&self);
    fn update(&self);
}
