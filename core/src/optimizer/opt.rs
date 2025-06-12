use crate::{
    objective::{Codomain, Outcome},
    Domain, Searchspace, Solution,
};
use std::fmt::{Debug, Display};

pub trait OptInfo
where
    Self: Sized,
{
    fn get_opt_info(&self) -> Self;
}

pub trait SolInfo {
    fn get_sol_info(&self) -> Self;
}

/// An empty [`OptInfo`] or [`SolInfo`].
pub struct EmptyInfo {}
impl SolInfo for EmptyInfo {
    fn get_sol_info(&self) -> Self {
        EmptyInfo {}
    }
}
impl OptInfo for EmptyInfo {
    fn get_opt_info(&self) -> Self {
        EmptyInfo {}
    }
}

pub trait OptState{}

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
        x: &[Solution<Opt, Cod, Out, SInfo, DIM>],
        sp: &Sp,
        state:&State,
    ) -> (&[Solution<Opt, Cod, Out, SInfo, DIM>], Info);
    fn iteration(&self) -> usize;
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
    const DIM: usize,
>: Optimizer<Obj, Opt, Cod, Out, Sp, Info, SInfo, DIM> where
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
