use crate::{objective::{Codomain, Outcome}, Domain, Searchspace, Solution};
use std::fmt::{Display, Debug};

pub trait OptimizerInfo 
where
    Self : Sized
{
    fn info(&self) -> Option<Self>;
}

pub trait OptimizerPerSolInfo 
where
    Self : Sized
{
    fn sol_info(&self) -> Option<Self>;
}

pub trait Optimizer<Obj, Opt, Cod, Out, const DIM : usize, const CRIT : usize>
where
    Obj : Domain + Clone + Display + Debug,
    Opt : Domain + Clone + Display + Debug,
    Out : Outcome,
    Cod : Codomain<Out>,
{
    fn step(&self, x : &[Solution<Opt, Cod, Out, DIM>]) -> &[Solution<Opt,Cod, Out ,DIM>];
    fn searchspace(&self) -> impl Searchspace<Obj, Opt, Cod, Out,DIM>;
    fn state(&self) -> Self;
    fn iteration(&self) -> usize;
}

#[cfg(feature="par")]
pub trait ParallelOptimizer<Obj, Opt, Cod, Out, const DIM : usize, const CRIT : usize>:Optimizer<Obj, Opt, Cod, Out, DIM, CRIT>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Cod : Codomain<Out>,
    Out : Outcome,
{
    fn interact(&self);
    fn update(&self);
}
