use crate::{objective::Codomain, Domain, Searchspace, Solution};
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

pub trait Optimizer<Obj, Opt, C, const DIM : usize, const CRIT : usize>
where
    Obj : Domain + Clone + Display + Debug,
    Opt : Domain + Clone + Display + Debug,
    C : Codomain,
{
    fn step(&self, x : &[(Solution<Opt, C, DIM>, [f64; CRIT])]) -> &[Solution<Opt,C ,DIM>];
    fn searchspace(&self) -> impl Searchspace<Obj, Opt, C, DIM>;
    fn state(&self) -> Self;
    fn iteration(&self) -> usize;
}

#[cfg(feature="par")]
pub trait ParallelOptimizer<Obj, Opt, C, const DIM : usize, const CRIT : usize>:Optimizer<Obj, Opt, C, DIM, CRIT>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    C : Codomain,
{
    fn interact(&self);
    fn update(&self);
}
