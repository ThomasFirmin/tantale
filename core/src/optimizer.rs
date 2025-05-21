use crate::{Domain, Searchspace, Solution};
use std::fmt::{Display, Debug};

pub trait Optimizer<D, const DIM : usize>
where
    D: Domain + Clone + Display + Debug,
{
    fn step(&self, x : &[Solution<D, DIM>]) -> (&[Solution<D, DIM>]);
    fn searchspace(&self) -> impl Searchspace<D, DIM>;
}
