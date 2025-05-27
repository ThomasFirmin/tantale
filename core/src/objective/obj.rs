
use crate::domain::Domain;
use crate::objective::outcome::Outcome;
use crate::objective::Codomain;

use std::fmt::{Debug, Display};

pub trait Objective<Obj, Cod, Out, const DIM: usize>
where
    Obj: Domain + Clone + Display + Debug,
    Out : Outcome,
    Cod : Codomain<Out>
{
    fn compute(&self, x:&[Obj]) -> (Cod,Out);
}

pub struct SimpleObjective<Obj, C, Out, const DIM: usize>
where
    Obj: Domain + Clone + Display + Debug,
    C : Codomain,
    Out : Outcome,
{   
    pub codomain : C,
    pub function : fn(&[Obj]) -> Out,
}

