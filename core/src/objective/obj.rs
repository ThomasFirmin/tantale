
use crate::domain::Domain;
use crate::objective::Codomain;

use std::fmt::{Debug, Display};

pub trait Objective<Obj, C, Out, const DIM: usize>
where
    Obj: Domain + Clone + Display + Debug,
    C : Codomain,
    Out : Outcome,
{
    fn compute(&self, x:&[Obj]) -> (C,Out);
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

