
use crate::domain::Domain;
use crate::objective::outcome::Outcome;
use crate::objective::Codomain;

use std::fmt::{Debug, Display};

pub trait Objective<Obj, Cod, Out>
where
    Obj: Domain + Clone + Display + Debug,
    Out : Outcome,
    Cod : Codomain<Out>
{
    fn compute(&self, x:&[Obj::TypeDom]) -> (Cod::TypeCodom,Out);
}

pub struct SimpleObjective<Obj, Cod, Out>
where
    Obj: Domain + Clone + Display + Debug,
    Cod : Codomain<Out>,
    Out : Outcome,
{   
    pub codomain : Cod  ,
    pub function : fn(&[Obj::TypeDom]) -> Out,
}

impl <Obj, Cod, Out> Objective<Obj, Cod, Out> for SimpleObjective<Obj, Cod, Out>
where
    Obj: Domain + Clone + Display + Debug,
    Out : Outcome,
    Cod : Codomain<Out>
{
    fn compute(&self, x:&[Obj::TypeDom]) -> (Cod::TypeCodom,Out) {
        let out = (self.function)(x);
        (self.codomain.get_elem(&out),out)
    }
}

