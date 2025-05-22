
use crate::{domain::Domain, Solution};
use crate::objective::criteria::Criteria;

use std::{collections::HashMap, fmt::{Debug, Display}};

pub struct Codomain<'a, D, const DIM: usize, const CRIT : usize>
where
    D: Domain + Clone + Display + Debug,
{
    pub sol : &'a Solution<D,DIM>,
    pub out : [f64;CRIT],
}

pub trait Objective<Obj, const DIM: usize, const CRIT: usize>
where
    Obj: Domain + Clone + Display + Debug,
{
    fn compute(&self, x:&Solution<Obj,DIM>) -> HashMap<&str,f64>;
}

pub struct SliceObjective<Obj, const DIM: usize, const CRIT: usize>
where
    Obj: Domain + Clone + Display + Debug,
{   
    pub criteria : [Box<dyn Criteria>; CRIT],
    pub function : fn(&[Obj]) -> HashMap<&str,f64>,
}

pub struct HashMapObjective{}

pub struct PyKwargsObjective{}


