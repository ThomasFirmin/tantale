use crate::{
    domain::Domain,
    objective::{Codomain, Objective, Outcome},
    optimizer::{OptInfo, Optimizer, SolInfo, OptState},
    searchspace::Searchspace,
    solution::Solution,
};
use std::fmt::{Display,Debug};

pub trait Saver<State, Dom, Cod, Out, Info, const N:usize>
where
    State: OptState,
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    Info : SolInfo,
{
    fn save_init(&self);
    fn save_sol(&self, sol: &[Solution<Dom, Cod, Out, Info, N>]);
    fn save_state(&self, state:&State);
}