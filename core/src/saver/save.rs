use crate::{
    domain::Domain,
    objective::{Codomain, Outcome},
    optimizer::{OptInfo, SolInfo, OptState},
    searchspace::Searchspace,
    solution::Solution,
};
use std::fmt::{Display,Debug};

pub trait Saver<State, Obj, Opt, Cod, Out, Info, SInfo, Sp, const N:usize>
where
    State: OptState,
    Opt: Domain + Clone + Display + Debug,
    Obj: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    Info:OptInfo,
    SInfo : SolInfo,
    Sp:Searchspace<Obj, Opt, Cod, Out, SInfo, N>,
{
    fn save_init(&self);
    fn save_obj(&self, sol: &[Solution<Obj, Cod, Out, SInfo, N>], sp : Sp, info : Info);
    fn save_opt(&self, sol: &[Solution<Opt, Cod, Out, SInfo, N>], sp : Sp, info : Info);
    fn save_state(&self, state:&State);
}