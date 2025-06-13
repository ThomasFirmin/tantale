use crate::{
    domain::Domain,
    objective::{Codomain,Outcome},
    optimizer::{OptOutput,OptState, OptInfo,SolInfo}
};
use std::fmt::{Display,Debug};


pub trait Stop<State, Obj, Opt, Cod, Out, Info, SInfo, const DIM:usize>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    State:OptState,
{
    fn stop(&self) -> bool;
    fn update(&mut self,
        current_sol : OptOutput<Obj, Opt, Cod, Out, Info, SInfo, DIM>,
        current_best : OptOutput<Obj, Opt, Cod, Out, Info, SInfo, DIM>,
        state_opt : State,
    );
}