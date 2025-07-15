use tantale_core::{
    domain::Domain, objective::{
        codomain::Codomain,
        outcome::Outcome
    }, optimizer::{EmptyInfo, OptInfo, OptState, Optimizer},
    searchspace::Searchspace
};

use std::fmt::{Debug, Display};
use rand::prelude::ThreadRng;

pub struct RSState{
    pub iteration : usize
}
impl OptState for RSState{}

pub struct RSInfo{
    pub iteration : usize
}

impl OptInfo for RSInfo{}


pub struct RandomSearch (usize, ThreadRng);
impl RandomSearch{
    pub fn new(batch:usize)->(Self,RSState)
    {
        let rng = rand::rng();
        (
            RandomSearch(batch, rng),
            RSState{iteration:0}
        )
    }
}

impl <Obj, Cod, Out, Sp, const DIM: usize> Optimizer<Obj, Obj, Cod, Out, Sp, RSInfo, EmptyInfo, RSState, DIM> for RandomSearch
where
    Obj: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Sp: Searchspace<Obj, Obj, Cod, Out, EmptyInfo, DIM>,
{
    fn step(
        &mut self,
        _x: tantale_core::optimizer::opt::ArcSol<Obj, Cod, Out, EmptyInfo, DIM>,
        sp: &Sp,
        state:&mut RSState,
        pid:u32,
    ) -> OptOutput<Obj, Obj, Cod, Out, RSInfo, EmptyInfo, DIM> {
        state.iteration += 1;
        let info = RSInfo{iteration:state.iteration};
        let samples = sp.vec_sample_obj(&mut self.1, pid, self.0);
        let sol_obj = ArcSol::from(samples);
        let sol_opt = sol_obj.clone();
        OptOutput::new(sol_obj, sol_opt, info)
    }
}