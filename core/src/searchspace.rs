use crate::domain::Domain;
use crate::solution::Solution;
use crate::variable::Var;

use std::fmt::{Debug, Display};
use rand::prelude::ThreadRng;

pub trait Searchspace<const N : usize, Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    fn onto_obj(&self, inp: &Solution<Opt, N>, out : &mut Solution<Obj, N>);
    fn onto_opt(&self, inp: &Solution<Obj, N>, out : &mut Solution<Opt, N>);
    fn sample_obj(&self, rng: &mut ThreadRng, out : &mut Solution<Obj, N>);
    fn sample_opt(&self, rng: &mut ThreadRng, out : &mut Solution<Opt, N>);
    fn onto_vec_obj(&self, inp: &Vec<Solution<Opt, N>>, out : &mut Vec<Solution<Obj, N>>);
    fn onto_vec_opt(&self, inp: &Vec<Solution<Obj, N>>, out : &mut Vec<Solution<Opt, N>>);
    fn sample_vec_obj(&self, rng: &mut ThreadRng, out : &mut Vec<Solution<Obj, N>>) ;
    fn sample_vec_opt(&self, rng: &mut ThreadRng, out : &mut Vec<Solution<Opt, N>>) ;
}

#[cfg(feature="par")]
pub trait ParSearchspace<const N : usize, Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    fn onto_par_obj(&self, inp: &Solution<Opt, N>, out : &mut Solution<Obj, N>);
    fn onto_par_opt(&self, inp: &Solution<Obj, N>, out : &mut Solution<Opt, N>);
    fn sample_par_obj(&self, out : &mut Solution<Obj, N>);
    fn sample_par_opt(&self, out : &mut Solution<Opt, N>);
    fn onto_par_vec_obj(&self, inp: &Vec<Solution<Opt, N>>, out : &mut Vec<Solution<Obj, N>>);
    fn onto_par_vec_opt(&self, inp: &Vec<Solution<Obj, N>>, out : &mut Vec<Solution<Opt, N>>);
    fn sample_par_vec_obj(&self, out : &mut Vec<Solution<Obj, N>>) ;
    fn sample_par_vec_opt(&self, out : &mut Vec<Solution<Opt, N>>) ;
}

pub struct Sp<const N : usize, Obj, Opt=Obj>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub variables: Vec<Var<Obj, Opt>>,
}

impl <const N : usize, Obj,Opt> Searchspace<N, Obj,Opt> for Sp<N, Obj,Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    fn onto_obj(&self, inp: &Solution<Opt, N>, out : &mut Solution<Obj, N>){
        out.x.iter_mut().zip(&inp.x).zip(&self.variables).for_each(
            |((o,i),v)|
            *o = v.onto_obj(i).unwrap()
        );
    }

    fn onto_opt(&self, inp: &Solution<Obj, N>, out : &mut Solution<Opt, N>){
        out.x.iter_mut().zip(&inp.x).zip(&self.variables).for_each(
            |((o,i),v)|
            *o = v.onto_opt(i).unwrap()
        );
    }
    
    fn sample_obj(&self, rng: &mut ThreadRng, out : &mut Solution<Obj, N>) {
        out.x.iter_mut().zip(&self.variables).for_each(
            |(inp,var)|
            *inp = var.sample_obj(rng)
        );
    }
    
    fn sample_opt(&self, rng: &mut ThreadRng, out : &mut Solution<Opt, N>) {
        out.x.iter_mut().zip(&self.variables).for_each(
            |(inp,var)|
            *inp = var.sample_opt(rng)
        );
    }
    
    fn onto_vec_obj(&self, inp: &Vec<Solution<Opt, N>>, out : &mut Vec<Solution<Obj, N>>) {
        out.iter_mut().zip(inp).for_each(|(o,i)| self.onto_obj(i,o));
    }
    
    fn onto_vec_opt(&self, inp: &Vec<Solution<Obj, N>>, out : &mut Vec<Solution<Opt, N>>) {
        out.iter_mut().zip(inp).for_each(|(o,i)| self.onto_opt(i,o));
    }
    
    fn sample_vec_obj(&self, rng: &mut ThreadRng, out : &mut Vec<Solution<Obj, N>>)  {
        out.iter_mut().for_each(|o| self.sample_obj(rng, o));
    }
    
    fn sample_vec_opt(&self, rng: &mut ThreadRng, out : &mut Vec<Solution<Opt, N>>)  {
        out.iter_mut().for_each(|o| self.sample_opt(rng, o));
    }
    
}

#[cfg(feature="par")]
use rayon::prelude::*;
#[cfg(feature="par")]
impl <const N : usize, Obj,Opt> ParSearchspace<N, Obj,Opt> for Sp<N, Obj,Opt>
where
    Obj: Domain + Clone + Display + Debug + Send + Sync,
    Opt: Domain + Clone + Display + Debug + Send + Sync,
    Obj::TypeDom : Default + Copy + Clone + Display + Debug + Send + Sync,
    Opt::TypeDom : Default + Copy + Clone + Display + Debug + Send + Sync,
{
    fn onto_par_obj(&self, inp: &Solution<Opt, N>, out : &mut Solution<Obj, N>){
        let inpiter = inp.x.par_iter();
        let variter = self.variables.par_iter();
        out.x.par_iter_mut().zip(inpiter).zip(variter).for_each(
            |((o,i),v)|
            *o = v.onto_obj(i).unwrap()
        );
    }

    fn onto_par_opt(&self, inp: &Solution<Obj, N>, out : &mut Solution<Opt, N>){
        let inpiter = inp.x.par_iter();
        let variter = self.variables.par_iter();
        out.x.par_iter_mut().zip(inpiter).zip(variter).for_each(
            |((o,i),v)|
            *o = v.onto_opt(i).unwrap()
        );
    }
    
    fn sample_par_obj(&self, out : &mut Solution<Obj, N>) {
        let variter = self.variables.par_iter();
        out.x.par_iter_mut().zip(variter).for_each_init(
            ||rand::rng(),
            |rng,(inp,var)|
            *inp = var.sample_obj(rng)
        );
    }
    
    fn sample_par_opt(&self, out : &mut Solution<Opt, N>) {
        let variter = self.variables.par_iter();
        out.x.par_iter_mut().zip(variter).for_each_init(
            ||rand::rng(),
            |rng,(inp,var)|
            *inp = var.sample_opt(rng)
        );
    }
    
    fn onto_par_vec_obj(&self, inp: &Vec<Solution<Opt, N>>, out : &mut Vec<Solution<Obj, N>>) {
        out.par_iter_mut().zip(inp).for_each(|(o,i)| self.onto_obj(i,o));
    }
    
    fn onto_par_vec_opt(&self, inp: &Vec<Solution<Obj, N>>, out : &mut Vec<Solution<Opt, N>>) {
        out.par_iter_mut().zip(inp).for_each(|(o,i)| self.onto_opt(i,o));
    }
    
    fn sample_par_vec_obj(&self, out : &mut Vec<Solution<Obj, N>>)  {
        out.par_iter_mut().for_each(|o| self.sample_par_obj(o));
    }
    
    fn sample_par_vec_opt(&self, out : &mut Vec<Solution<Opt, N>>)  {
        out.par_iter_mut().for_each(|o| self.sample_par_opt(o));
    }
}