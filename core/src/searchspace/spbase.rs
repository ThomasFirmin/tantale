use crate::{
    domain::{Domain, TypeDom},
    searchspace::{Searchspace, SolInfo},
    solution::{PartialSol, Solution,Partial},
    variable::Var,
};

use rand::prelude::ThreadRng;
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

#[cfg(feature = "par")]
use rayon::prelude::*;
#[cfg(feature = "par")]
use crate::searchspace::ParSearchspace;

pub struct Sp<Obj, Opt, const N: usize>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub variables: Box<[Var<Obj, Opt>]>,
}

impl<Obj, Opt, SInfo, const N: usize>
    Searchspace<PartialSol<Obj, SInfo, N>, PartialSol<Opt, SInfo, N>, Obj, Opt, SInfo>
    for Sp<Obj, Opt, N>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
{
    fn onto_obj(&self, inp: &PartialSol<Opt, SInfo, N>) -> PartialSol<Obj, SInfo, N> {
        let outx: Vec<TypeDom<Obj>> = inp
            .x
            .iter()
            .zip(&self.variables)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();

        inp.twin(outx)
    }

    fn onto_opt(&self, inp: &PartialSol<Obj, SInfo, N>) -> PartialSol<Opt, SInfo, N> {
        let outx: Vec<TypeDom<Opt>> = inp
            .x
            .iter()
            .zip(&self.variables)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();

        inp.twin(outx)
    }

    fn sample_obj(
        &self,
        rng: &mut ThreadRng,
        pid: u32,
        info: Arc<SInfo>,
    ) -> PartialSol<Obj, SInfo, N> {
        let outx: Vec<TypeDom<Obj>> = self.variables.iter().map(|v| v.sample_obj(rng)).collect();
        PartialSol::<Obj, SInfo, N>::new(pid, outx, info)
    }

    fn sample_opt(
        &self,
        rng: &mut ThreadRng,
        pid: u32,
        info: Arc<SInfo>,
    ) -> PartialSol<Opt, SInfo, N> {
        let outx: Vec<TypeDom<Opt>> = self.variables.iter().map(|v| v.sample_opt(rng)).collect();
        PartialSol::<Opt, SInfo, N>::new(pid, outx, info)
    }

    fn vec_onto_obj(&self, inp: &[PartialSol<Opt, SInfo, N>]) -> Vec<PartialSol<Obj, SInfo, N>> {
        inp.iter().map(|i| self.onto_obj(i)).collect()
    }

    fn vec_onto_opt(&self, inp: &[PartialSol<Obj, SInfo, N>]) -> Vec<PartialSol<Opt, SInfo, N>> {
        inp.iter().map(|i| self.onto_opt(i)).collect()
    }

    fn vec_sample_obj(
        &self,
        rng: &mut ThreadRng,
        pid: u32,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<PartialSol<Obj, SInfo, N>> {
        (0..size)
            .map(|_| self.sample_obj(rng, pid, info.clone()))
            .collect()
    }

    fn vec_sample_opt(
        &self,
        rng: &mut ThreadRng,
        pid: u32,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<PartialSol<Opt, SInfo, N>> {
        (0..size)
            .map(|_| self.sample_opt(rng, pid, info.clone()))
            .collect()
    }

    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<Obj, SInfo>,
    {
        inp.get_x()
            .iter()
            .zip(&self.variables)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<Opt, SInfo>,
    {
        inp.get_x()
            .iter()
            .zip(&self.variables)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<Obj, SInfo>,
    {
        inp.iter().all(|sol| self.is_in_obj(sol))
    }

    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<Opt, SInfo>,
    {
        inp.iter().all(|sol| self.is_in_opt(sol))
    }
}

#[cfg(feature = "par")]
impl<Obj, Opt, SInfo, const N: usize>
    ParSearchspace<PartialSol<Obj, SInfo, N>, PartialSol<Opt, SInfo, N>, Obj, Opt, SInfo>
    for Sp<Obj, Opt, N>
where
    Obj: Domain + Clone + Display + Debug + Send + Sync,
    Opt: Domain + Clone + Display + Debug + Send + Sync,
    Obj::TypeDom: Default + Copy + Clone + Display + Debug + Send + Sync,
    Opt::TypeDom: Default + Copy + Clone + Display + Debug + Send + Sync,
    SInfo: SolInfo + Send + Sync,
{
    fn par_onto_obj(&self, inp: &PartialSol<Opt, SInfo, N>) -> PartialSol<Obj, SInfo, N> {
        let var_it = self.variables.par_iter();
        let outx: Vec<TypeDom<Obj>> = inp
            .x
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        inp.twin(outx)
    }

    fn par_onto_opt(&self, inp: &PartialSol<Obj, SInfo, N>) -> PartialSol<Opt, SInfo, N> {
        let var_it = self.variables.par_iter();
        let outx: Vec<TypeDom<Opt>> = inp
            .x
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();
        inp.twin(outx)
    }

    fn par_sample_obj(&self, pid: u32, info: Arc<SInfo>) -> PartialSol<Obj, SInfo, N> {
        let variter = self.variables.par_iter();
        let outx: Vec<TypeDom<Obj>> = variter
            .map_init(rand::rng, |rng, var| var.sample_obj(rng))
            .collect();
        PartialSol::<Obj, SInfo, N>::new(pid, outx, info)
    }

    fn par_sample_opt(&self, pid: u32, info: Arc<SInfo>) -> PartialSol<Opt, SInfo, N> {
        let variter = self.variables.par_iter();
        let outx: Vec<TypeDom<Opt>> = variter
            .map_init(rand::rng, |rng, var| var.sample_opt(rng))
            .collect();
        PartialSol::<Opt, SInfo, N>::new(pid, outx, info)
    }

    fn par_vec_onto_obj(
        &self,
        inp: &[PartialSol<Opt, SInfo, N>],
    ) -> Vec<PartialSol<Obj, SInfo, N>> {
        inp.par_iter().map(|sol| self.par_onto_obj(sol)).collect()
    }

    fn par_vec_onto_opt(
        &self,
        inp: &[PartialSol<Obj, SInfo, N>],
    ) -> Vec<PartialSol<Opt, SInfo, N>> {
        inp.par_iter().map(|sol| self.par_onto_opt(sol)).collect()
    }

    fn par_vec_sample_obj(
        &self,
        pid: u32,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<PartialSol<Obj, SInfo, N>> {
        (0..size)
            .into_par_iter()
            .map(|_| self.par_sample_obj(pid, info.clone()))
            .collect()
    }

    fn par_vec_sample_opt(
        &self,
        pid: u32,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<PartialSol<Opt, SInfo, N>> {
        (0..size)
            .into_par_iter()
            .map(|_| self.par_sample_opt(pid, info.clone()))
            .collect()
    }

    fn par_is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<Obj, SInfo>,
    {
        let variter = self.variables.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn par_is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<Opt, SInfo>,
    {
        let variter = self.variables.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    fn par_vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<Obj, SInfo> + Send + Sync,
    {
        inp.par_iter().all(|sol| self.par_is_in_obj(sol))
    }

    fn par_vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<Opt, SInfo> + Send + Sync,
    {
        inp.par_iter().all(|sol| self.par_is_in_opt(sol))
    }
}
