use crate::{
    domain::{Domain, TypeDom},
    optimizer::VecArc,
    saver::CSVLeftRight,
    searchspace::{Searchspace, SolInfo},
    solution::{Id, Partial, Solution},
    variable::Var,
    Sp,
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

use rayon::prelude::*;

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct ParSp<Obj: Domain, Opt: Domain> {
    pub variables: Box<[Var<Obj, Opt>]>,
}

impl<SolId, Obj, Opt, SInfo> Searchspace<SolId, Obj, Opt, SInfo> for ParSp<Obj, Opt>
where
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    TypeDom<Obj>: Send + Sync,
    TypeDom<Opt>: Send + Sync,
    SInfo: SolInfo + Send + Sync,
    SolId: Id + Send + Sync,
{
    fn onto_obj(&self, inp: Arc<Partial<SolId, Opt, SInfo>>) -> Arc<Partial<SolId, Obj, SInfo>> {
        let var_it = self.variables.par_iter();
        let outx: Vec<TypeDom<Obj>> = inp
            .x
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        Arc::new(inp.twin(outx))
    }

    fn onto_opt(&self, inp: Arc<Partial<SolId, Obj, SInfo>>) -> Arc<Partial<SolId, Opt, SInfo>> {
        let var_it = self.variables.par_iter();
        let outx: Vec<TypeDom<Opt>> = inp
            .x
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();
        Arc::new(inp.twin(outx))
    }

    /// [`None`] should be used for `_rng`.
    fn sample_obj(
        &self,
        _rng: Option<&mut ThreadRng>,
        info: Arc<SInfo>,
    ) -> Arc<Partial<SolId, Obj, SInfo>> {
        let variter = self.variables.par_iter();
        let outx: Vec<TypeDom<Obj>> = variter
            .map_init(rand::rng, |rng, var| var.sample_obj(rng))
            .collect();
        Arc::new(Partial::<SolId, Obj, SInfo>::new(
            SolId::generate(),
            outx,
            info,
        ))
    }

    /// [`None`] should be used for `_rng`.
    fn sample_opt(
        &self,
        _rng: Option<&mut ThreadRng>,
        info: Arc<SInfo>,
    ) -> Arc<Partial<SolId, Opt, SInfo>> {
        let variter = self.variables.par_iter();
        let outx: Vec<TypeDom<Opt>> = variter
            .map_init(rand::rng, |rng, var| var.sample_opt(rng))
            .collect();
        Arc::new(Partial::<SolId, Opt, SInfo>::new(
            SolId::generate(),
            outx,
            info,
        ))
    }

    fn vec_onto_obj(
        &self,
        inp: VecArc<Partial<SolId, Opt, SInfo>>,
    ) -> VecArc<Partial<SolId, Obj, SInfo>> {
        inp.par_iter().map(|sol| self.onto_obj(sol.clone())).collect()
    }

    fn vec_onto_opt(
        &self,
        inp: VecArc<Partial<SolId, Obj, SInfo>>,
    ) -> VecArc<Partial<SolId, Opt, SInfo>> {
        inp.par_iter().map(|sol| self.onto_opt(sol.clone())).collect()
    }

    /// [`None`] should be used for `_rng`.
    fn vec_sample_obj(
        &self,
        _rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> VecArc<Partial<SolId, Obj, SInfo>> {
        (0..size).into_par_iter().map(|_| self.sample_obj(None, info.clone())).collect()
    }

    /// [`None`] should be used for `_rng`.
    fn vec_sample_opt(
        &self,
        _rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> VecArc<Partial<SolId, Opt, SInfo>> {
        (0..size).into_par_iter().map(|_| self.sample_opt(None, info.clone())).collect()
    }

    fn is_in_obj<S>(&self, inp: Arc<S>) -> bool
    where
        S: Solution<SolId, Obj, SInfo> + Send + Sync,
    {
        let variter = self.variables.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn is_in_opt<S>(&self, inp: Arc<S>) -> bool
    where
        S: Solution<SolId, Opt, SInfo> + Send + Sync,
    {
        let variter = self.variables.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    fn vec_is_in_obj<S>(&self, inp: VecArc<S>) -> bool
    where
        S: Solution<SolId, Obj, SInfo> + Send + Sync,
    {
        inp.par_iter().all(|sol| self.is_in_obj(sol.clone()))
    }

    fn vec_is_in_opt<S>(&self, inp: VecArc<S>) -> bool
    where
        S: Solution<SolId, Opt, SInfo> + Send + Sync,
    {
        inp.par_iter().all(|sol| self.is_in_opt(sol.clone()))
    }
}

impl<Obj: Domain, Opt: Domain> From<Sp<Obj, Opt>> for ParSp<Obj, Opt> {
    fn from(value: Sp<Obj, Opt>) -> Self {
        ParSp {
            variables: value.variables,
        }
    }
}

impl<Obj, Opt> CSVLeftRight<ParSp<Obj, Opt>, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>>
    for ParSp<Obj, Opt>
where
    Obj: Domain,
    Opt: Domain,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, Obj::TypeDom, Opt::TypeDom>,
{
    fn header(elem: &ParSp<Obj, Opt>) -> Vec<String> {
        elem.variables
            .iter()
            .flat_map(Var::<Obj, Opt>::header)
            .collect()
    }

    fn write_left(&self, comp: &Arc<[Obj::TypeDom]>) -> Vec<String> {
        let var_iter = self.variables.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    fn write_right(&self, comp: &Arc<[Opt::TypeDom]>) -> Vec<String> {
        let var_iter = self.variables.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}
