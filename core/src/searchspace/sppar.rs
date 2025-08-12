use crate::{
    domain::{Domain, TypeDom},
    optimizer::ArcVecArc,
    saver::CSVLeftRight,
    searchspace::{Searchspace, SolInfo},
    solution::{Id, Partial, PartialSol, Solution},
    variable::Var,
};

use rand::prelude::ThreadRng;
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use rayon::prelude::*;

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct SpPar<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub variables: Box<[Var<Obj, Opt>]>,
}

impl<SolId, Obj, Opt, SInfo>
    Searchspace<
        SolId,
        PartialSol<SolId, Obj, SInfo>,
        PartialSol<SolId, Opt, SInfo>,
        Obj,
        Opt,
        SInfo,
    > for SpPar<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug + Send + Sync,
    Opt: Domain + Clone + Display + Debug + Send + Sync,
    Obj::TypeDom: Default + Copy + Clone + Display + Debug + Send + Sync,
    Opt::TypeDom: Default + Copy + Clone + Display + Debug + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    SolId: Id + PartialEq + Copy + Clone + Send + Sync,
{
    fn onto_obj(
        &self,
        inp: Arc<PartialSol<SolId, Opt, SInfo>>,
    ) -> Arc<PartialSol<SolId, Obj, SInfo>> {
        let var_it = self.variables.par_iter();
        let outx: Vec<TypeDom<Obj>> = inp
            .x
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        Arc::new(inp.twin(outx))
    }

    fn onto_opt(
        &self,
        inp: Arc<PartialSol<SolId, Obj, SInfo>>,
    ) -> Arc<PartialSol<SolId, Opt, SInfo>> {
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
    ) -> Arc<PartialSol<SolId, Obj, SInfo>> {
        let variter = self.variables.par_iter();
        let outx: Vec<TypeDom<Obj>> = variter
            .map_init(rand::rng, |rng, var| var.sample_obj(rng))
            .collect();
        Arc::new(PartialSol::<SolId, Obj, SInfo>::new(
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
    ) -> Arc<PartialSol<SolId, Opt, SInfo>> {
        let variter = self.variables.par_iter();
        let outx: Vec<TypeDom<Opt>> = variter
            .map_init(rand::rng, |rng, var| var.sample_opt(rng))
            .collect();
        Arc::new(PartialSol::<SolId, Opt, SInfo>::new(
            SolId::generate(),
            outx,
            info,
        ))
    }

    fn vec_onto_obj(
        &self,
        inp: ArcVecArc<PartialSol<SolId, Opt, SInfo>>,
    ) -> ArcVecArc<PartialSol<SolId, Obj, SInfo>> {
        Arc::new(
            inp.par_iter()
                .map(|sol| self.onto_obj(sol.clone()))
                .collect(),
        )
    }

    fn vec_onto_opt(
        &self,
        inp: ArcVecArc<PartialSol<SolId, Obj, SInfo>>,
    ) -> ArcVecArc<PartialSol<SolId, Opt, SInfo>> {
        Arc::new(
            inp.par_iter()
                .map(|sol| self.onto_opt(sol.clone()))
                .collect(),
        )
    }

    /// [`None`] should be used for `_rng`.
    fn vec_sample_obj(
        &self,
        _rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> ArcVecArc<PartialSol<SolId, Obj, SInfo>> {
        Arc::new(
            (0..size)
                .into_par_iter()
                .map(|_| self.sample_obj(None, info.clone()))
                .collect(),
        )
    }

    /// [`None`] should be used for `_rng`.
    fn vec_sample_opt(
        &self,
        _rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> ArcVecArc<PartialSol<SolId, Opt, SInfo>> {
        Arc::new(
            (0..size)
                .into_par_iter()
                .map(|_| self.sample_opt(None, info.clone()))
                .collect(),
        )
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

    fn vec_is_in_obj<S>(&self, inp: ArcVecArc<S>) -> bool
    where
        S: Solution<SolId, Obj, SInfo> + Send + Sync,
    {
        inp.par_iter().all(|sol| self.is_in_obj(sol.clone()))
    }

    fn vec_is_in_opt<S>(&self, inp: ArcVecArc<S>) -> bool
    where
        S: Solution<SolId, Opt, SInfo> + Send + Sync,
    {
        inp.par_iter().all(|sol| self.is_in_opt(sol.clone()))
    }
}

impl<Obj, Opt> CSVLeftRight<Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>> for SpPar<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Var<Obj, Opt>: CSVLeftRight<Obj::TypeDom, Opt::TypeDom>,
{
    fn header(&self) -> Vec<String> {
        self.variables.iter().flat_map(|v| v.header()).collect()
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
