use crate::{
    domain::{Domain, TypeDom},
    optimizer::VecArc,
    recorder::csv::CSVLeftRight,
    searchspace::{Searchspace, SolInfo},
    solution::{Id, Partial, Solution},
    variable::Var,
    Onto, Sp,
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

use rayon::prelude::*;

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct ParSp<Obj: Domain, Opt: Domain> {
    pub variables: Box<[Var<Obj, Opt>]>,
}

impl<PSol, SolId, Obj, Opt, SInfo> Searchspace<PSol, SolId, Obj, Opt, SInfo> for ParSp<Obj, Opt>
where
    PSol: Partial<SolId, Obj, SInfo> + Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol> + Send + Sync,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom> + Send + Sync,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom> + Send + Sync,
    TypeDom<Obj>: Send + Sync,
    TypeDom<Opt>: Send + Sync,
    SInfo: SolInfo + Send + Sync,
    SolId: Id + Send + Sync,
{
    fn onto_obj(&self, inp: Arc<PSol::Twin<Opt>>) -> Arc<PSol> {
        let var_it = self.variables.par_iter();
        let outx: Vec<TypeDom<Obj>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        Arc::new(inp.twin::<Obj, _>(outx))
    }

    fn onto_opt(&self, inp: Arc<PSol>) -> Arc<PSol::Twin<Opt>> {
        let var_it = self.variables.par_iter();
        let outx: Vec<TypeDom<Opt>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();
        Arc::new(inp.twin::<Opt, _>(outx))
    }

    /// [`None`] should be used for `_rng`.
    fn sample_obj(&self, _rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> Arc<PSol> {
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
    fn sample_opt(&self, _rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> Arc<PSol::Twin<Opt>> {
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

    fn vec_onto_obj(&self, inp: VecArc<PSol::Twin<Opt>>) -> VecArc<PSol> {
        inp.par_iter()
            .map(|sol| self.onto_obj(sol.clone()))
            .collect()
    }

    fn vec_onto_opt(&self, inp: VecArc<PSol>) -> VecArc<PSol::Twin<Opt>> {
        inp.par_iter()
            .map(|sol| self.onto_opt(sol.clone()))
            .collect()
    }

    /// [`None`] should be used for `_rng`.
    fn vec_sample_obj(
        &self,
        _rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> VecArc<PSol> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_obj(None, info.clone()))
            .collect()
    }

    /// [`None`] should be used for `_rng`.
    fn vec_sample_opt(
        &self,
        _rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> VecArc<PSol::Twin<Opt>> {
        (0..size)
            .into_par_iter()
            .map(|_| {
                <ParSp<Obj, Opt> as Searchspace<PSol, SolId, Obj, Opt, SInfo>>::sample_opt(
                    self,
                    None,
                    info.clone(),
                )
            })
            .collect()
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
        inp.iter().all(|sol| {
            <ParSp<Obj, Opt> as Searchspace<PSol, SolId, Obj, Opt, SInfo>>::is_in_obj::<S>(
                self,
                sol.clone(),
            )
        })
    }

    fn vec_is_in_opt<S>(&self, inp: VecArc<S>) -> bool
    where
        S: Solution<SolId, Opt, SInfo> + Send + Sync,
    {
        inp.iter().all(|sol| {
            <ParSp<Obj, Opt> as Searchspace<PSol, SolId, Obj, Opt, SInfo>>::is_in_opt::<S>(
                self,
                sol.clone(),
            )
        })
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
