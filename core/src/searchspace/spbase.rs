use crate::{
    domain::{Domain, TypeDom},
    optimizer::ArcVecArc,
    saver::CSVLeftRight,
    searchspace::{Searchspace, SolInfo},
    solution::{Id, Partial, Solution},
    variable::Var,
};

use rand::prelude::ThreadRng;
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct Sp<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub variables: Box<[Var<Obj, Opt>]>,
}

impl<SolId, Obj, Opt, SInfo> Searchspace<SolId, Obj, Opt, SInfo> for Sp<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    SInfo: SolInfo,
    SolId: Id + PartialEq + Clone + Copy,
{
    fn onto_obj(&self, inp: Arc<Partial<SolId, Opt, SInfo>>) -> Arc<Partial<SolId, Obj, SInfo>> {
        let outx: Vec<TypeDom<Obj>> = inp
            .x
            .iter()
            .zip(&self.variables)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();

        Arc::new(inp.twin(outx))
    }

    fn onto_opt(&self, inp: Arc<Partial<SolId, Obj, SInfo>>) -> Arc<Partial<SolId, Opt, SInfo>> {
        let outx: Vec<TypeDom<Opt>> = inp
            .x
            .iter()
            .zip(&self.variables)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();

        Arc::new(inp.twin(outx))
    }

    fn sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        info: Arc<SInfo>,
    ) -> Arc<Partial<SolId, Obj, SInfo>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<TypeDom<Obj>> = self.variables.iter().map(|v| v.sample_obj(rn)).collect();
        Arc::new(Partial::<SolId, Obj, SInfo>::new(
            SolId::generate(),
            outx,
            info,
        ))
    }

    fn sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        info: Arc<SInfo>,
    ) -> Arc<Partial<SolId, Opt, SInfo>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<TypeDom<Opt>> = self.variables.iter().map(|v| v.sample_opt(rn)).collect();
        Arc::new(Partial::<SolId, Opt, SInfo>::new(
            SolId::generate(),
            outx,
            info,
        ))
    }

    fn vec_onto_obj(
        &self,
        inp: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    ) -> ArcVecArc<Partial<SolId, Obj, SInfo>> {
        Arc::new(inp.iter().map(|i| self.onto_obj(i.clone())).collect())
    }

    fn vec_onto_opt(
        &self,
        inp: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    ) -> ArcVecArc<Partial<SolId, Opt, SInfo>> {
        Arc::new(inp.iter().map(|i| self.onto_opt(i.clone())).collect())
    }

    fn vec_sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> ArcVecArc<Partial<SolId, Obj, SInfo>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        Arc::new(
            (0..size)
                .map(|_| self.sample_obj(Some(rn), info.clone()))
                .collect(),
        )
    }

    fn vec_sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> ArcVecArc<Partial<SolId, Opt, SInfo>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        Arc::new(
            (0..size)
                .map(|_| self.sample_opt(Some(rn), info.clone()))
                .collect(),
        )
    }

    fn is_in_obj<S>(&self, inp: Arc<S>) -> bool
    where
        S: Solution<SolId, Obj, SInfo>,
    {
        inp.get_x()
            .iter()
            .zip(&self.variables)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn is_in_opt<S>(&self, inp: Arc<S>) -> bool
    where
        S: Solution<SolId, Opt, SInfo>,
    {
        inp.get_x()
            .iter()
            .zip(&self.variables)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    fn vec_is_in_obj<S>(&self, inp: ArcVecArc<S>) -> bool
    where
        S: Solution<SolId, Obj, SInfo> + Send + Sync,
    {
        inp.iter().all(|sol| self.is_in_obj(sol.clone()))
    }

    fn vec_is_in_opt<S>(&self, inp: ArcVecArc<S>) -> bool
    where
        S: Solution<SolId, Opt, SInfo> + Send + Sync,
    {
        inp.iter().all(|sol| self.is_in_opt(sol.clone()))
    }
}

impl<Obj, Opt> CSVLeftRight<Sp<Obj, Opt>, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>> for Sp<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, Obj::TypeDom, Opt::TypeDom>,
{
    fn header(elem: &Sp<Obj, Opt>) -> Vec<String> {
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
