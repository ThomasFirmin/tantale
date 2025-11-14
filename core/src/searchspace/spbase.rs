use crate::{
    domain::{Domain, TypeDom},
    recorder::csv::CSVLeftRight,
    searchspace::{ParSp, Searchspace, SolInfo},
    solution::{Id, Partial, Solution},
    variable::Var,
    Onto,
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct Sp<Obj, Opt>
where
    Obj: Domain,
    Opt: Domain,
{
    pub variables: Box<[Var<Obj, Opt>]>,
}

impl<PSol, SolId, Obj, Opt, SInfo> Searchspace<PSol, SolId, Obj, Opt, SInfo> for Sp<Obj, Opt>
where
    PSol: Partial<SolId, Obj, SInfo>,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo, Twin<Obj> = PSol>,
    Obj: Domain + Onto<Opt, TargetItem = Opt::TypeDom, Item = Obj::TypeDom>,
    Opt: Domain + Onto<Obj, TargetItem = Obj::TypeDom, Item = Opt::TypeDom>,
    SInfo: SolInfo,
    SolId: Id,
{
    fn onto_obj(&self, inp: &PSol::Twin<Opt>) -> PSol {
        let outx: Vec<TypeDom<Obj>> = inp
            .get_x()
            .iter()
            .zip(&self.variables)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();

        inp.twin::<Obj, _>(outx)
    }

    fn onto_opt(&self, inp: &PSol) -> PSol::Twin<Opt> {
        let outx: Vec<TypeDom<Opt>> = inp
            .get_x()
            .iter()
            .zip(&self.variables)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();

        inp.twin::<Opt, _>(outx)
    }

    fn sample_obj(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> PSol {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<TypeDom<Obj>> = self.variables.iter().map(|v| v.sample_obj(rn)).collect();
        Partial::<SolId, Obj, SInfo>::new(SolId::generate(),outx,info)
    }

    fn sample_opt(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> PSol::Twin<Opt> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<TypeDom<Opt>> = self.variables.iter().map(|v| v.sample_opt(rn)).collect();
        Partial::<SolId, Opt, SInfo>::new(SolId::generate(),outx,info)
    }

    fn vec_onto_obj(&self, inp: &[PSol::Twin<Opt>]) -> Vec<PSol> {
        inp.iter().map(|i| self.onto_obj(i)).collect()
    }

    fn vec_onto_opt(&self, inp: &[PSol]) -> Vec<PSol::Twin<Opt>> {
        inp.iter().map(|i| self.onto_opt(i)).collect()
    }

    fn vec_sample_obj(&self,rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<PSol> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| self.sample_obj(Some(rn), info.clone()))
            .collect()
    }

    fn vec_sample_opt(&self,rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<PSol::Twin<Opt>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| {
                <Sp<Obj, Opt> as Searchspace<PSol, SolId, Obj, Opt, SInfo>>::sample_opt(
                    self,
                    Some(rn),
                    info.clone(),
                )
            })
            .collect()
    }

    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Obj, SInfo>,
    {
        inp.get_x()
            .iter()
            .zip(&self.variables)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Opt, SInfo>,
    {
        inp.get_x()
            .iter()
            .zip(&self.variables)
            .all(|(elem, v)| v.is_in_opt(elem))
    }

    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Obj, SInfo> + Send + Sync,
    {
        inp.iter().all(|sol| {
            <Sp<Obj, Opt> as Searchspace<PSol, SolId, Obj, Opt, SInfo>>::is_in_obj::<S>(self,sol)
        })
    }

    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Opt, SInfo> + Send + Sync,
    {
        inp.iter().all(|sol| {
            <Sp<Obj, Opt> as Searchspace<PSol, SolId, Obj, Opt, SInfo>>::is_in_opt::<S>(self,sol)
        })
    }
}

impl<Obj: Domain, Opt: Domain> From<ParSp<Obj, Opt>> for Sp<Obj, Opt> {
    fn from(value: ParSp<Obj, Opt>) -> Self {
        Sp {
            variables: value.variables,
        }
    }
}

impl<Obj, Opt> CSVLeftRight<Sp<Obj, Opt>, Arc<[Obj::TypeDom]>, Arc<[Opt::TypeDom]>> for Sp<Obj, Opt>
where
    Obj: Domain,
    Opt: Domain,
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
