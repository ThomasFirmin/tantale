use crate::{
    Sp,
    domain::{Domain, NoDomain, PreDomain, onto::{LinkTyObj, LinkTyOpt, Linked, OntoDom}},
    recorder::csv::{CSVLeftRight, CSVWritable},
    searchspace::{Searchspace, SolInfo},
    solution::{Id, Lone, Pair, Partial, Solution},
    variable::Var
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

use rayon::prelude::*;
use rayon::iter::{IntoParallelIterator,IntoParallelRefIterator};

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct SpPar<Obj: Domain, Opt: PreDomain> {
    pub var: Box<[Var<Obj, Opt>]>,
}

impl<Obj:OntoDom<Opt>, Opt:OntoDom<Obj>> Linked for SpPar<Obj,Opt> 
{
    type Obj = Obj;
    type Opt = Opt;
}

impl<Obj:Domain> Linked for SpPar<Obj,NoDomain> 
{
    type Obj = Obj;
    type Opt = Obj;
}

impl<PartObj,PartOpt,SolId, Obj, Opt, SInfo> Searchspace<PartObj,PartOpt,SolId, SInfo> for SpPar<Obj, Opt>
where
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj: OntoDom<Opt> + Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    PartObj: Partial<SolId,Obj,SInfo, Twin<Opt> = PartOpt, TwinP<Opt> = PartOpt, Raw=Arc<[Obj::TypeDom]>> + Send  + Sync,
    PartOpt: Partial<SolId,Opt,SInfo, Twin<Obj> = PartObj, TwinP<Obj> = PartObj, Raw=Arc<[Opt::TypeDom]>> + Send  + Sync,
    Obj::TypeDom: Send + Sync,
    Opt::TypeDom: Send + Sync,
{
    type PartShape = Pair<PartObj,PartOpt,SolId,Obj,Opt,SInfo>;

    fn onto_obj(&self, inp: &PartOpt) -> PartObj
    {
        let var_it = self.var.par_iter();
        let outx: Vec<LinkTyObj<Self>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        inp.twin::<Obj>(outx.into())
    }

    fn onto_opt(&self,inp: &PartObj) -> PartOpt
    {
        let var_it = self.var.par_iter();
        let outx: Vec<LinkTyOpt<Self>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();
        inp.twin::<Opt>(outx.into())
    }

    fn sample_obj(&self, _rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> PartObj{
        let variter = self.var.par_iter();
        let outx: Vec<LinkTyObj<Self>> = variter
            .map_init(rand::rng, |rng, var| var.sample_obj(rng))
            .collect();
        Partial::new(SolId::generate(), outx, info)
    }
    
    fn sample_opt(&self, _rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> PartOpt {
        let variter = self.var.par_iter();
        let outx: Vec<LinkTyOpt<Self>> = variter
            .map_init(rand::rng, |rng, var| var.sample_opt(rng))
            .collect();
        Partial::new(SolId::generate(), outx, info)
    }
    
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = <PartObj as Solution<SolId,Self::Obj,SInfo>>::Raw> + Send + Sync
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }
    
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw =<PartOpt as Solution<SolId,Opt,SInfo>>::Raw> + Send + Sync
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }
    
    fn vec_onto_obj(&self, inp: &Vec<PartOpt>) -> Vec<PartObj> {
        inp.par_iter().map(|sol| self.onto_obj(sol)).collect()
    }
    
    fn vec_onto_opt(&self, inp: &Vec<PartObj>) -> Vec<PartOpt> {
        inp.par_iter().map(|sol| self.onto_opt(sol)).collect()
    }
    
    fn vec_sample_obj(&self,_rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<PartObj> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_obj(None, info.clone()))
            .collect()
    }
    
    fn vec_sample_opt(&self,_rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<PartOpt> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_opt(None, info.clone()))
            .collect()
    }
    
    fn vec_is_in_obj<S>(&self, inp: &Vec<S>) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync 
    {
        inp.par_iter().all(|sol| {<SpPar<Obj, Opt> as Searchspace<PartObj, PartOpt, SolId, SInfo>>::is_in_obj::<S>(self, sol)})
    }
    
    fn vec_is_in_opt<S>(&self, inp: &Vec<S>) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw =Arc<[Opt::TypeDom]>> + Send + Sync 
    {
        inp.par_iter().all(|sol| {<SpPar<Obj, Opt> as Searchspace<PartObj, PartOpt, SolId, SInfo>>::is_in_opt::<S>(self, sol)})
    }

    fn sample_pair(&self,_rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<Self::PartShape>
    {
        (0..size).into_par_iter().map(
            |_|
            {
                let s = self.sample_obj(None, info.clone()); // sample
                let c = self.onto_opt(&s); // converted
                (s,c).into() // obj,opt
            }
        ).collect()
    }
}

impl<PartObj, SolId, Obj, SInfo> Searchspace<PartObj,PartObj,SolId, SInfo> for SpPar<Obj, NoDomain>
where
    SolId: Id + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Obj: Domain + Send + Sync,
    PartObj: Partial<SolId,Obj,SInfo, Twin<Obj> = PartObj, TwinP<Obj> = PartObj, Raw=Arc<[Obj::TypeDom]>> + Send + Sync,
    Obj::TypeDom: Send + Sync,
{
    type PartShape = Lone<PartObj,SolId,Obj,SInfo>;

    fn onto_obj(&self, inp: &PartObj) -> PartObj {
        inp.twin::<Obj>(inp.get_x())
    }

    fn onto_opt(&self,inp: &PartObj) -> PartObj {
        inp.twin::<Obj>(inp.get_x())
    }

    fn sample_obj(&self, _rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> PartObj {
        let variter = self.var.par_iter();
        let outx: Vec<LinkTyObj<Self>> = variter
            .map_init(rand::rng, |rng, var| var.sample_obj(rng))
            .collect();
        Partial::new(SolId::generate(), outx, info)
    }
    
    fn sample_opt(&self, _rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> PartObj {
        let variter = self.var.par_iter();
        let outx: Vec<LinkTyOpt<Self>> = variter
            .map_init(rand::rng, |rng, var| var.sample_opt(rng))
            .collect();
        Partial::new(SolId::generate(), outx, info)
    }
    
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }
    
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Obj::TypeDom]>>
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }
    
    fn vec_onto_obj(&self, inp: &Vec<PartObj>) -> Vec<PartObj> {
        inp.par_iter().map(|sol| self.onto_obj(sol)).collect()
    }
    
    fn vec_onto_opt(&self, inp: &Vec<PartObj>) -> Vec<PartObj> {
        inp.par_iter().map(|sol| self.onto_opt(sol)).collect()
    }
    
    fn vec_sample_obj(&self,_rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<PartObj> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_obj(None, info.clone()))
            .collect()
    }
    
    fn vec_sample_opt(&self,_rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<PartObj> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_opt(None, info.clone()))
            .collect()
    }
    
    fn vec_is_in_obj<S>(&self, inp: &Vec<S>) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync
    {
        inp.par_iter().all(|sol| {<SpPar<Obj, NoDomain> as Searchspace<PartObj, PartObj, SolId, SInfo>>::is_in_obj::<S>(self, sol)})
    }
    
    fn vec_is_in_opt<S>(&self, inp: &Vec<S>) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync
    {
        inp.par_iter().all(|sol| {<SpPar<Obj, NoDomain> as Searchspace<PartObj, PartObj, SolId, SInfo>>::is_in_obj::<S>(self, sol)})
    }

    fn sample_pair(&self,_rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<Self::PartShape>
    {
        (0..size).into_par_iter().map(
            |_|
            {
                let s = self.sample_obj(None, info.clone()); // sample
                Lone::new(s)
            }
        ).collect()
    }
}



impl<Obj: Domain, Opt: Domain> From<Sp<Obj, Opt>> for SpPar<Obj, Opt> {
    fn from(value: Sp<Obj, Opt>) -> Self {
        SpPar {
            var: value.var,
        }
    }
}

impl<Obj, Opt> CSVLeftRight<SpPar<Obj, Opt>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>> for SpPar<Obj, Opt>
where
    Obj: OntoDom<Opt> + CSVWritable<(), LinkTyObj<Self>>,
    Opt: OntoDom<Obj> + CSVWritable<(), LinkTyOpt<Self>>,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    fn header(elem: &SpPar<Obj, Opt>) -> Vec<String> {
        elem.var
            .iter()
            .flat_map(Var::<Obj, Opt>::header)
            .collect()
    }

    fn write_left(&self, comp: &Arc<[LinkTyObj<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    fn write_right(&self, comp: &Arc<[LinkTyOpt<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}

impl<Obj> CSVLeftRight<SpPar<Obj, NoDomain>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>> for SpPar<Obj, NoDomain>
where
    Obj: Domain + CSVWritable<(), LinkTyObj<Self>>,
    Var<Obj, NoDomain>: CSVLeftRight<Var<Obj, NoDomain>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    fn header(elem: &SpPar<Obj, NoDomain>) -> Vec<String> {
        elem.var
            .iter()
            .flat_map(Var::<Obj, NoDomain>::header)
            .collect()
    }

    fn write_left(&self, comp: &Arc<[LinkTyObj<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    fn write_right(&self, comp: &Arc<[LinkTyOpt<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}