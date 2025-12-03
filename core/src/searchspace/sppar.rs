use crate::{
    BasePartial, Sp, domain::{Domain, NoDomain, PreDomain, TypeDom, onto::{OntoDom, TwinDom, TwinObj, TwinOpt, TwinTyObj, TwinTyOpt}}, recorder::csv::{CSVLeftRight, CSVWritable}, searchspace::{Searchspace, SolInfo}, solution::{Id, Pair, Partial, Solution}, variable::Var
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

use rayon::prelude::*;

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct SpPar<Obj: Domain, Opt: PreDomain> {
    pub var: Box<[Var<Obj, Opt>]>,
}

impl<Obj:OntoDom<Opt>, Opt:OntoDom<Obj>> TwinDom for SpPar<Obj,Opt> 
{
    type Obj = Obj;
    type Opt = Opt;
}

impl<Obj:Domain> TwinDom for SpPar<Obj,NoDomain> 
{
    type Obj = Obj;
    type Opt = Obj;
}


impl<SolId, Obj, Opt, SInfo> Searchspace<BasePartial<SolId, Opt, SInfo>, SolId, SInfo> for SpPar<Obj, Opt>
where
    Obj: OntoDom<Opt> + Send + Sync,
    Obj::TypeDom: Send + Sync,
    Opt: OntoDom<Obj> + Send + Sync,
    Opt::TypeDom: Send + Sync,
    SInfo: SolInfo + Send + Sync,
    SolId: Id + Send + Sync,
{
    type ObjSol = <BasePartial<SolId, Opt, SInfo> as Partial<SolId,Opt,SInfo>>::TwinP<Obj>;
    fn onto_obj(
        &self,
        inp: &Self::ObjSol
    ) -> BasePartial<SolId, TwinObj<Self>, SInfo> 
    {
        let var_it = self.var.par_iter();
        let outx: Vec<TypeDom<Obj>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_obj(i).unwrap())
            .collect();
        inp.twin(outx)
    }

    fn onto_opt(
        &self,
        inp: &Self::ObjSol
    ) -> BasePartial<SolId, Opt, SInfo>
    {
        let var_it = self.var.par_iter();
        let outx: Vec<TypeDom<Opt>> = inp
            .get_x()
            .par_iter()
            .zip(var_it)
            .map(|(i, v)| v.onto_opt(i).unwrap())
            .collect();
        inp.twin(outx)
    }

    fn sample_obj(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> BasePartial<SolId, TwinObj<Self>, SInfo> {
        let variter = self.var.par_iter();
        let outx: Vec<TypeDom<Obj>> = variter
            .map_init(rand::rng, |rng, var| var.sample_obj(rng))
            .collect();
        BasePartial::<SolId, Obj, SInfo>::new(SolId::generate(), outx, info)
    }
    
    fn sample_opt(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> Self::ObjSol {
        let variter = self.var.par_iter();
        let outx: Vec<TypeDom<Opt>> = variter
            .map_init(rand::rng, |rng, var| var.sample_opt(rng))
            .collect();
        BasePartial::<SolId, Opt, SInfo>::new(SolId::generate(), outx, info)
    }
    
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, TwinObj<Self>, SInfo, Raw = <Self::ObjSol as Solution<SolId, TwinObj<Self>, SInfo>>::Raw>
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }
    
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, TwinOpt<Self>, SInfo, Raw = <BasePartial<SolId, Opt, SInfo> as Solution<SolId, Opt, SInfo>>::Raw>
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_opt(elem))
    }
    
    fn vec_onto_obj(&self, inp: &[BasePartial<SolId, Opt, SInfo>]) -> Vec<Self::ObjSol> {
        inp.par_iter().map(|sol| self.onto_obj(sol)).collect()
    }
    
    fn vec_onto_opt(&self, inp: &[Self::ObjSol]) -> Vec<BasePartial<SolId, Opt, SInfo>> {
        inp.par_iter().map(|sol| self.onto_opt(sol)).collect()
    }
    
    fn vec_sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::ObjSol> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_obj(None, info.clone()))
            .collect()
    }
    
    fn vec_sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<BasePartial<SolId, Opt, SInfo>> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_opt(None, info.clone()))
            .collect()
    }
    
    fn sample_pair(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Pair<BasePartial<SolId, Opt, SInfo>,SolId,Obj,Opt,SInfo>>
    {
        (0..size).into_par_iter().map(
            |_|
            {
                let s = self.sample_obj(None, info.clone()); // sample
                let c = self.onto_opt(&s); // converted
                (s,c) // obj,opt
            }
        ).collect()
    }
    
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, TwinObj<Self>, SInfo, Raw = <BasePartial<SolId, TwinObj<Self>, SInfo> as Solution<SolId, TwinObj<Self>, SInfo>>::Raw> + Send + Sync
    {
        inp.iter().all(|sol| {self.is_in_obj::<S>(sol)})
    }
    
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, TwinOpt<Self>, SInfo, Raw = <<BasePartial<SolId, TwinObj<Self>, SInfo> as Solution<SolId, TwinObj<Self>, SInfo>>::Twin<TwinOpt<Self>> as Solution<SolId, TwinOpt<Self>, SInfo>>::Raw> + Send + Sync
    {
        inp.iter().all(|sol| {self.is_in_opt::<S>(sol)})
    }
}

impl<SolId, Obj, SInfo> Searchspace<BasePartial<SolId, TwinOpt<Self>, SInfo>, SolId, SInfo> for SpPar<Obj, NoDomain>
where
    Obj: Domain + Send + Sync,
    Obj::TypeDom: Send + Sync,
    SInfo: SolInfo + Send + Sync,
    SolId: Id + Send + Sync,
{
    type ObjSol = <BasePartial<SolId, TwinOpt<Self>, SInfo> as Partial<SolId,TwinOpt<Self>,SInfo>>::TwinP<Obj>;

    fn onto_opt(&self, inp: &Self::ObjSol) -> BasePartial<SolId, TwinOpt<Self>, SInfo> {
        inp.twin(inp.get_x())
    }

    fn onto_obj(&self, inp: &BasePartial<SolId, TwinOpt<Self>, SInfo>) -> Self::ObjSol {
        inp.twin(inp.get_x())
    }

    fn sample_obj(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> Self::ObjSol {
        let variter = self.var.par_iter();
        let outx: Vec<TypeDom<Obj>> = variter
            .map_init(rand::rng, |rng, var| var.sample_obj(rng))
            .collect();
        Partial::<SolId, Obj, SInfo>::new(SolId::generate(), outx, info)
    }

    fn sample_opt(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> BasePartial<SolId, TwinOpt<Self>, SInfo> {
        let variter = self.var.par_iter();
        let outx: Vec<TypeDom<Obj>> = variter
            .map_init(rand::rng, |rng, var| var.sample_obj(rng))
            .collect();
        Partial::<SolId, Obj, SInfo>::new(SolId::generate(), outx, info)
    }

    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, TwinObj<Self>, SInfo, Raw = <BasePartial<SolId, TwinObj<Self>, SInfo> as Solution<SolId, TwinObj<Self>, SInfo>>::Raw>
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, TwinOpt<Self>, SInfo, Raw = <<Self::ObjSol as Solution<SolId, TwinObj<Self>, SInfo>>::Twin<TwinOpt<Self>> as Solution<SolId, TwinOpt<Self>, SInfo>>::Raw>
    {
        let variter = self.var.par_iter();
        inp.get_x()
            .par_iter()
            .zip(variter)
            .all(|(elem, v)| v.is_in_obj(elem))
    }

    fn vec_onto_obj(&self, inp: &[BasePartial<SolId, TwinOpt<Self>, SInfo>]) -> Vec<Self::ObjSol> {
        inp.par_iter().map(|sol| self.onto_obj(sol)).collect()
    }

    fn vec_onto_opt(&self, inp: &[Self::ObjSol]) -> Vec<BasePartial<SolId, TwinOpt<Self>, SInfo>> {
        inp.par_iter().map(|sol| self.onto_obj(sol)).collect()
    }

    fn vec_sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::ObjSol> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_obj(None, info.clone()))
            .collect()
    }

    fn vec_sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<BasePartial<SolId, TwinOpt<Self>, SInfo>> {
        (0..size)
            .into_par_iter()
            .map(|_| self.sample_obj(None, info.clone()))
            .collect()
    }

    fn sample_pair(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Pair<BasePartial<SolId, TwinOpt<Self>, SInfo>,SolId,Obj,TwinOpt<Self>,SInfo>>
    {
            (0..size).into_par_iter().map(
            |_|
            {
                let s = self.sample_obj(None, info.clone()); // sample
                let c = self.onto_opt(&s); // converted
                (s,c) // obj,opt
            }
        ).collect()
    }

    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, TwinObj<Self>, SInfo, Raw = <BasePartial<SolId, TwinObj<Self>, SInfo> as Solution<SolId, TwinObj<Self>, SInfo>>::Raw> + Send + Sync
    {
        inp.iter().all(|sol| {self.is_in_obj::<S>(sol)})
    }
    
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, TwinOpt<Self>, SInfo, Raw = <<Self::ObjSol as Solution<SolId, TwinObj<Self>, SInfo>>::Twin<TwinOpt<Self>> as Solution<SolId, TwinOpt<Self>, SInfo>>::Raw> + Send + Sync
    {
        inp.iter().all(|sol| {self.is_in_obj::<S>(sol)})
    }
    
}



impl<Obj: Domain, Opt: Domain> From<Sp<Obj, Opt>> for SpPar<Obj, Opt> {
    fn from(value: Sp<Obj, Opt>) -> Self {
        SpPar {
            var: value.var,
        }
    }
}

impl<Obj, Opt> CSVLeftRight<SpPar<Obj, Opt>, Arc<[TwinTyObj<Self>]>, Arc<[TwinTyOpt<Self>]>> for SpPar<Obj, Opt>
where
    Obj: OntoDom<Opt> + CSVWritable<(), TwinTyObj<Self>>,
    Opt: OntoDom<Obj> + CSVWritable<(), TwinTyOpt<Self>>,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, TwinTyObj<Self>, TwinTyOpt<Self>>,
{
    fn header(elem: &SpPar<Obj, Opt>) -> Vec<String> {
        elem.var
            .iter()
            .flat_map(Var::<Obj, Opt>::header)
            .collect()
    }

    fn write_left(&self, comp: &Arc<[TwinTyObj<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    fn write_right(&self, comp: &Arc<[TwinTyOpt<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}

impl<Obj> CSVLeftRight<SpPar<Obj, NoDomain>, Arc<[TwinTyObj<Self>]>, Arc<[TwinTyOpt<Self>]>> for SpPar<Obj, NoDomain>
where
    Obj: Domain + CSVWritable<(), TwinTyObj<Self>>,
    Var<Obj, NoDomain>: CSVLeftRight<Var<Obj, NoDomain>, TwinTyObj<Self>, TwinTyOpt<Self>>,
{
    fn header(elem: &SpPar<Obj, NoDomain>) -> Vec<String> {
        elem.var
            .iter()
            .flat_map(Var::<Obj, NoDomain>::header)
            .collect()
    }

    fn write_left(&self, comp: &Arc<[TwinTyObj<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_left(c))
            .collect()
    }

    fn write_right(&self, comp: &Arc<[TwinTyOpt<Self>]>) -> Vec<String> {
        let var_iter = self.var.iter();
        comp.iter()
            .zip(var_iter)
            .flat_map(|(c, v)| v.write_right(c))
            .collect()
    }
}