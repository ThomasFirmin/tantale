use crate::{
    BasePartial, domain::{
        Domain, PreDomain, NoDomain,
        onto::{OntoDom, TwinDom, TwinObj, TwinOpt, TwinTyObj, TwinTyOpt}
    }, recorder::csv::{CSVLeftRight, CSVWritable}, searchspace::{Searchspace, SolInfo, SpPar}, solution::{Id, Pair, Partial, Solution}, variable::Var
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct Sp<Obj: Domain, Opt: PreDomain>
{
    pub var: Box<[Var<Obj,Opt>]>,
}

impl<Obj:OntoDom<Opt>, Opt:OntoDom<Obj>> TwinDom for Sp<Obj,Opt> 
{
    type Obj = Obj;
    type Opt = Opt;
}
impl<Obj:Domain> TwinDom for Sp<Obj,NoDomain> 
{
    type Obj = Obj;
    type Opt = Obj;
}

impl<SolId, Obj, Opt, SInfo> Searchspace<BasePartial<SolId, Opt, SInfo>, SolId, SInfo> for Sp<Obj, Opt>
where
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SInfo: SolInfo,
    SolId: Id,
{
    type ObjSol = <BasePartial<SolId, Opt, SInfo> as Partial<SolId,Opt,SInfo>>::TwinP<Obj>;
    fn onto_obj(
        &self,
        inp: &Self::ObjSol
    ) -> BasePartial<SolId, TwinObj<Self>, SInfo> 
    {
        let outx:Vec<TwinTyObj<Self>> = self.var.iter().zip(inp.get_x().iter()).map(
            |(v,xi)|
            {
                v.onto_obj(xi).unwrap()
            }
        ).collect();
        inp.twin(outx)
    }

    fn onto_opt(
        &self,
        inp: &Self::ObjSol
    ) -> BasePartial<SolId, Opt, SInfo>
    {
        let outx:Vec<TwinTyOpt<Self>> = self.var.iter().zip(inp.get_x().iter()).map(
            |(v,xi)|
            {
                v.onto_opt(xi).unwrap()
            }
        ).collect();
        inp.twin(outx)
    }

    fn sample_obj(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> BasePartial<SolId, TwinObj<Self>, SInfo> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_obj(rn)).collect();
        BasePartial::<SolId, TwinObj<Self>, SInfo>::new(SolId::generate(), outx, info)
    }
    
    fn sample_opt(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> Self::ObjSol {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_opt(rn)).collect();
        BasePartial::<SolId, TwinOpt<Self>, SInfo>::new(SolId::generate(), outx, info)
    }
    
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, TwinObj<Self>, SInfo, Raw = <Self::ObjSol as Solution<SolId, TwinObj<Self>, SInfo>>::Raw>
    {
        inp.get_x().iter().zip(&self.var).all(|(elem, v)| v.is_in_obj(elem))
    }
    
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, TwinOpt<Self>, SInfo, Raw = <BasePartial<SolId, Opt, SInfo> as Solution<SolId, Opt, SInfo>>::Raw>
    {
        inp.get_x().iter().zip(&self.var).all(|(elem, v)| v.is_in_opt(elem))
    }
    
    fn vec_onto_obj(&self, inp: &[BasePartial<SolId, Opt, SInfo>]) -> Vec<Self::ObjSol> {
        inp.iter().map(|i| self.onto_obj(i)).collect()
    }
    
    fn vec_onto_opt(&self, inp: &[Self::ObjSol]) -> Vec<BasePartial<SolId, Opt, SInfo>> {
        inp.iter().map(|i| self.onto_opt(i)).collect()
    }
    
    fn vec_sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::ObjSol> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| self.sample_obj(Some(rn), info.clone()))
            .collect()
    }
    
    fn vec_sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<BasePartial<SolId, Opt, SInfo>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| self.sample_opt(Some(rn), info.clone()))
            .collect()
    }
    
    fn sample_pair(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Pair<BasePartial<SolId, Opt, SInfo>,SolId,Obj,Opt,SInfo>>
    {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size).map(
            |_|
            {
                let s = self.sample_obj(Some(rn), info.clone()); // sample
                let c = self.onto_opt(&s); // converted
                (s,c) // obj,opt
            }
        ).collect()
    }
    
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, TwinObj<Self>, SInfo, Raw = <BasePartial<SolId, TwinObj<Self>, SInfo> as Solution<SolId, TwinObj<Self>, SInfo>>::Raw> + Send + Sync
    {
        inp.iter().all(|sol| {
            self.is_in_obj::<S>(sol)
        })
    }
    
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, TwinOpt<Self>, SInfo, Raw = <<BasePartial<SolId, TwinObj<Self>, SInfo> as Solution<SolId, TwinObj<Self>, SInfo>>::Twin<TwinOpt<Self>> as Solution<SolId, TwinOpt<Self>, SInfo>>::Raw> + Send + Sync
    {
        inp.iter().all(|sol| {
            self.is_in_opt::<S>(sol)
        })
    }
}

impl<SolId, Obj, SInfo> Searchspace<BasePartial<SolId, TwinOpt<Self>, SInfo>, SolId, SInfo> for Sp<Obj, NoDomain>
where
    Obj: Domain,
    SInfo: SolInfo,
    SolId: Id,
{
    type ObjSol = <BasePartial<SolId, TwinOpt<Self>, SInfo> as Partial<SolId,TwinOpt<Self>,SInfo>>::TwinP<Obj>;

    fn onto_opt(&self, inp: &Self::ObjSol) -> BasePartial<SolId, TwinOpt<Self>, SInfo> {
        inp.twin(inp.get_x())
    }

    fn onto_obj(&self, inp: &BasePartial<SolId, TwinOpt<Self>, SInfo>) -> Self::ObjSol {
        inp.twin(inp.get_x())
    }

    fn sample_obj(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> Self::ObjSol {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_obj(rn)).collect();
        BasePartial::<SolId, TwinObj<Self>, SInfo>::new(SolId::generate(), outx, info)
    }

    fn sample_opt(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> BasePartial<SolId, TwinOpt<Self>, SInfo> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_opt(rn)).collect();
        BasePartial::<SolId, TwinOpt<Self>, SInfo>::new(SolId::generate(), outx, info)
    }

    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, TwinObj<Self>, SInfo, Raw = <BasePartial<SolId, TwinObj<Self>, SInfo> as Solution<SolId, TwinObj<Self>, SInfo>>::Raw>
    {
        inp.get_x().iter().zip(&self.var).all(|(elem, v)| v.is_in_obj(elem))
    }

    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, TwinOpt<Self>, SInfo, Raw = <<Self::ObjSol as Solution<SolId, TwinObj<Self>, SInfo>>::Twin<TwinOpt<Self>> as Solution<SolId, TwinOpt<Self>, SInfo>>::Raw>
    {
        inp.get_x().iter().zip(&self.var).all(|(elem, v)| v.is_in_opt(elem))
    }

    fn vec_onto_obj(&self, inp: &[BasePartial<SolId, TwinOpt<Self>, SInfo>]) -> Vec<Self::ObjSol> {
        inp.iter().map(|i| self.onto_obj(i)).collect()
    }

    fn vec_onto_opt(&self, inp: &[Self::ObjSol]) -> Vec<BasePartial<SolId, TwinOpt<Self>, SInfo>> {
        inp.iter().map(|i| self.onto_opt(i)).collect()
    }

    fn vec_sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::ObjSol> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| self.sample_obj(Some(rn), info.clone()))
            .collect()
    }

    fn vec_sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<BasePartial<SolId, TwinOpt<Self>, SInfo>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| self.sample_opt(Some(rn), info.clone()))
            .collect()
    }

    fn sample_pair(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Pair<BasePartial<SolId, TwinOpt<Self>, SInfo>,SolId,Obj,TwinOpt<Self>,SInfo>>
    {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size).map(
            |_|
            {
                let s = self.sample_obj(Some(rn), info.clone()); // sample
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
        inp.iter().all(|sol| {self.is_in_opt::<S>(sol)})
    }
    
}


impl<Obj: Domain, Opt: Domain> From<SpPar<Obj, Opt>> for Sp<Obj, Opt> {
    fn from(value: SpPar<Obj, Opt>) -> Self {
        Sp {
            var: value.var,
        }
    }
}

impl<Obj, Opt> CSVLeftRight<Sp<Obj, Opt>, Arc<[TwinTyObj<Self>]>, Arc<[TwinTyOpt<Self>]>> for Sp<Obj, Opt>
where
    Obj: OntoDom<Opt> + CSVWritable<(), TwinTyObj<Self>>,
    Opt: OntoDom<Obj> + CSVWritable<(), TwinTyOpt<Self>>,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, TwinTyObj<Self>, TwinTyOpt<Self>>,
{
    fn header(elem: &Sp<Obj, Opt>) -> Vec<String> {
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

impl<Obj> CSVLeftRight<Sp<Obj, NoDomain>, Arc<[TwinTyObj<Self>]>, Arc<[TwinTyOpt<Self>]>> for Sp<Obj, NoDomain>
where
    Obj: Domain + CSVWritable<(), TwinTyObj<Self>>,
    Var<Obj, NoDomain>: CSVLeftRight<Var<Obj, NoDomain>, TwinTyObj<Self>, TwinTyOpt<Self>>,
{
    fn header(elem: &Sp<Obj, NoDomain>) -> Vec<String> {
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