use crate::{
    domain::{
        Domain, NoDomain, PreDomain, onto::{LinkTyObj, LinkTyOpt, Linked, OntoDom}
    }, recorder::csv::{CSVLeftRight, CSVWritable},
    searchspace::{Searchspace, SolInfo, SpPar},
    solution::{Id, IntoComputed, Lone, Pair, Solution, Uncomputed, shape::{RawObj, RawOpt}},
    variable::Var
};

use rand::prelude::ThreadRng;
use std::sync::Arc;

/// A basic [`Searchspace`] made of a [`Box`] slice of [`Variable`].
pub struct Sp<Obj: Domain, Opt: PreDomain>
{
    pub var: Box<[Var<Obj,Opt>]>,
}

impl<Obj:OntoDom<Opt>, Opt:OntoDom<Obj>> Linked for Sp<Obj,Opt> 
{
    type Obj = Obj;
    type Opt = Opt;
}

impl<Obj:Domain> Linked for Sp<Obj,NoDomain> 
{
    type Obj = Obj;
    type Opt = Obj;
}

impl<SolOpt,SolId, Obj, Opt, SInfo> Searchspace<SolOpt,SolId, SInfo> for Sp<Obj, Opt>
where
    SolId: Id,
    SInfo: SolInfo,
    Obj: OntoDom<Opt>,
    Opt: OntoDom<Obj>,
    SolOpt: Uncomputed<SolId,Opt,SInfo, Raw=Arc<[Opt::TypeDom]>> + IntoComputed,
    SolOpt::Twin<Obj>: Uncomputed<SolId,Obj,SInfo, Twin<Opt>=SolOpt, Raw=Arc<[Obj::TypeDom]>> + IntoComputed,
{
    type SolShape = Pair<SolOpt::Twin<Obj>,SolOpt,SolId,Self::Obj,Self::Opt,SInfo>;

    fn onto_obj(&self, inp: SolOpt) -> Self::SolShape
    {
        let outx:Vec<LinkTyObj<Self>> = self.var.iter().zip(inp.get_x().iter()).map(
            |(v,xi)|
            {
                v.onto_obj(xi).unwrap()
            }
        ).collect();
        let solobj = Solution::twin::<Obj>(&inp, outx.into());
        Pair::new(solobj, inp)
    }

    fn onto_opt(&self,inp: SolOpt::Twin<Obj>) -> Self::SolShape
    {
        let outx:Vec<LinkTyOpt<Self>> = self.var.iter().zip(inp.get_x().iter()).map(
            |(v,xi)|
            {
                v.onto_opt(xi).unwrap()
            }
        ).collect();
        let solopt = Solution::twin::<Opt>(&inp, outx.into());
        Pair::new(inp, solopt)
    }

    fn sample_obj(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> SolOpt::Twin<Obj>{
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_obj(rn)).collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }
    
    fn sample_opt(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> SolOpt {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_opt(rn)).collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }
    
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = <SolOpt::Twin<Obj> as Solution<SolId,Self::Obj,SInfo>>::Raw> + Send + Sync
    {
        inp.get_x().iter().zip(&self.var).all(|(elem, v)| v.is_in_obj(elem))
    }
    
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw =<SolOpt as Solution<SolId,Opt,SInfo>>::Raw> + Send + Sync
    {
        inp.get_x().iter().zip(&self.var).all(|(elem, v)| v.is_in_opt(elem))
    }
    
    fn vec_onto_obj(&self, inp: Vec<SolOpt>) -> Vec<Self::SolShape> {
        inp.into_iter().map(|i| self.onto_obj(i)).collect()
    }
    
    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_iter().map(|i| self.onto_opt(i)).collect()
    }
    
    fn vec_sample_obj(&self,rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<SolOpt::Twin<Obj>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, Some(rn), info.clone()))
            .collect()
    }
    
    fn vec_sample_opt(&self,rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<SolOpt> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| self.sample_opt(Some(rn), info.clone()))
            .collect()
    }
    
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = RawObj<Self::SolShape,SolId,SInfo>> + Send + Sync 
    {
        inp.iter().all(|sol| {
            <Sp<Obj, Opt> as Searchspace<SolOpt, SolId, SInfo>>::is_in_obj::<S>(self, sol)
        })
    }
    
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw =RawOpt<Self::SolShape,SolId,SInfo>> + Send + Sync 
    {
        inp.iter().all(|sol| {
            <Sp<Obj, Opt> as Searchspace<SolOpt, SolId, SInfo>>::is_in_opt::<S>(self, sol)
        })
    }

    fn sample_pair(&self,rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<Self::SolShape>
    {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size).map(
            |_|
            {
                let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, Some(rn), info.clone()); // sample
                self.onto_opt(s)
            }
        ).collect()
    }
}

impl<SolOpt,SolId, Obj, SInfo> Searchspace<SolOpt,SolId, SInfo> for Sp<Obj, NoDomain>
where
    SolId: Id,
    SInfo: SolInfo,
    Obj: Domain,
    SolOpt: Solution<SolId,Obj,SInfo, Raw=Arc<[Obj::TypeDom]>, Twin<Obj> = SolOpt> + Uncomputed<SolId,Obj,SInfo> + IntoComputed,
{
    type SolShape = Lone<SolOpt,SolId,Obj,SInfo>;

    fn onto_obj(&self, inp: SolOpt) -> Self::SolShape
    {
        Lone::new(inp)
    }

    fn onto_opt(&self,inp: SolOpt::Twin<Obj>) -> Self::SolShape
    {
        Lone::new(inp)
    }

    fn sample_obj(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_obj(rn)).collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }

    fn sample_opt(&self, rng: Option<&mut ThreadRng>, info: Arc<SInfo>) -> SolOpt::Twin<Obj> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        let outx: Vec<_> = self.var.iter().map(|v| v.sample_opt(rn)).collect();
        Uncomputed::new(SolId::generate(), outx, info)
    }
    
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = <SolOpt::Twin<Obj> as Solution<SolId,Self::Obj,SInfo>>::Raw> + Send + Sync
    {
        inp.get_x().iter().zip(&self.var).all(|(elem, v)| v.is_in_obj(elem))
    }
    
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw =<SolOpt::Twin<Obj> as Solution<SolId,Self::Opt,SInfo>>::Raw>
    {
        inp.get_x().iter().zip(&self.var).all(|(elem, v)| v.is_in_opt(elem))
    }

    fn vec_onto_obj(&self, inp: Vec<SolOpt>) -> Vec<Self::SolShape> {
        inp.into_iter().map(|sol| Lone::new(sol)).collect()
    }
    
    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Obj>>) -> Vec<Self::SolShape> {
        inp.into_iter().map(|sol| Lone::new(sol)).collect()
    }

    fn vec_sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, Some(rn), info.clone()))
            .collect()
    }

    fn vec_sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Obj>> {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size)
            .map(|_| self.sample_opt(Some(rn), info.clone()))
            .collect()
    }

    fn sample_pair(&self,rng: Option<&mut ThreadRng>,size: usize,info: Arc<SInfo>) -> Vec<Self::SolShape>
    {
        let rn = match rng {
            Some(r) => r,
            None => &mut rand::rng(),
        };
        (0..size).map(
            |_|
            {
                let s = <Self as Searchspace<SolOpt, SolId, SInfo>>::sample_obj(self, Some(rn), info.clone()); // sample
                Lone::new(s)
            }
        ).collect()
    }

    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw =  Arc<[Obj::TypeDom]>> + Send + Sync
    {
        inp.iter().all(|sol| {<Sp<Obj, NoDomain> as Searchspace<SolOpt::Twin<Obj>, SolId, SInfo>>::is_in_obj::<S>(self, sol)})
    }
    
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = Arc<[Obj::TypeDom]>> + Send + Sync
    {
        inp.iter().all(|sol| {<Sp<Obj, NoDomain> as Searchspace<SolOpt::Twin<Obj>, SolId, SInfo>>::is_in_obj::<S>(self, sol)})
    }
}


impl<Obj: Domain, Opt: Domain> From<SpPar<Obj, Opt>> for Sp<Obj, Opt> {
    fn from(value: SpPar<Obj, Opt>) -> Self {
        Sp {
            var: value.var,
        }
    }
}

impl<Obj, Opt> CSVLeftRight<Sp<Obj, Opt>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>> for Sp<Obj, Opt>
where
    Obj: OntoDom<Opt> + CSVWritable<(), LinkTyObj<Self>>,
    Opt: OntoDom<Obj> + CSVWritable<(), LinkTyOpt<Self>>,
    Var<Obj, Opt>: CSVLeftRight<Var<Obj, Opt>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    fn header(elem: &Sp<Obj, Opt>) -> Vec<String> {
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

impl<Obj> CSVLeftRight<Sp<Obj, NoDomain>, Arc<[LinkTyObj<Self>]>, Arc<[LinkTyOpt<Self>]>> for Sp<Obj, NoDomain>
where
    Obj: Domain + CSVWritable<(), LinkTyObj<Self>>,
    Var<Obj, NoDomain>: CSVLeftRight<Var<Obj, NoDomain>, LinkTyObj<Self>, LinkTyOpt<Self>>,
{
    fn header(elem: &Sp<Obj, NoDomain>) -> Vec<String> {
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