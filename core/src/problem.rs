use std::{fmt::Debug, marker::PhantomData};
use serde::{Deserialize, Serialize};

use crate::{BaseSol, FidelitySol, FuncWrapper, HasVariables, HasX, Linked, MultiStep, OneStep, Outcome, SId, Searchspace, SolId, SolInfo, SolutionShape, StepSId, Uncomputed, aliases::{GetFuncSol, LinkObj, LinkOpt}, solution::shape::IntoComputedShape};

pub trait PbMode: Serialize + for<'de> Deserialize<'de> {}

#[derive(Serialize, Deserialize, Debug)]
pub struct SimpleMode;
impl PbMode for SimpleMode {}

#[derive(Serialize, Deserialize, Debug)]
pub struct FidelityMode;
impl PbMode for FidelityMode {}

pub trait Problem: Sized
where
    GetFuncSol<Self::Func, LinkObj<Self::Scp>>: Uncomputed<Dom= LinkObj<Self::Scp>, X = Self::XObj, Id = Self::Id, Info = Self::Info>,
    GetFuncSol<Self::Func, LinkOpt<Self::Scp>>: Uncomputed<Dom= LinkOpt<Self::Scp>, X = Self::XOpt, Id = Self::Id, Info = Self::Info>,
{
    type Id: SolId;
    type Info: SolInfo;

    type XObj: Send + Sync + Clone + Debug + Serialize + for<'de> Deserialize<'de>;
    type XOpt: Send + Sync + Clone + Debug + Serialize + for<'de> Deserialize<'de>;

    type Scp: Searchspace<Self::Func,Shape = Self::Shape>;

    type Func: FuncWrapper<Dom = <Self::Scp as Linked>::Obj, Info = Self::Info, Id = Self::Id>;

    type Shape: SolutionShape<
        Id = Self::Id,
        Info = Self::Info,
        ObjSol = GetFuncSol<Self::Func, LinkObj<Self::Scp>>,
        OptSol = GetFuncSol<Self::Func, LinkOpt<Self::Scp>>,
        Obj = <Self::Scp as Linked>::Obj,
        Opt = <Self::Scp as Linked>::Opt,
    > + IntoComputedShape;

    fn get_searchspace(&self) -> &Self::Scp;
    fn get_mut_searchspace(&mut self) -> &mut Self::Scp;
    fn get_objective(&self) -> &Self::Func;
    fn get_mut_objective(&mut self) -> &mut Self::Func;
    fn get_codomain(&self) -> &<<Self::Func as FuncWrapper>::Out as Outcome>::Cod;
    fn get_mut_codomain(&mut self) -> &mut <<Self::Func as FuncWrapper>::Out as Outcome>::Cod;
}

pub struct Pb<Id, Info, Scp, Func>
where
    Id: SolId,
    Info: SolInfo,
    Scp: Searchspace<Func>,
    Func: FuncWrapper<Dom = LinkObj<Scp>, Id =  Id, Info = Info>,
{
    searchspace: Scp,
    objective: Func,
    codomain: <Func::Out as Outcome>::Cod,
    _phantom: PhantomData<(Id, Info)>,
}

impl <Id, Info, Scp, Func> Pb<Id, Info, Scp, Func>
where
    Id: SolId,
    Info: SolInfo,
    Scp: Searchspace<Func>,
    Func: FuncWrapper<Dom = LinkObj<Scp>, Id =  Id, Info = Info>,
{
    pub fn new(searchspace: Scp, objective: Func) -> Self {
        Self {
            searchspace,
            objective,
            codomain: <Func::Out as Outcome>::codomain(),
            _phantom: PhantomData,
        }
    }
}

impl<Id, Info, Scp, Func> Problem for Pb<Id, Info, Scp, Func>
where
    Id: SolId,
    Info: SolInfo,
    Scp: Searchspace<Func> + HasVariables,
    Scp::Shape: SolutionShape<
        Id = Id,
        Info = Info,
        ObjSol = <Func as FuncWrapper>::Sol<LinkObj<Scp>>,
        OptSol = <Func as FuncWrapper>::Sol<LinkOpt<Scp>>,
        Obj = LinkObj<Scp>,
        Opt = LinkOpt<Scp>,
    > + IntoComputedShape,
    Func: FuncWrapper<Dom = LinkObj<Scp>, Id = Id, Info = Info>,
{
    type Id = Id;

    type Info = Info;

    type XObj = <<Self::Func as FuncWrapper>::Sol<<Self::Scp as Linked>::Obj> as HasX>::X;
    type XOpt = <<Self::Func as FuncWrapper>::Sol<<Self::Scp as Linked>::Opt> as HasX>::X;

    type Scp = Scp;

    type Func = Func;

    type Shape = Scp::Shape;
    
    fn get_searchspace(&self) -> &Self::Scp {
        &self.searchspace
    }
    
    fn get_mut_searchspace(&mut self) -> &mut Self::Scp {
        &mut self.searchspace
    }
    
    fn get_objective(&self) -> &Self::Func {
        &self.objective
    }
    
    fn get_mut_objective(&mut self) -> &mut Self::Func {
        &mut self.objective
    }
    
    fn get_codomain(&self) -> &<<Self::Func as FuncWrapper>::Out as Outcome>::Cod {
        &self.codomain
    }
    
    fn get_mut_codomain(&mut self) -> &mut <<Self::Func as FuncWrapper>::Out as Outcome>::Cod {
        &mut self.codomain
    }
}

pub type SimplePb<Info, Scp, Out> = Pb<SId, Info, Scp, SimpleFn<Info, LinkObj<Scp>, Out>>;
pub type SimpleFn<Info, Obj, Out> = OneStep<BaseSol<SId, Obj, Info>, Out>;
pub type FidPb<Info, Scp, Out, FnState> = Pb<StepSId, Info, Scp, FidFn<Info, LinkObj<Scp>, Out, FnState>>;
pub type FidFn<Info, Obj, Out, FnState> = MultiStep<FidelitySol<StepSId, Obj, Info>, Out, FnState>;