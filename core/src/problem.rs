use std::marker::PhantomData;

use crate::{BaseSol, Domain, FidOutcome, FidelitySol, FuncState, FuncWrapper, HasFidelity, HasStep, HasVariables, Linked, MultiStep, 
    OneStep, Outcome, SId, Searchspace, SolId, SolInfo, SolutionShape, StepId, StepSId, 
    aliases::{LinkObj, LinkOpt}, solution::shape::IntoComputedShape};

pub trait Problem {
    type Id: SolId;
    type Info: SolInfo;

    type Scp: Searchspace<
        Self::Func,
        Shape = Self::Shape,
    >;

    type Func: FuncWrapper<
        Dom = <Self::Scp as Linked>::Obj,
    >;

    type Shape: SolutionShape<
        Id = Self::Id,
        Info = Self::Info,
        ObjSol = <Self::Func as FuncWrapper>::Sol<
            <Self::Scp as Linked>::Obj
        >,
        OptSol = <Self::Func as FuncWrapper>::Sol<
            <Self::Scp as Linked>::Opt
        >,
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

pub trait SimpleProblem<Opt: Domain>: Problem 
where
    Self::Scp: Searchspace<Self::Func,Shape = Self::Shape,Opt = Opt>
{ }

impl<Opt, T, Id, Scp, Info, Out> SimpleProblem<Opt> for T
where
    Opt: Domain,
    Id: SolId,
    Scp: Searchspace<OneStep<BaseSol<Id, LinkObj<Scp>, Info>, Out>, Opt = Opt>,
    Info: SolInfo,
    Out: Outcome,
    T: Problem<Id = Id, Scp = Scp, Info = Info, Func = OneStep<BaseSol<Id, LinkObj<Scp>, Info>, Out>>,
    T::Scp: Searchspace<T::Func, Shape = T::Shape, Opt = Opt>,
{}

pub trait FidelityProblem<Opt: Domain>: Problem 
where
    Self::Scp: Searchspace<Self::Func,Shape = Self::Shape,Opt = Opt>
{ }

impl<Opt, T, Id, Scp, Info, Out, FnState> FidelityProblem<Opt> for T
where
    Opt: Domain,
    Id: StepId,
    Scp: Searchspace<MultiStep<FidelitySol<Id, LinkObj<Scp>, Info>, Out, FnState>, Opt = Opt>,
    Info: SolInfo,
    Out: FidOutcome,
    FnState: FuncState,
    T: Problem<Id = Id, Scp = Scp, Info = Info, Func = MultiStep<FidelitySol<Id, LinkObj<Scp>, Info>, Out, FnState>>,
    T::Scp: Searchspace<T::Func, Shape = T::Shape, Opt = Opt>,
    T::Shape: HasFidelity + HasStep,
{ }


pub struct Pb<Id, Info, Scp, Func>
where
    Id: SolId,
    Info: SolInfo,
    Scp: Searchspace<Func>,
    Func: FuncWrapper<Dom = LinkObj<Scp>>
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
    Func: FuncWrapper<Dom = LinkObj<Scp>>,
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
    Func: FuncWrapper<Dom = LinkObj<Scp>>,
{
    type Id = Id;

    type Info = Info;

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

pub type SimplePb<Info, Scp, Out> = Pb<SId, Info, Scp, OneStep<BaseSol<SId, LinkObj<Scp>, Info>, Out>>;
pub type FidPb<Info, Scp, Out, FnState> = Pb<StepSId, Info, Scp, MultiStep<FidelitySol<StepSId, LinkObj<Scp>, Info>, Out, FnState>>;