use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

use crate::{
    BaseSol, FidelitySol, FuncWrapper, HasVariables, HasX, Linked, MultiStep, OneStep, Outcome,
    SId, Searchspace, SolId, SolInfo, SolutionShape, StepSId, Uncomputed,
    aliases::{GetFuncSol, LinkObj, LinkOpt},
    solution::shape::IntoComputedShape,
};

/// Marker trait for a problem that can be optimized by an [`Sampler`](crate::Sampler).
/// A [`Problem`] is a combination of a [searchspace](crate::Searchspace) and an [objective](crate::objective::FuncWrapper) function, 
/// along with associated types for solution identifiers,
/// solution information, and solution shapes. It defines the interface for accessing the searchspace, objective function, and codomain of the problem.
/// Formally, a [`Problem`] defines:
/// $$ x \rightarrow f(x) $$ with:
/// - $x$ the input of the objective function, defined by the [`Searchspace`] ($x \in \mathcal{X}$). This $x$ is an [`Uncomputed`]
///   solution with a given [`SolId`] and [`SolInfo`], made of raw [`Element`](crate::Domain::Element) from the [`Domain`](crate::Domain) of the searchspace.
/// - $f$ the objective function, defined by the [`FuncWrapper`] ($f: \mathcal{X} \rightarrow \mathcal{Y}$)
/// - $\mathcal{Y}$ the [`Codomain`](crate::Codomain) of the objective function, defined by the [`Outcome`] ($f(x) \in \mathcal{Y}$).
///   Hence, [`TypeCodom`](crate::Codomain::TypeCodom) is $f(x)$ or $y$.
/// 
/// The [`Problem`] trait is implemented by the [`Pb`] struct, which provides a concrete implementation of a problem with a searchspace and an objective function.
/// 
/// # Type aliases
/// 
/// - [`SimplePb`] is a [`Problem`] with a single-step objective function.
/// - [`FidPb`] is a [`Problem`] with a multi-step objective function that supports fidelity evaluations.
pub trait Problem: Sized + Send + Sync
where
    GetFuncSol<Self::Func, LinkObj<Self::Scp>>:
        Uncomputed<Dom = LinkObj<Self::Scp>, X = Self::XObj, Id = Self::Id, Info = Self::Info>,
    GetFuncSol<Self::Func, LinkOpt<Self::Scp>>:
        Uncomputed<Dom = LinkOpt<Self::Scp>, X = Self::XOpt, Id = Self::Id, Info = Self::Info>,
{
    /// Associated type for solution identifiers, which must implement [`SolId`].
    type Id: SolId;
    /// Associated type for solution information, which must implement [`SolInfo`].
    type Info: SolInfo;
    /// Associated type for the [raw](crate::HasX::X) input type of the objective function for the object solution.
    type XObj: Send + Sync + Clone + Debug + Serialize + for<'de> Deserialize<'de>;
    /// Associated type for the [raw](crate::HasX::X) input type of the objective function for the optimization solution.
    type XOpt: Send + Sync + Clone + Debug + Serialize + for<'de> Deserialize<'de>;
    /// Associated type for the searchspace of the problem, which must implement [`Searchspace`].
    type Scp: Searchspace<Self::Func, Shape = Self::Shape>;
    /// Associated type for the objective function of the problem, which must implement [`FuncWrapper`].
    type Func: FuncWrapper<Dom = <Self::Scp as Linked>::Obj, Info = Self::Info, Id = Self::Id>;
    /// Associated type for the solution shape of the problem, which must implement [`SolutionShape`] and [`IntoComputedShape`].
    type Shape: SolutionShape<
            Id = Self::Id,
            Info = Self::Info,
            ObjSol = GetFuncSol<Self::Func, LinkObj<Self::Scp>>,
            OptSol = GetFuncSol<Self::Func, LinkOpt<Self::Scp>>,
            Obj = <Self::Scp as Linked>::Obj,
            Opt = <Self::Scp as Linked>::Opt,
        > + IntoComputedShape;

    /// Returns a reference to the searchspace of the problem.
    fn get_searchspace(&self) -> &Self::Scp;
    /// Returns a mutable reference to the searchspace of the problem.
    fn get_mut_searchspace(&mut self) -> &mut Self::Scp;
    /// Returns a reference to the objective function of the problem.
    fn get_objective(&self) -> &Self::Func;
    /// Creates a clone of the objective function of the problem.
    fn clone_objective(&self) -> Self::Func {
        self.get_objective().clone()
    }
    /// Returns a mutable reference to the objective function of the problem.
    fn get_mut_objective(&mut self) -> &mut Self::Func;
    /// Returns a reference to the codomain of the objective function of the problem.
    fn get_codomain(&self) -> &<<Self::Func as FuncWrapper>::Out as Outcome>::Cod;
    /// Returns a mutable reference to the codomain of the objective function of the problem.
    fn get_mut_codomain(&mut self) -> &mut <<Self::Func as FuncWrapper>::Out as Outcome>::Cod;
}

/// A [`Problem`] implementation that combines a [`Searchspace`] and an [`objective`](crate::objective::FuncWrapper) function.
pub struct Pb<Id, Info, Scp, Func>
where
    Id: SolId,
    Info: SolInfo,
    Scp: Searchspace<Func>,
    Func: FuncWrapper<Dom = LinkObj<Scp>, Id = Id, Info = Info>,
{
    searchspace: Scp,
    objective: Func,
    codomain: <Func::Out as Outcome>::Cod,
    _phantom: PhantomData<(Id, Info)>,
}

impl<Id, Info, Scp, Func> Pb<Id, Info, Scp, Func>
where
    Id: SolId,
    Info: SolInfo,
    Scp: Searchspace<Func>,
    Func: FuncWrapper<Dom = LinkObj<Scp>, Id = Id, Info = Info>,
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

/// Type aliases for common problem types with a [`SimpleFn`].
pub type SimplePb<Info, Scp, Out> = Pb<SId, Info, Scp, SimpleFn<Info, LinkObj<Scp>, Out>>;
/// Type aliases for a basic [`OneStep`] made of [`BaseSol`].
pub type SimpleFn<Info, Obj, Out> = OneStep<BaseSol<SId, Obj, Info>, Out>;
/// Type aliases for common problem types with a [`FidFn`].
pub type FidPb<Info, Scp, Out, FnState> =
    Pb<StepSId, Info, Scp, FidFn<Info, LinkObj<Scp>, Out, FnState>>;
/// Type aliases for a basic [`MultiStep`] made of [`FidelitySol`].    
pub type FidFn<Info, Obj, Out, FnState> = MultiStep<FidelitySol<StepSId, Obj, Info>, Out, FnState>;
