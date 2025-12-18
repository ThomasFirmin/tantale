//! A [`Solution`] defines a point sampled within a [`Searchspace`] made of a [`Vec`] of [`Var`](tantale::core::Var).
//! Thus there are two types of solutions:
//!     * A [`Solution`] from the [`Objective`](tantale::core::objective::Objective) [`Domains`](tantale::core::domain::Domain).
//!     * A [`Solution`] from the [`Optimizer`](tantale::core::optimizer::Optimizer) [`Domains`](tantale::core::domain::Domain).
//!
//! A [`Solution`] is statically typed by a [`Domain`].
//! A [`Solution`] cannot be typed by a `dyn Trait`. This is why the [`Mixed`](tantale::core::Mixed) trait is used
//! within the [`sp!`](../../../tantale/macros/macro.sp.html) macro, to create `enum` of used [`Domains`](tantale::core::Domain) and
//! and `enum` of [`TypeDom`](tantale::core::Domain::TypeDom).
//!
//! A [`Solution`] is made of a unique `id`, an [`Arc`]`<[`[`TypeDom`](tantale::core::Domain::TypeDom)`]>` and a [`SolInfo`](tantale::core::SolInfo).
//! [`Solutions`](Solution) are strongly bounded to [`Searchspace`](tantale::core::Searchspace), from which
//! most of them will be created via simplified methods.
//! The types of [`Solution`], are then used to statically constrain
//! the [`Objective`](tantale::core::Objective) and [`Optimizer`](tantale::core::Optimizer) inputs.
//!
//! The unique `id` of a solution is computed according to a static atomic variable `SOL_ID`, and the `pid` of the process creating that solution.
//!
//! # Notes
//!
//! A [`Solution`] with an unknown [`Outcome`](tantale::core::Outcome) is [`Partial`](tantale::core::Partial).
//! When the [`Outcome`](tantale::core::Outcome) is known, a [`Partial`](tantale::core::Partial)
//! becomes a [`Computed`](tantale::core::Computed) [`Solution`], made of a [`Partial`](tantale::core::Partial),
//! and a [`Codomain`](tantale::core::Codomain).
//!
use crate::{Codomain, EvalStep, OptInfo, Outcome, domain::Domain, objective::Step};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

/// Describes single-[`Solution`] information
/// obtained after each iteration of the [`Optimizer`].
pub trait SolInfo
where
    Self: Debug + Serialize + for<'a> Deserialize<'a>,
{
}

/// Describes an object associated to an [`Id`].
pub trait HasId<SolId:Id> {
    /// Returns the current [`Id`].
    /// 
    /// # Note
    /// The [`Id`] of a [`Solution`] is a unique object identifying a [`Solution`]. See [`here`](Id).
    /// The [`Id`] of [`Solution`] is only shared with its [`Twin`](Solution::Twin).
    fn get_id(&self) -> SolId;

    /// Checks if two object containing an [`Id`] are twins (same [`Id`]).
    /// 
    /// # Note
    /// Twins [`Solution`] share equal ids.
    fn is_twin<Twin:HasId<SolId>>(&self, solb: Twin) -> bool
    {
        self.get_id() == solb.get_id()
    }
}

/// Describes an object associated to a [`SolInfo`].
pub trait HasSolInfo<Info:SolInfo> {
    /// Returns the current [`SolInfo`].
    fn get_sinfo(&self) -> Arc<Info>;

}

/// Describes an object associated to a [`TypeCodom`](Codomain::TypeCodom).
pub trait HasY<Cod:Codomain<Out>,Out:Outcome> {
    /// Returns the current [`OptInfo`].
    fn get_y(&self) -> Arc<Cod::TypeCodom>;
}

/// Describes an object associated to an [`OptInfo`].
pub trait HasInfo<Info:OptInfo> {
    /// Returns the current [`OptInfo`].
    fn get_info(&self) -> Arc<Info>;
}


/// Describes an object associated to an [`Step`].
pub trait HasStep {
    /// Returns the current [`Step`].
    fn step(&self) -> Step;
    /// Returns the current [`EvalStep`].
    fn raw_step(&self) -> EvalStep;
    /// Set the current [`EvalStep`].
    fn set_step(&mut self,value:EvalStep);
    /// Set the current [`Step`] to [`Pending`](Step::Pending).
    fn pending(&mut self);
    /// Set the current [`Step`] to [`Partially`](Step::Partially).
    fn partially(&mut self, value:isize);
    /// Set the current [`Step`] to [`Discard`](Step::Discard).
    fn discard(&mut self);
    /// Set the current [`Step`] to [`Evaluated`](Step::Evaluated).
    fn evaluated(&mut self);
    /// Set the current [`Step`] to [`Error`](Step::Error).
    fn error(&mut self);
}

/// Describes an object associated to a [`Fidelity`].
pub trait HasFidelity {
    /// Returns the current [`Fidelity`].
    fn fidelity(&self) ->  Fidelity;
    /// Set the [`Fidelity`] of the object.
    fn set_fidelity(&mut self, fidelity: Fidelity);
}

/// An abstract [`Solution`].
pub trait Solution<SolId, Dom, SInfo>: HasId<SolId> + HasSolInfo<SInfo>
where
    Self: Sized + Debug + Serialize + for<'de> Deserialize<'de>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
{
    type Raw: Debug + Serialize + for<'de> Deserialize<'de>;
    type Twin<B: Domain>: Solution<SolId, B, SInfo, Twin<Dom> = Self>;

    fn twin<B:Domain>(&self, x: <Self::Twin<B> as Solution<SolId,B,SInfo>>::Raw) -> Self::Twin<B>;
    /// Returns the actual sampled point from the set of [`Domain`].
    fn get_x(&self) -> Self::Raw;
}

/// Describes an object containing non-[`Computed`] object.
pub trait HasUncomputed<SolId:Id,Dom:Domain,SInfo:SolInfo>
{
    type Uncomputed: Uncomputed<SolId,Dom,SInfo>;
    fn get_uncomputed(&self) -> &Self::Uncomputed;
}

/// Describes a non-[`Computed`] [`Solution`].
pub trait Uncomputed<SolId,Dom,SInfo>:Solution<SolId,Dom,SInfo> + IntoComputed
where
    SolId : Id,
    Dom : Domain,
    SInfo : SolInfo,
{
    type TwinUC<B:Domain>: Uncomputed<SolId,B,SInfo, Twin<Dom> = Self, TwinUC<Dom> = Self>;
    fn twin<B:Domain>(&self, x: <Self::TwinUC<B> as Solution<SolId,B,SInfo>>::Raw) -> Self::TwinUC<B>;
    fn new<T>(id: SolId, x: T, info: Arc<SInfo>) -> Self
        where
            T: Into<Self::Raw>;
}

/// A trait allowing to convert an object into an object that [`HasY`] with a [`TypeCodom`](Codomain::TypeCodom).
/// A concrete example is the [`Computed`] structure.
pub trait IntoComputed
{
    type Computed<Cod:Codomain<Out>,Out:Outcome>: HasY<Cod,Out>;
    fn into_computed<Cod:Codomain<Out>,Out:Outcome>(self, y: Arc<Cod::TypeCodom>) -> Self::Computed<Cod,Out>;
}


/// The [`Twin`](Solution::Twin) [`Solution`] of type `B` from a [`Solution`] of type `A`.
pub type SolTwin<S, SolId, A, B, SInfo> = <S as Solution<SolId,A,SInfo>>::Twin<B>;
/// The [`Raw`](Solution::Raw) a [`Solution`].
pub type SolRaw<S, SolId, Dom, Info> = <S as Solution<SolId,Dom,Info>>::Raw;

pub mod id;
pub use id::{Id, ParSId, SId};

pub mod partial;
pub use partial::{BasePartial, FidBasePartial, Fidelity};

pub mod computed;
pub use computed::Computed;

pub mod batchtype;
pub use batchtype::{Batch, OutBatch};

pub mod shape;
pub use shape::{SolutionShape,CompLone,Pair,Lone};