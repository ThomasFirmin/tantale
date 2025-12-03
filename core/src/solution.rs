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
use crate::{domain::{Domain, onto::{Paired, TwinDom}}, objective::Step, solution::partial::FidelityPartial};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

/// Describes single-[`Solution`] information
/// obtained after each iteration of the [`Optimizer`].
pub trait SolInfo
where
    Self: Debug + Serialize + for<'a> Deserialize<'a>,
{
}

/// An abstract [`Solution`] made of at least a [`Domain`] and a [`SolInfo`].
pub trait Solution<SolId, Dom, Info>
where
    Self: Sized + Debug + Serialize + for<'de> Deserialize<'de>,
    Dom: Domain,
    Info: SolInfo,
    SolId: Id + PartialEq,
{
    type Raw: Debug + Serialize;
    type Twin<B: Domain>: Solution<SolId, B, Info, Twin<Dom> = Self>;

    /// The `id` of a [`Solution`] is made of a `pid` and a unique number.
    /// This `id` can be shared with a twin [`Solution`].
    fn get_id(&self) -> SolId;

    /// Returns the actual sampled point from the set of [`Domain`].
    fn get_x(&self) -> Self::Raw;

    /// Returns the [`SolInfo`] bounded to this [`Solution`].
    fn get_info(&self) -> Arc<Info>;

    /// Checks if two [`Solution`] are twins.
    /// Twins [`Solution`] share equal ids.
    fn is_twin<Twin, B>(&self, solb: Twin) -> bool
    where
        B: Domain,
        Twin: Solution<SolId, B, Info>,
    {
        self.get_id() == solb.get_id()
    }
}

/// A pair made of a `Obj` [`Partial`] and its `Opt` [`Twin`](Partial::Twin).
#[derive(Debug,Serialize,Deserialize)]
#[serde(bound(
    serialize = "P: Serialize, P::Twin<Opt>: Serialize",
    deserialize = "P: for<'a> Deserialize<'a>, P::Twin<Opt>: for<'a> Deserialize<'a>",
))]
pub struct Pair<P,SolId,Obj,Opt,SInfo>(P::Twin<Obj>,P)
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    P: Solution<SolId,Opt,SInfo>;

pub type CompPair<P,SolId,Obj,Opt,SInfo, Cod, Out> = Pair<Computed<P,SolId,Opt,Cod,Out,SInfo>,SolId,Obj,Opt,SInfo>;

impl<P,SolId,Obj,Opt,SInfo> Pair<P,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    P: Solution<SolId,Opt,SInfo>,
    P::Twin<Obj>: Solution<SolId,Opt,SInfo, Twin<Obj>=P>,
{
    pub fn new(solobj:P::Twin<Obj>,solopt:P)->Self{Self(solobj,solopt)}
    pub fn get_id(&self)->SolId{self.0.get_id()}
    pub fn get_info(&self)->Arc<SInfo>{self.0.get_info()}
}

impl<P,SolId,Obj,Opt,SInfo> Pair<P,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    P: Solution<SolId,Opt,SInfo> + FidelityPartial<SolId,Opt,SInfo>,
    <P as FidelityPartial<SolId,Opt,SInfo>>::Twin<Obj>:  Solution<SolId,Opt,SInfo, Twin<Obj>=P> + FidelityPartial<SolId,Obj,SInfo, Twin<Opt>=P>,
{
    pub fn get_fidelity(&self)->Option<Fidelity>{
        <P as FidelityPartial<SolId,Opt,SInfo>>::get_fidelity(&self.1)
    }
    pub fn get_step(&self)->Step{<P as FidelityPartial<SolId,Opt,SInfo>>::get_step(&self.1)}
}

impl<P,SolId,Obj,Opt,SInfo> From<(P::Twin<Obj>,P)> for Pair<P,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    P: Solution<SolId,Opt,SInfo>,
    P::Twin<Obj>: Solution<SolId,Opt,SInfo, Twin<Obj>=P>,
{
    fn from(value: (P::Twin<Obj>,P)) -> Self {
        Self(value.0,value.1)
    }
}

pub mod id;
pub use id::{Id, ParSId, SId};

pub mod partial;
pub use partial::{BasePartial, FidBasePartial, Fidelity, Partial};

pub mod computed;
pub use computed::Computed;

pub mod outsol;
pub use outsol::OutSol;

pub mod batchtype;
pub use batchtype::{Batch, CompBatch, OutBatch};
