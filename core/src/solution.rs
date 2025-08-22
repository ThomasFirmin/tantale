//! A [`Solution`] defines a point sampled within a [`Searchspace`] made of a [`Vec`] of [`Var`](tantale::core::Var).
//! Thus there are two types of solutions:
//!     * A [`Solution`] from the [`Objective`](tantale::core::objective::Objective) [`Domains`](tantale::core::domain::Domain).
//!     * A [`Solution`] from the [`Optimizer`](tantale::core::optimizer::Optimizer) [`Domains`](tantale::core::domain::Domain).
//!
//! A [`Solution`] is statically typed by a [`Domain`].
//! A [`Solution`] cannot typed by a `dyn Trait`. This is why the [`Mixed`](tantale::core::Mixed) trait is used
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
use crate::domain::{Domain, TypeDom};

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};
use serde::{Serialize,Deserialize};

/// Describes single-[`Solution`] information
/// obtained after each iteration of the [`Optimizer`].
pub trait SolInfo
where
    Self: Debug + Serialize + for<'a> Deserialize<'a>
{

}

/// An abstract [`Solution`] made of at least a [`Domain`] and a [`SolInfo`].
pub trait Solution<SolId, Dom, Info>
where
    Self: Sized,
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    SolId: Id + PartialEq,
{
    /// The `id` of a [`Solution`] is made of a `pid` and a unique number.
    /// This `id` can be shared with a twin [`Solution`].
    fn get_id(&self) -> SolId;

    /// Returns the actual sampled point from the set of [`Domains`](Domain).
    fn get_x(&self) -> Arc<[TypeDom<Dom>]>;

    /// Returns the [`SolInfo`] bounded to this [`Solution`].
    fn get_info(&self) -> Arc<Info>;

    /// Checks if two [`Solutions`](Solution) are twins.
    /// Twins [`Solutions`](Solution) share equal ids.
    fn is_twin<Twin, B>(&self, solb: Twin) -> bool
    where
        B: Domain + Clone + Display + Debug,
        Twin: Solution<SolId, B, Info>,
    {
        self.get_id() == solb.get_id()
    }
}

pub mod partial;
pub use partial::Partial;

pub mod computed;
pub use computed::Computed;

pub mod id;
pub use id::{Id, ParSId, SId};
