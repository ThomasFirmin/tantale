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
use crate::optimizer::SolInfo;

use std::{
    fmt::{Debug, Display},
    sync::{Arc, atomic::{AtomicUsize,Ordering}},
};

static SOL_ID: AtomicUsize = AtomicUsize::new(0);

/// An abstract [`Solution`] made of at least a [`Domain`] and a [`SolInfo`].
pub trait Solution<Dom,Info>
where
    Self: Sized,
    Dom: Domain + Clone + Display + Debug,
    Info : SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    /// The `id` of a [`Solution`] is made of a `pid` and a unique number.
    /// This `id` can be shared with a twin [`Solution`].
    fn get_id(&self)->(u32, usize);
    
    /// Returns the actual sampled point from the set of [`Domains`](Domain).
    fn get_x(&self)->Arc<[TypeDom<Dom>]>;
    
    /// Returns the [`SolInfo`] bounded to this [`Solution`].
    fn get_info(&self)->Arc<Info>;
    
    #[doc(hidden)]
    fn _get_new_id()->usize{
        let idsol = SOL_ID.fetch_add(1, Ordering::Relaxed);
        idsol
    }
    /// Checks if two [`Solutions`](Solution) are twins.
    /// Twins [`Solutions`](Solution) share equal ids.
    fn is_twin<Twin,B>(&self, solb: Twin) -> bool
    where 
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        Twin:Solution<B,Info>
    {
        let ida = self.get_id();
        let idb = solb.get_id();
        (ida.0 == idb.0) && (ida.1 == idb.1)
    }
}