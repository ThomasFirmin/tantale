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
    fn get_id(&self)->(usize, u32);
    
    fn get_x(&self)->Arc<[TypeDom<Dom>]>;
    
    fn get_info(&self)->Arc<Info>;
    
    fn _get_new_id()->usize{
        let idsol = SOL_ID.fetch_add(1, Ordering::Relaxed);
        idsol
    }

    /// Creates a new [`Solution`] from a given `pid` of the process it is being created,
    /// a single `id`, a concrete sample `x` from the set of [`Domains`](Domain), and a [`SolInfo`].
    fn build(pid:u32, id:usize, x : Arc<[TypeDom<Dom>]>, info : Arc<Info>) -> Self;

    /// Creates a new [`Solution`] from a given `pid` of the process it is being created,
    /// a concrete sample `x` from the set of [`Domains`](Domain), and a [`SolInfo`].
    fn new(pid:u32, x : Arc<[TypeDom<Dom>]>, info : Arc<Info>) -> Self
    {
        let id = Self::_get_new_id();
        Self::build(pid, id, x, info)
    }
    
    /// Creates a twin [`Solution`] of [`Domain`] type `B` from
    /// the [`Self`] [`Solution`] of [`Domain`] type `Dom`.
    /// The twins are linked by the same `id`.
    fn twin<Twin,B,T>(&self, x: T) -> Twin
    where 
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        Twin:Solution<B,Info>,
        T : AsRef<[TypeDom<B>]>;

    /// Checks if two [`Solutions`](Solution) are twins.
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