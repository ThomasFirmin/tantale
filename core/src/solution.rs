//! A [`Solution`] is defines a point sampled within a [`Searchspace`] made of a [`Vec`] of [`Var`](tantale::core::Var).
//! Thus there are two types of solutions:
//!     * A [`Solution`] from the set made of [`Objective`](tantale::core::objective::Objective) [`Domains`](tantale::core::domain::Domain).
//!     * A [`Solution`] from the set made of [`Optimizer`](tantale::core::optimizer::Optimizer) [`Domains`](tantale::core::domain::Domain).
//!
//! A [`Solution`] is statically typed by a [`Domain`], and a constant size `N` automatically defined by a the [`sp!`](../../../tantale/macros/macro.sp.html) macro.
//! A [`Solution`] cannot typed by a `dyn Trait`. This is why the [`Mixed`](tantale::core::Mixed) trait is used
//! within the [`sp!`](../../../tantale/macros/macro.sp.html) macro, to create `enum` of used [`Domains`](tantale::core::Domain) and
//! and `enum` of [`TypeDom`](tantale::core::Domain::TypeDom).
//! 
//! A [`Solution`] is made of a [`Box`]`<[`[`TypeDom`](tantale::core::Domain::TypeDom)`]>` of length `N`.
//! [`Solutions`](Solution) are strongly bounded to [`Searchspace`](tantale::core::Searchspace), from which
//! most of them will be created via simplified methods.
//! The type [`Solution`]`<D, const N: usize>`, is then used to statically constrain
//! the [`Objective`](tantale::core::Objective) and [`Optimizer`](tantale::core::Optimizer) inputs.
//! The unique `id` of a solution is computed according to a static atomic variable `SOL_ID`, and the `pid` of the process creating that solution.
//!

use crate::domain::Domain;

use std::{
    fmt::{Debug, Display},
    sync::atomic::{AtomicUsize, Ordering},
};
#[cfg(feature="par")]
use rayon::prelude::*;

static SOL_ID: AtomicUsize = AtomicUsize::new(0);

/// A solution of the [`Objective`] or of the [`Optimizer`] [`Domains`](Domain).
/// The solution is mostly defined by an associated unique [`Domain`].
///
/// # Attributes
/// * `id` : `(usize, u32)` - Contains the ID of the solution combined with the PID of the process.
/// * `x` : [`Vec`]`<D::`[`TypeDom`](Domain::TypeDom)`>` - A vector of [`TypeDom`](Domain::TypeDom)
///
pub struct Solution<D, const N: usize>
where
    D: Domain + Clone + Display + Debug,
    D::TypeDom: Default + Copy + Clone + Display + Debug,
{
    pub id: (usize, u32), // ID + PID for parallel algorithms.
    pub x: Box<[D::TypeDom]>,
}

impl<D, const N: usize> Solution<D, N>
where
    D: Domain + Clone + Display + Debug,
    D::TypeDom: Default + Copy + Clone + Display + Debug,
{
    /// Creates a new solution with a boxed slice.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real};
    /// 
    /// let x = vec![0.0;5].into_boxed_slice();
    /// let real_sol : Solution<Real,5> = Solution::new(std::process::id(),x);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    pub fn new(pid: u32, x: Box<[D::TypeDom]>) -> Self {
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        Solution { id: (id, pid), x }
    }

    pub fn new_default(pid: u32) -> Self {
        Self::new(pid, Self::default_x())
    }

    pub fn default_x() -> Box<[D::TypeDom]> {
        vec![D::TypeDom::default();N].into_boxed_slice()
    }
    
    pub fn new_vec(size:usize) -> Vec<Self> {
        let mut v = Vec::new();
        v.reserve_exact(size);
        v
    }

    pub fn new_default_vec(pid:u32, size:usize)-> Vec<Self>{
        let mut v = Self::new_vec(size);
        for _ in 0..size{
            v.push(Self::new_default(pid));
        }
        v
    }

    pub fn new_fill_vec(pid:u32, size:usize, x: Box<[D::TypeDom]>)-> Vec<Self>{
        let mut v = Self::new_vec(size);
        for _ in 0..size{
            v.push(Self::new(pid, x.clone()));
        }
        v
    }

    pub fn twin<B>(&self, x: Box<[B::TypeDom]>) -> Solution<B, N> 
    where
        B: Domain + Clone + Display + Debug,
        B::TypeDom: Default + Copy + Clone + Display + Debug,
    {
        Solution { id: self.id, x }
    }
}

#[cfg(feature="par")]
impl<D, const N: usize> Solution<D, N>
where
    D: Domain + Clone + Display + Debug,
    D::TypeDom: Default + Copy + Clone + Display + Debug + Send + Sync,
{
    pub fn par_new_default_vec(pid:u32, size:usize)-> Vec<Self>{
        (0..size).into_par_iter().map(|_| Self::new_default(pid)).collect()
    }
}