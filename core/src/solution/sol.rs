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
//! The types of [`Solution`], are then used to statically constrain
//! the [`Objective`](tantale::core::Objective) and [`Optimizer`](tantale::core::Optimizer) inputs.
//! 
//! The outcome of the evaluation of a [`Solution`] is stored in `y`.
//! 
//! The unique `id` of a solution is computed according to a static atomic variable `SOL_ID`, and the `pid` of the process creating that solution.
//!

use crate::domain::{Domain, TypeDom};
use crate::objective::codomain::Codomain;
use crate::objective::Outcome;

use std::{
    fmt::{Debug, Display},
    sync::{Arc, atomic::{AtomicUsize, Ordering}},
};

#[cfg(feature="par")]
use rayon::prelude::*;

static SOL_ID: AtomicUsize = AtomicUsize::new(0);

/// A solution of the [`Objective`](tantale::core::Objective) or of the [`Optimizer`](tantale::core::Optimizer) [`Domains`](Domain).
/// The solution is mostly defined by an associated unique [`Domain`].
///
/// # Attributes
/// * `id` : `(usize, u32)` - Contains the ID of the solution combined with the PID of the process.
/// * `x` : [`Box`]`<[Dom::`[`TypeDom`](Domain::TypeDom)`]>` - A vector of [`TypeDom`](Domain::TypeDom)
/// * `y` : [`Option`]`<`[`Arc`]`<Cod<Out>>` - State of the evaluation of a solution. It is an optional [`TypeCodom`](tantale::core::objective::comdomain::Codomain::TypeCodom), 
///   if the [`Solution`] was not yet evaluated, then it is `None`.
pub struct Solution<Dom, Cod, Out, const N:usize>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    pub id: (usize, u32), // ID + PID for parallel algorithms.
    pub x: Box<[TypeDom<Dom>]>,
    pub y: Option<Arc<Cod::TypeCodom>>,
}

impl<Dom, Cod, Out, const N: usize> Solution<Dom, Cod, Out, N>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    /// Creates a new solution from a boxed slice.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,SingleCodomain,HashOut};
    /// use std::sync::Arc;
    /// 
    /// let x = vec![0.0;5].into_boxed_slice();
    /// let real_sol : Solution<Real,SingleCodomain<HashOut>,HashOut,5> = Solution::new(std::process::id(),x, None);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    pub fn new(pid: u32, x: Box<[TypeDom<Dom>]>, y: Option<Arc<Cod::TypeCodom>>) -> Self {
        let id = SOL_ID.fetch_add(1, Ordering::Relaxed);
        Solution { id: (id, pid), x, y}
    }
    
    /// Creates a new solution containing default value of the domain.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,SingleCodomain,HashOut};
    /// 
    /// let real_sol : Solution<Real,SingleCodomain<HashOut>,HashOut,5> = Solution::new_default(std::process::id());
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    pub fn new_default(pid: u32) -> Self {
        Self::new(pid, Self::default_x(), None)
    }

    /// Creates the default array of a [`Solution`].
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,SingleCodomain,HashOut};
    /// use std::sync::Arc;
    /// 
    /// let x = Solution::<Real,SingleCodomain<HashOut>,HashOut,5>::default_x();
    /// let real_sol : Solution<Real,SingleCodomain<HashOut>,HashOut,5> = Solution::new(std::process::id(),x, None);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    pub fn default_x() -> Box<[TypeDom<Dom>]> {
        vec![TypeDom::<Dom>::default();N].into_boxed_slice()
    }

    /// Creates an empty [`Vec`] of [`Solutions`](Solution) with reserved capacity
    /// with `size`.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,SingleCodomain,HashOut};
    /// 
    /// let mut vec_sol = Solution::<Real,SingleCodomain<HashOut>,HashOut,5>::new_vec(10);
    /// 
    /// for _ in 0..10{
    ///     vec_sol.push(Solution::new_default(std::process::id()));
    /// }
    /// 
    /// for sol in &vec_sol{
    ///     println!("[");
    ///     for elem in &sol.x{
    ///         println!("{},", elem);
    ///     }
    ///     println!("], ");
    /// }
    /// 
    /// ```
    pub fn new_vec(size:usize) -> Vec<Self> {
        let mut v = Vec::new();
        v.reserve_exact(size);
        v
    }

    /// Creates a [`Vec`] of [`Solutions`](Solution) with default `x`.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,SingleCodomain,HashOut};
    /// 
    /// let vec_sol = Solution::<Real,SingleCodomain<HashOut>,HashOut,5>::new_default_vec(std::process::id(), 10);
    /// 
    /// for sol in &vec_sol{
    ///     println!("[");
    ///     for elem in &sol.x{
    ///         println!("{},", elem);
    ///     }
    ///     println!("], ");
    /// }
    /// 
    /// ```
    pub fn new_default_vec(pid:u32, size:usize)-> Vec<Self>{
        let mut v = Self::new_vec(size);
        for _ in 0..size{
            v.push(Self::new_default(pid));
        }
        v
    }

    /// Creates a [`Vec`] of [`Solutions`](Solution) containing a slice `x`.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,SingleCodomain,HashOut};
    /// 
    /// 
    /// let x = vec![0.0,1.0,2.0,3.0,4.0].into_boxed_slice();
    /// let vec_sol = Solution::<Real,SingleCodomain<HashOut>,HashOut,5>::new_fill_vec(std::process::id(), 10, x);
    /// 
    /// for sol in &vec_sol{
    ///     println!("[");
    ///     for elem in &sol.x{
    ///         println!("{},", elem);
    ///     }
    ///     println!("], ");
    /// }
    /// 
    /// ```
    pub fn new_fill_vec(pid:u32, size:usize, x: Box<[TypeDom<Dom>]>)-> Vec<Self>{
        let mut v = Self::new_vec(size);
        for _ in 0..size{
            v.push(Self::new(pid, x.clone(), None));
        }
        v
    }

    /// Given a [`Solution`] of type [`Self`] and a slice of type `B`, 
    /// creates the twin [`Solution`] of type [`B`].
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used  in [`onto_opt`](tantale::core::searchspace::onto_opt) 
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,Int,SingleCodomain,HashOut};
    /// use std::sync::Arc;
    /// 
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0].into_boxed_slice();
    /// let x_2 = vec![5,6,7,8].into_boxed_slice();
    /// let real_sol = Solution::<Real,SingleCodomain<HashOut>,HashOut,5>::new(std::process::id(), x_1, None);
    /// let int_sol : Solution<Int,SingleCodomain<HashOut>,HashOut,5> = real_sol.twin(x_2);
    /// 
    /// println!("REAL ID : {},{}", real_sol.id.0,real_sol.id.1);
    /// println!("INT ID : {},{}", int_sol.id.0,int_sol.id.1);
    /// 
    /// for (elem1, elem2) in real_sol.x.iter().zip(int_sol.x.iter()){
    ///     println!("{},{}", elem1, elem2);
    /// }
    /// 
    /// ```
    pub fn twin<B>(&self, x: Box<[TypeDom<B>]>) -> Solution<B,Cod,Out,N> 
    where
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
    {
        Solution { id: self.id, x, y:self.y.clone()}
    }

    /// Updates a [`Solution`] with an optional and related [`TypeCodom`](Codom::TypeCodom) from an [`Outcome`].
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,Int,HashOut,Codomain,SingleCodomain};
    /// use std::sync::Arc;
    /// 
    /// let codom = SingleCodomain::new(
    ///     |h : &HashOut| *h.get("y").unwrap()
    /// );
    /// 
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0].into_boxed_slice();
    /// let mut real_sol = Solution::<Real,SingleCodomain<HashOut>,HashOut,5>::new(std::process::id(),x_1, None);
    /// 
    /// // Obtain some outcomes from the objective function.
    /// let outcome = HashOut::from([("y", 1.0), ("other", 2.0)]);
    /// 
    /// let y_elem = Arc::new(codom.get_elem(&outcome));
    /// 
    /// real_sol.update(Some(y_elem.clone()));
    /// 
    /// assert_eq!(real_sol.y.unwrap().value, 1.0);
    /// 
    /// ```
    pub fn update(&mut self, y: Option<Arc<Cod::TypeCodom>>)
    {
        self.y = y.clone();
    }
}

#[cfg(feature="par")]
impl<Dom,Cod,Out,const N: usize> Solution<Dom,Cod,Out,N>
where
    Dom: Domain + Clone + Display + Debug,
    Cod : Codomain<Out> + Send + Sync,
    Out : Outcome + Send + Sync,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug + Send + Sync,
    Cod::TypeCodom : Send + Sync,
{
    /// Parallel version of [`new_default_vec`](Solution::new_default_vec) using [`rayon`].
    pub fn par_new_default_vec(pid:u32, size:usize)-> Vec<Self>{
        (0..size).into_par_iter().map(|_| Self::new_default(pid)).collect()
    }
}