use crate::domain::{Domain, TypeDom};
use crate::solution::{Solution,Computed};
use crate::objective::{Codomain,Outcome};
use crate::optimizer::SolInfo;

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

#[cfg(feature="par")]
use rayon::prelude::*;

pub trait Partial<Dom,Info>:Solution<Dom,Info>
where
    Dom: Domain + Clone + Display + Debug,
    Info : SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    /// Creates a new [`Partial`] from a given `pid` of the process it is being created,
    /// a concrete sample `x` from the set of [`Domains`](Domain), and a [`SolInfo`].
    fn new(pid:u32, x : Arc<[TypeDom<Dom>]>, info : Arc<Info>) -> Self;

    /// Given a [`Partial`] of type [`Self`] and a slice of type [`TypeDom`]`<B>`, 
    /// creates the twin [`Partial`] of type [`B`].
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used in [`onto_opt`](tantale::core::searchspace::onto_opt) 
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{PartialSol,Real,Int,EmptyInfo};
    /// use std::sync::Arc;
    /// 
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0];
    /// let x_2 = vec![5,6,7,8];
    /// let pid = std::process::id();
    /// let info = Arc::new(EmptyInfo{});
    /// 
    /// let real_sol = PartialSol::<Real,EmptyInfo,5>::new(pid,x_1,info);
    /// let int_sol : PartialSol<Int,EmptyInfo,5> = real_sol.twin(x_2);
    /// 
    /// println!("REAL ID : {},{}", real_sol.id.0,real_sol.id.1);
    /// println!("INT ID : {},{}", int_sol.id.0,int_sol.id.1);
    /// 
    /// for (elem1, elem2) in real_sol.x.iter().zip(int_sol.x.iter()){
    ///     println!("{},{}", elem1, elem2);
    /// }
    /// 
    /// ```
    fn twin<Twin,B,T>(&self, x: T) -> Twin
    where 
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        Twin:Partial<B,Info>,
        T : AsRef<[TypeDom<B>]>
    {
        Twin::new
    }

    /// Creates a pair of [`Computed`] [`Solutions`](Solution) of type [`Domain`] types `Dom` and `B`
    /// from a pair of [`twin`](Partial::twin) [`Partial`] of type [`Self`] and `B`, and a shared [`Codomain`].
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,SingleCodomain,HashOut, EmptyInfo};
    /// use std::sync::Arc;
    /// 
    /// let x = vec![0.0;5].into_boxed_slice();
    /// let real_sol : Solution<Real,SingleCodomain<HashOut>,HashOut, EmptyInfo, 5> = Solution::new(std::process::id(),x,None,None);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    fn computed<B,BDom,Ca,Cb,Cod,Out>(self, xb : B, y: Arc<Cod::TypeCodom>) -> 
    (
        Ca,
        Cb,
    )
    where 
        Ca:Computed<Dom,Info,Cod,Out>,
        Cb:Computed<BDom,Info,Cod,Out>,
        B:Partial<BDom,Info>,
        BDom: Domain + Clone + Display + Debug,
        TypeDom<BDom>: Default + Copy + Clone + Display + Debug,
        Cod: Codomain<Out>,
        Out : Outcome;
    
}

/// A non-evaluated [`Solution`].
///
/// # Attributes
/// * `id` : `(usize, u32)` - Contains the ID of the solution combined with the PID of the process.
/// * `x` : [`Arc`]`<[Dom::`[`TypeDom`](Domain::TypeDom)`]>` - A vector of [`TypeDom`](Domain::TypeDom)
/// * `info` : `[`Arc`]`<Info>` - Information given by the [`Optimizer`] and linked to a specific [`Solution`]
///   if the [`Solution`] was not yet evaluated, then it is `None`.
pub struct PartialSol<Dom,Info,const N:usize>
where
    Dom: Domain + Clone + Display + Debug,
    Info : SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    pub id: (usize, u32), // ID + PID for parallel algorithms.
    pub x: Arc<[TypeDom<Dom>]>,
    pub info : Arc<Info>,
}

impl <Dom,Info,const N:usize> Solution<Dom,Info> for PartialSol<Dom,Info,N>
where
    Dom: Domain + Clone + Display + Debug,
    Info : SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    fn get_id(&self)->(usize, u32) {
        self.id
    }

    fn get_x(&self)->Arc<[TypeDom<Dom>]> {
        self.x.clone()
    }

    fn get_info(&self)->Arc<Info> {
        self.info.clone()
    }

    /// Creates a new [`PartialSol`] from an [`Arc`] slice.
    /// 
    /// # Attributes
    /// 
    /// * `pid` : `u32` - The process id from which the [`PartialSol`] is created.
    /// * `x` : [`Arc`]`<[`[`TypeDom`]`<Dom>]>` - A basic solution from the [`Searchspace`].
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{PartialSol,Real,EmptyInfo};
    /// use std::sync::Arc;
    /// 
    /// let x = Arc::from(vec![0.0;5]);
    /// let pid = std::process::id();
    /// let info = Arc::new(EmptyInfo{});
    /// let real_sol : PartialSol<Real,_,5> = PartialSol::new(pid,x,info);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    fn new(pid:u32, x : Arc<[TypeDom<Dom>]>, info : Arc<Info>) -> Self
    {
        let solid = Self::_get_new_id();
        let id = (pid,solid);
        PartialSol { id, x, info }
    }
}

impl <Dom,Info,const N:usize> Partial<Dom,Info> for PartialSol<Dom,Info,N>
where
    Dom: Domain + Clone + Display + Debug,
    Info : SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    fn twin<Twin,B,T>(&self, x: T) -> Twin
    where
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        T : AsRef<[TypeDom<B>]>,
        Twin: Partial<B,Info>,
    {
        let idtwin = self.get_id();
        Twi
    }

    fn computed<B,BDom,Ca,Cb,Cod,Out>(self, xb : B, y: Arc<Cod::TypeCodom>) -> 
    (
        Ca,
        Cb,
    )
    where 
        Ca:Computed<Dom,Info,Cod,Out>,
        Cb:Computed<BDom,Info,Cod,Out>,
        B:Partial<BDom,Info>,
        BDom: Domain + Clone + Display + Debug,
        TypeDom<BDom>: Default + Copy + Clone + Display + Debug,
        Cod: Codomain<Out>,
        Out : Outcome
    {
        todo!()
    }
}

impl <Dom, Info, const N:usize> PartialSol<Dom,Info,N>
where
    Dom: Domain + Clone + Display + Debug,
    Info : SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{       
    /// Create a [`PartialSol`] from a [`AsRef`]`<[TypeDom<Dom>]>`, and a given [`SolInfo`].
    /// 
    /// # Attributes
    /// 
    /// * `pid` : `u32` - The process id from which the [`PartialSol`] is created.
    /// * `x` : `T` - A base representation of a solution from the [`Searchspace`] implementing [`AsRef`]`<[TypeDom<Dom>]>`.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{PartialSol,Real,EmptyInfo};
    /// use std::sync::Arc;
    /// 
    /// let x = vec![0.0;5];
    /// let pid = std::process::id();
    /// let info = Arc::new(EmptyInfo{});
    /// let real_sol : PartialSol<Real,_,5> = PartialSol::from_slice(pid,x,info);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    pub fn from_slice<T>(pid:u32,x : T, info:Arc<Info>) -> Self
    where 
        T : AsRef<[TypeDom<Dom>]>,
    {
        let xarc: Arc<[TypeDom<Dom>]>= Arc::from(x.as_ref());
        Self::new(pid, xarc, info)
    }

    /// Creates the default slice of a [`PartialSol`].
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{PartialSol,Real,EmptyInfo};
    /// use std::sync::Arc;
    /// 
    /// let x = PartialSol::<Real,EmptyInfo,5>::default_x();
    /// let info = Arc::new(EmptyInfo{});
    /// let pid = std::process::id();
    /// let real_sol : PartialSol<Real,EmptyInfo,5> = PartialSol::new(pid,x,info);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    pub fn default_x() -> Arc<[TypeDom<Dom>]> {
        Arc::from(vec![TypeDom::<Dom>::default();N])
    }

    /// Creates a default [`PartialSol`].
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{PartialSol,Real,EmptyInfo};
    /// use std::sync::Arc;
    /// 
    /// let x = PartialSol::<Real,EmptyInfo,5>::default_x();
    /// let info = Arc::new(EmptyInfo{});
    /// let pid = std::process::id();
    /// let real_sol : PartialSol<Real,EmptyInfo,5> = PartialSol::new_default(pid,info);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    pub fn new_default(pid:u32, info : Arc<Info>) -> Self {
        Self::new(pid, Self::default_x(), info)
    }

    /// Creates an empty slice of [`PartialSols`](PartialSol) with `size` reserved capacity.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{PartialSol,Real,EmptyInfo};
    /// 
    /// let mut vec_sol = PartialSol::<Real,EmptyInfo,5>::new_vec(10);
    /// let info = Arc::new(EmptyInfo{});
    /// 
    /// for _ in 0..10{
    ///     let pid = std::process::id();
    ///     vec_sol.push(PartialSol::new_default(pid,info.clone()));
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

    /// Creates a [`Vec`] of [`PartialSols`](PartialSol) with default `x`.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{PartialSol,Real,EmptyInfo};
    /// 
    /// let pid = std::process::id();
    /// let info = Arc::new(EmptyInfo{});
    /// let vec_sol = PartialSol::<Real,EmptyInfo,5>::new_default_vec(pid,info,10);
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
    pub fn new_default_vec(pid:u32, info : Arc<Info>,size:usize)-> Vec<Self>{
        let mut v = Self::new_vec(size);
        for _ in 0..size{
            v.push(Self::new_default(pid,info.clone()));
        }
        v
    }

    pub fn twin<B,T>(&self, x: T) -> PartialSol<B,Info,N> 
    where 
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        T : AsRef<[TypeDom<B>]>,
    {
        let xarc: Arc<[TypeDom<B>]>= Arc::from(x.as_ref());
        PartialSol{
            id: self.id,
            x: xarc,
            info: self.info.clone(),
        }
    }
    /// Creates a new solution from two [`PartialSol`] of respectively the [`Opt`](Optimizer) and the [`Obj`](Objective)
    /// [`Domain`] type.
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{Solution,Real,SingleCodomain,HashOut, EmptyInfo};
    /// use std::sync::Arc;
    /// 
    /// let x = vec![0.0;5].into_boxed_slice();
    /// let real_sol : Solution<Real,SingleCodomain<HashOut>,HashOut, EmptyInfo, 5> = Solution::new(std::process::id(),x,None,None);
    /// 
    /// for elem in &real_sol.x{
    ///     println!("{},", elem);
    /// }
    /// 
    /// ```
    pub fn pair<BDom,Cod,Out>(self, xb : PartialSol<BDom,Info,N>, y: Arc<Cod::TypeCodom>) -> 
    (
        ComputedSol<Dom, Cod, Out, Info, N>,
        ComputedSol<BDom, Cod, Out, Info, N>,
    )
    where 
        Cod: Codomain<Out>,
        Out : Outcome,
        BDom: Domain + Clone + Display + Debug,
        TypeDom<BDom>: Default + Copy + Clone + Display + Debug,
    {
        (
            ComputedSol::<Dom, Cod, Out, Info, N>::new(self, y.clone()),
            ComputedSol::<BDom, Cod, Out, Info, N>::new(xb, y.clone())
        )
    }
}


#[cfg(feature="par")]
impl<Dom,Info,const N: usize> PartialSol<Dom, Info, N>
where
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug + Send + Sync,
    Info : SolInfo + Send + Sync,
{
    /// Parallel version of [`new_default_vec`](Solution::new_default_vec) using [`rayon`].
    pub fn par_new_default_vec(pid:u32, info : Arc<Info>, size:usize)-> Vec<Self>{
        (0..size).into_par_iter().map(|_| Self::new_default(pid, info.clone())).collect()
    }
}