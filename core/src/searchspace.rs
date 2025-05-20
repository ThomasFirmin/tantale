//! A [`Searchspace`] is a trait defining interactions with an actual searchspace made of [`Var`](tantale::core::Var).
//! The [`sp!`](../../../tantale/macros/macro.sp.html) handles the complex relationships between [`Domains`](tantale::core::Domain)
//! to create a [`Sp`] instance made of a [`Vec`] of [`Var`](tantale::core::Var).
//!
//! # Example
//!
//! ```
//!
//! use tantale::core::{uniform_cat, uniform_nat, uniform_real,
//!                     Bool, Cat, Nat, Real, Searchspace};
//! use tantale::macros::sp;
//! 
//! static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
//!    
//! sp!(
//!     a | Real(0.0,1.0)                   |                               ;
//!     b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
//!     c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
//!     d | Bool()                          | Real(0.0,1.0)                 ;
//! );
//! 
//! 
//! let mut rng =rand::rng();
//! 
//! let sp = get_searchspace();
//! 
//! let obj = sp.sample_obj(&mut rng, std::process::id());
//! let opt = sp.onto_opt(&obj); // Map obj => opt
//! 
//! // Paired solutions have the same ID
//! println!("Obj ID : {} <=> Opt ID : {}", obj.id.0, opt.id.0);
//! 
//! ```
//! 
//! ## Note
//! 
//! In the following examples and for readability, a helper macro `initialize_sp_example!()` to create the searchspace contained in
//! a module named `sp`.
//! 

#[macro_export]
#[doc(hidden)]
macro_rules! initialize_sp_example {
    () => {
        mod sp{
            use tantale::core::{uniform_cat, uniform_nat, uniform_real,
                                Bool, Cat, Nat, Real, Searchspace};
            use tantale::macros::sp;
            
            static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
            
            sp!(
                a | Real(0.0,1.0)                   |                               ;
                b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
                c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
                d | Bool()                          | Real(0.0,1.0)                 ;
            );
        }
    };
}


use crate::domain::Domain;
use crate::solution::Solution;
use crate::variable::Var;

use std::fmt::{Debug, Display};
use rand::prelude::ThreadRng;

#[cfg(feature="par")]
use rayon::prelude::*;

/// The trait [`Searchspace`] defines
pub trait Searchspace<Obj, Opt, const N : usize>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{

    /// Maps a [`Solution`]`<Obj, N>` onto an [`Solution`]`<Opt, N>`,
    /// using the [`onto_opt_fn`](tantale::core::Var::onto_opt_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    /// 
    /// # Example
    ///
    /// ```
    /// tantale::core::initialize_sp_example!();
    /// 
    /// use tantale::core::Searchspace;
    /// 
    /// let mut rng =rand::rng();
    /// 
    /// let sp = sp::get_searchspace();
    /// 
    /// let obj = sp.sample_obj(&mut rng, std::process::id());
    /// let opt = sp.onto_opt(&obj); // Map obj => opt
    /// 
    /// for (i,o) in obj.x.into_iter().zip(&opt.x){
    ///     println!("Obj: {} => Opt: {}", i, o);
    /// }
    /// 
    /// ```
    fn onto_opt(&self, inp: &Solution<Obj, N>) -> Solution<Opt, N>;
    /// Maps a [`Solution`]`<Opt, N>` onto an [`Solution`]`<Obj, N>`,
    /// using the [`onto_obj_fn`](tantale::core::Var::onto_obj_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    /// 
    /// # Example
    ///
    /// ```
    /// tantale::core::initialize_sp_example!();
    /// 
    /// use tantale::core::Searchspace;
    /// 
    /// let mut rng =rand::rng();
    /// 
    /// let sp = sp::get_searchspace();
    /// 
    /// let opt = sp.sample_opt(&mut rng, std::process::id());
    /// let obj = sp.onto_obj(&opt);
    /// 
    /// for (i,o) in opt.x.into_iter().zip(obj.x){
    ///     println!("Opt: {} => Obj: {}", i, o);
    /// }
    /// 
    /// ```
    fn onto_obj(&self, inp: &Solution<Opt, N>) -> Solution<Obj, N>;
    /// Sample a random [`Solution`]`<Obj, N>` 
    /// using the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    /// 
    /// # Example
    ///
    /// ```
    /// tantale::core::initialize_sp_example!();
    /// 
    /// use tantale::core::Searchspace;
    /// 
    /// let mut rng =rand::rng();
    /// 
    /// let sp = sp::get_searchspace();
    /// 
    /// let obj = sp.sample_obj(&mut rng, std::process::id());
    /// 
    /// for i in obj.x.into_iter(){
    ///     println!("{}", i);
    /// }
    /// 
    /// ```
    fn sample_obj(&self, rng: &mut ThreadRng, pid:u32) -> Solution<Obj, N>;
    /// Sample a random [`Solution`]`<Opt, N>` 
    /// using the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    /// 
    /// # Example
    ///
    /// ```
    /// tantale::core::initialize_sp_example!();
    /// 
    /// use tantale::core::Searchspace;
    /// 
    /// let mut rng =rand::rng();
    /// 
    /// let sp = sp::get_searchspace();
    /// 
    /// let opt = sp.sample_opt(&mut rng, std::process::id());
    /// 
    /// for i in opt.x.into_iter(){
    ///     println!("{}", i);
    /// }
    /// 
    /// ```
    fn sample_opt(&self, rng: &mut ThreadRng, pid:u32) -> Solution<Opt, N>;
    /// Maps a [`Vec`] of [`Solution`]`<Obj, N>` onto a [`Vec`] [`Solution`]`<Opt, N>`,
    /// using the [`onto_opt_fn`](tantale::core::Var::onto_opt_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    /// 
    /// # Example
    ///
    /// ```
    /// tantale::core::initialize_sp_example!();
    /// 
    /// use tantale::core::Searchspace;
    /// 
    /// let mut rng =rand::rng();
    /// 
    /// let sp = sp::get_searchspace();
    /// 
    /// let vec_obj = sp.sample_vec_obj(&mut rng, std::process::id(), 10);
    /// let vec_opt = sp.onto_vec_opt(&vec_obj); // Map obj => opt
    /// 
    /// for (obj,opt) in vec_obj.iter().zip(vec_opt){
    ///     println!("[");
    ///     for (i,o) in obj.x.into_iter().zip(opt.x){
    ///         println!("Obj: {} => Opt: {}", i, o);
    ///     }
    ///     println!("]\n");
    /// }
    /// 
    /// ```
    fn onto_vec_obj(&self, inp: &Vec<Solution<Opt, N>>) -> Vec<Solution<Obj, N>>;
    /// Maps a [`Solution`]`<Opt, N>` onto an [`Solution`]`<Obj, N>`,
    /// using the [`onto_obj_fn`](tantale::core::Var::onto_obj_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    /// 
    /// # Example
    ///
    /// ```
    /// tantale::core::initialize_sp_example!();
    /// 
    /// use tantale::core::Searchspace;
    /// 
    /// let mut rng =rand::rng();
    /// 
    /// let sp = sp::get_searchspace();
    /// 
    /// let vec_opt = sp.sample_vec_opt(&mut rng, std::process::id(), 10);
    /// let vec_obj = sp.onto_vec_obj(&vec_opt);
    /// 
    /// for (opt,obj) in vec_opt.iter().zip(vec_obj){
    ///     println!("[");
    ///     for (i,o) in opt.x.into_iter().zip(obj.x){
    ///         println!("Opt: {} => Obj: {}", i, o);
    ///     }
    ///     println!("]\n");
    /// }
    /// 
    /// ```
    fn onto_vec_opt(&self, inp: &Vec<Solution<Obj, N>>) -> Vec<Solution<Opt, N>>;
    /// Sample a random [`Solution`]`<Obj, N>` 
    /// using the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    /// 
    /// # Example
    ///
    /// ```
    /// tantale::core::initialize_sp_example!();
    /// 
    /// use tantale::core::Searchspace;
    /// 
    /// let mut rng =rand::rng();
    /// 
    /// let sp = sp::get_searchspace();
    /// 
    /// let vec_obj = sp.sample_vec_obj(&mut rng, std::process::id(), 10);
    /// 
    /// for obj in vec_obj{
    ///     println!("[");
    ///     for i in obj.x.into_iter(){
    ///         println!("Obj: {}", i);
    ///     }
    ///     println!("]\n");
    /// }
    /// 
    /// ```
    /// 
    fn sample_vec_obj(&self, rng: &mut ThreadRng, pid:u32, size:usize) -> Vec<Solution<Obj, N>>;
    /// Sample a random [`Solution`]`<Opt, N>` 
    /// using the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    /// 
    /// ```
    /// tantale::core::initialize_sp_example!();
    /// 
    /// use tantale::core::Searchspace;
    /// 
    /// let mut rng =rand::rng();
    /// 
    /// let sp = sp::get_searchspace();
    /// 
    /// let vec_opt = sp.sample_vec_opt(&mut rng, std::process::id(), 10);
    /// 
    /// for opt in vec_opt{
    ///     println!("[");
    ///     for i in opt.x.into_iter(){
    ///         println!("Opt: {}", i);
    ///     }
    ///     println!("]\n");
    /// }
    /// 
    /// ```
    fn sample_vec_opt(&self, rng: &mut ThreadRng, pid:u32, size:usize) -> Vec<Solution<Opt, N>>;
}

#[cfg(feature="par")]
pub trait ParSearchspace<Obj, Opt, const N : usize>: Searchspace<Obj, Opt, N>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    /// Parallel version of [`onto_obj`](Searchspace::onto_obj) using [rayon].
    fn par_onto_obj(&self, inp: &Solution<Opt, N>) -> Solution<Obj, N>;
    /// Parallel version of [`onto_opt`](Searchspace::onto_opt) using [rayon].
    fn par_onto_opt(&self, inp: &Solution<Obj, N>) -> Solution<Opt, N>;
    /// Parallel version of [`sample_obj`](Searchspace::sample_obj) using [rayon].
    fn par_sample_obj(&self, pid:u32) -> Solution<Obj, N>;
    /// Parallel version of [`sample_opt`](Searchspace::sample_opt) using [rayon].
    fn par_sample_opt(&self, pid:u32) -> Solution<Opt, N>;
    /// Parallel version of [`onto_vec_obj`](Searchspace::onto_vec_obj) using [rayon].
    fn par_onto_vec_obj(&self, inp: &Vec<Solution<Opt, N>>) -> Vec<Solution<Obj, N>>;
    /// Parallel version of [`onto_vec_opt`](Searchspace::onto_vec_opt) using [rayon].
    fn par_onto_vec_opt(&self, inp: &Vec<Solution<Obj, N>>) -> Vec<Solution<Opt, N>>;
    /// Parallel version of [`sample_vec_obj`](Searchspace::sample_vec_obj) using [rayon].
    fn par_sample_vec_obj(&self, pid:u32, size:usize) -> Vec<Solution<Obj, N>>;
    /// Parallel version of [`sample_vec_opt`](Searchspace::sample_vec_opt) using [rayon].
    fn par_sample_vec_opt(&self, pid:u32, size:usize) -> Vec<Solution<Opt, N>>;
}

pub struct Sp<Obj, Opt, const N : usize>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    pub variables: Box<[Var<Obj, Opt>]>,
}

impl <Obj,Opt,const N : usize> Searchspace<Obj,Opt,N> for Sp<Obj,Opt,N>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    fn onto_obj(&self, inp: &Solution<Opt, N>) -> Solution<Obj, N>{
        let mut out = inp.twin(Solution::<Obj, N>::default_x());
        out.x.iter_mut().zip(&inp.x).zip(&self.variables).for_each(
            |((o,i),v)|
            *o = v.onto_obj(i).unwrap()
        );
        out
    }

    fn onto_opt(&self, inp: &Solution<Obj, N>) -> Solution<Opt, N>{
        let mut out = inp.twin(Solution::<Opt, N>::default_x());
        out.x.iter_mut().zip(&inp.x).zip(&self.variables).for_each(
            |((o,i),v)|
            *o = v.onto_opt(i).unwrap()
        );
        out
    }
    
    fn sample_obj(&self, rng: &mut ThreadRng, pid:u32) -> Solution<Obj, N> {
        let mut out = Solution::new_default(pid);
        out.x.iter_mut().zip(&self.variables).for_each(
            |(inp,var)|
            *inp = var.sample_obj(rng)
        );
        out
    }
    
    fn sample_opt(&self, rng: &mut ThreadRng, pid:u32) -> Solution<Opt, N> {
        let mut out = Solution::new_default(pid);
        out.x.iter_mut().zip(&self.variables).for_each(
            |(inp,var)|
            *inp = var.sample_opt(rng)
        );
        out
    }
    
    fn onto_vec_obj(&self, inp: &Vec<Solution<Opt, N>>) -> Vec<Solution<Obj, N>> {
        inp.iter().map(|i| self.onto_obj(i)).collect()
    }
    
    fn onto_vec_opt(&self, inp: &Vec<Solution<Obj, N>>) -> Vec<Solution<Opt, N>> {
        inp.iter().map(|i| self.onto_opt(i)).collect()
    }
    
    fn sample_vec_obj(&self, rng: &mut ThreadRng, pid:u32, size:usize) -> Vec<Solution<Obj, N>>  {
        (0..size).into_iter().map(|_| self.sample_obj(rng, pid)).collect()
    }
    
    fn sample_vec_opt(&self, rng: &mut ThreadRng, pid:u32, size:usize) -> Vec<Solution<Opt, N>>  {
        (0..size).into_iter().map(|_| self.sample_opt(rng, pid)).collect()
    }
    
}


#[cfg(feature="par")]
impl <Obj,Opt,const N : usize> ParSearchspace<Obj,Opt,N> for Sp<Obj,Opt,N>
where
    Obj: Domain + Clone + Display + Debug + Send + Sync,
    Opt: Domain + Clone + Display + Debug + Send + Sync,
    Obj::TypeDom : Default + Copy + Clone + Display + Debug + Send + Sync,
    Opt::TypeDom : Default + Copy + Clone + Display + Debug + Send + Sync,
{
    fn par_onto_obj(&self, inp: &Solution<Opt, N>) -> Solution<Obj, N>{
        let inpiter = inp.x.par_iter();
        let variter = self.variables.par_iter();

        let outx = inpiter.zip(variter).map(
            |(i,v)|
            v.onto_obj(i).unwrap()
        ).collect();
        inp.twin(outx)
    }

    fn par_onto_opt(&self, inp: &Solution<Obj, N>) -> Solution<Opt, N>{
        let inpiter = inp.x.par_iter();
        let variter = self.variables.par_iter();

        let outx = inpiter.zip(variter).map(
            |(i,v)|
            v.onto_opt(i).unwrap()
        ).collect();
        inp.twin(outx)
    }
    
    fn par_sample_obj(&self, pid:u32) -> Solution<Obj, N> {
        let variter = self.variables.par_iter();
        let outx = variter.map_init(
            ||rand::rng(),
            |rng,var|
            var.sample_obj(rng)
        ).collect();
        Solution::new(pid, outx)
    }
    
    fn par_sample_opt(&self, pid:u32) -> Solution<Opt, N> {
        let variter = self.variables.par_iter();
        let outx = variter.map_init(
            ||rand::rng(),
            |rng,var|
            var.sample_opt(rng)
        ).collect();
        Solution::new(pid, outx)
    }
    
    fn par_onto_vec_obj(&self, inp: &Vec<Solution<Opt, N>>) -> Vec<Solution<Obj, N>> {
        inp.par_iter().map(|sol|self.par_onto_obj(sol)).collect()
    }
    
    fn par_onto_vec_opt(&self, inp: &Vec<Solution<Obj, N>>) -> Vec<Solution<Opt, N>> {
        inp.par_iter().map(|sol|self.par_onto_opt(sol)).collect()
    }
    
    fn par_sample_vec_obj(&self, pid:u32, size:usize) -> Vec<Solution<Obj, N>> {
        (0..size).into_par_iter().map(|_| self.par_sample_obj(pid)).collect()
    }
    
    fn par_sample_vec_opt(&self, pid:u32, size:usize) -> Vec<Solution<Opt, N>> {
        (0..size).into_par_iter().map(|_| self.par_sample_opt(pid)).collect()
    }
}