//! A [`Searchspace`] is a trait defining interactions with an actual searchspace made of [`Var`](tantale::core::Var).
//! The [`sp!`](../../../tantale/macros/macro.sp.html) handles the complex relationships between [`Domains`](tantale::core::Domain)
//! to create a [`Sp`] instance made of a [`Vec`] of [`Var`](tantale::core::Var).
//!
//! # Example
//!
//! ```
//!
//! use tantale::core::{uniform_cat, uniform_nat, uniform_real,
//!                     Bool, Cat, Nat, Real, Searchspace,
//!                     PartialSol, EmptyInfo
//!                     };
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
//! let obj : PartialSol<_, EmptyInfo, 4> = sp.sample_obj(&mut rng, std::process::id());
//! let opt = sp.onto_opt(&obj); // Map obj => opt
//!
//! // Paired solutions have the same ID
//! println!("Obj ID : {} <=> Opt ID : {}", obj.id.0, opt.id.0);
//!
//! ```
//!
//! ## Notes
//!
//! In the following examples and for readability, a helper macro `init_sp_example!()` is used
//! to create the searchspace contained in a module named `sp`.
//!

#[macro_export]
#[doc(hidden)]
macro_rules! init_sp_example {
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

use crate::{
    domain::Domain,
    solution::{Partial, SolInfo, Solution},
};

use rand::prelude::ThreadRng;
use std::fmt::{Debug, Display};
use std::sync::Arc;

/// The [`Searchspace`] handles the [`Domains`](Domain) of the [`Objective`], of the [`Optimizer`], and the [`Codomain`].
pub trait Searchspace<PObj, POpt, Obj, Opt, SInfo>
where
    SInfo: SolInfo,
    Obj: Domain + Clone + Display + Debug,
    PObj: Partial<Obj, SInfo>,
    Opt: Domain + Clone + Display + Debug,
    POpt: Partial<Opt, SInfo>,
{
    /// Maps a [`Partial`]`<Obj,SInfo>` onto an [`Partial`]`<Opt,SInfo>`,
    /// using the [`onto_opt_fn`](tantale::core::Var::onto_opt_fn) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace,PartialSol,EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let obj : PartialSol<_, EmptyInfo, 4> = sp.sample_obj(&mut rng, std::process::id());
    /// let opt = sp.onto_opt(&obj); // Map obj => opt
    ///
    /// for (i,o) in obj.x.into_iter().zip(&opt.x){
    ///     println!("Obj: {} => Opt: {}", i, o);
    /// }
    ///
    /// ```
    fn onto_opt(&self, inp: &PObj) -> POpt;
    /// Maps a [`Partial`]`<Opt,SInfo,N>` onto an [`Partial`]`<Obj,SInfo,N>`,
    /// using the [`onto_obj_fn`](tantale::core::Var::onto_obj_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace,PartialSol,EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let opt : PartialSol<_,EmptyInfo,4> = sp.sample_opt(&mut rng, std::process::id());
    /// let obj = sp.onto_obj(&opt);
    ///
    /// for (i,o) in opt.x.into_iter().zip(obj.x){
    ///     println!("Opt: {} => Obj: {}", i, o);
    /// }
    ///
    /// ```
    fn onto_obj(&self, inp: &POpt) -> PObj;
    /// Sample a random [`Partial`]`<Obj,SInfo,N>`
    /// using the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace,PartialSol,EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let obj : PartialSol<_,EmptyInfo,4> = sp.sample_obj(&mut rng, std::process::id());
    ///
    /// for i in obj.x.into_iter(){
    ///     println!("{}", i);
    /// }
    ///
    /// ```
    fn sample_obj(&self, rng: &mut ThreadRng, pid: u32, info: Arc<SInfo>) -> PObj;
    /// Sample a random [`Partial`]`<Opt,SInfo,N>`
    /// using the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace,PartialSol,EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let opt : PartialSol<_,EmptyInfo,4> = sp.sample_opt(&mut rng, std::process::id());
    ///
    /// for i in opt.x.into_iter(){
    ///     println!("{}", i);
    /// }
    ///
    /// ```
    fn sample_opt(&self, rng: &mut ThreadRng, pid: u32, info: Arc<SInfo>) -> POpt;
    /// Check if a given `Obj` [`Solution`] is in the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace,PartialSol,EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let obj : PartialSol<_,EmptyInfo,4> = sp.sample_obj(&mut rng, std::process::id());
    ///
    /// sp.is_in_obj(&obj);
    ///
    /// ```
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<Obj, SInfo>;
    /// Check if a given `Opt` [`Solution`] is in the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace,PartialSol,EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let opt : PartialSol<_,EmptyInfo,4> = sp.sample_opt(&mut rng, std::process::id());
    ///
    /// sp.is_in_opt(&opt);
    ///
    /// ```
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<Opt, SInfo>;
    /// Maps a [`Vec`] of [`Solution`]`<Obj, N>` onto a [`Vec`] [`Solution`]`<Opt, N>`,
    /// using the [`onto_opt_fn`](tantale::core::Var::onto_opt_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace,PartialSol,EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let vec_obj : Vec<Solution<_,EmptyInfo,4>> = sp.vec_sample_obj(&mut rng, std::process::id(), 10);
    /// let vec_opt = sp.vec_onto_opt(&vec_obj); // Map obj => opt
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
    fn vec_onto_obj(&self, inp: &[POpt]) -> Vec<PObj>;
    /// Maps a [`Solution`]`<Opt, N>` onto an [`Solution`]`<Obj, N>`,
    /// using the [`onto_obj_fn`](tantale::core::Var::onto_obj_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace, Solution, SingleCodomain, HashOut, EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let vec_opt : Vec<Solution<_, SingleCodomain<HashOut>,HashOut, EmptyInfo, 4>> = sp.vec_sample_opt(&mut rng, std::process::id(), 10);
    /// let vec_obj = sp.vec_onto_obj(&vec_opt);
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
    fn vec_onto_opt(&self, inp: &[PObj]) -> Vec<POpt>;
    /// Sample a random [`Solution`]`<Obj, N>`
    /// using the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace, Solution, SingleCodomain, HashOut, EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let vec_obj : Vec<Solution<_, SingleCodomain<HashOut>,HashOut, EmptyInfo, 4>> = sp.vec_sample_obj(&mut rng, std::process::id(), 10);
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
    fn vec_sample_obj(
        &self,
        rng: &mut ThreadRng,
        pid: u32,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<PObj>;
    /// Sample a random [`Solution`]`<Opt, N>`
    /// using the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// ```
    /// tantale::core::init_sp_example!();
    /// use tantale::core::{Searchspace, Solution, SingleCodomain, HashOut, EmptyInfo};
    ///
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let vec_opt : Vec<Solution<_, SingleCodomain<HashOut>,HashOut, EmptyInfo, 4>> = sp.vec_sample_opt(&mut rng, std::process::id(), 10);
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
    fn vec_sample_opt(
        &self,
        rng: &mut ThreadRng,
        pid: u32,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<POpt>;
    /// Check if all [`Solutions`](tantale::core::Solution) from a given [`Vec`] of `Opt` [`Solution`] is in the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace, Solution, SingleCodomain, HashOut, EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let vobj : Vec<Solution<_, SingleCodomain<HashOut>,HashOut, EmptyInfo, 4>> = sp.vec_sample_obj(&mut rng, std::process::id(),10);
    ///
    /// sp.vec_is_in_obj(&vobj);
    ///
    /// ```
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<Obj, SInfo>;
    /// Check if all [`Solution`](tantale::core::Solution) from a given [`Vec`] of `Opt` [`Solution`] is in the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Searchspace, Solution, SingleCodomain, HashOut, EmptyInfo};
    ///
    /// tantale::core::init_sp_example!();
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    ///
    /// let vopt : Vec<Solution<_, SingleCodomain<HashOut>,HashOut, EmptyInfo, 4>> = sp.vec_sample_opt(&mut rng, std::process::id(),10);
    ///
    /// sp.vec_is_in_opt(&vopt);
    ///
    /// ```
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<Opt, SInfo>;
}

#[cfg(feature = "par")]
pub trait ParSearchspace<PObj, POpt, Obj, Opt, SInfo>:
    Searchspace<PObj, POpt, Obj, Opt, SInfo>
where
    SInfo: SolInfo,
    Obj: Domain + Clone + Display + Debug,
    PObj: Partial<Obj, SInfo>,
    Opt: Domain + Clone + Display + Debug,
    POpt: Partial<Opt, SInfo>,
{
    /// Parallel version of [`onto_obj`](Searchspace::onto_obj) using [rayon].
    fn par_onto_obj(&self, inp: &POpt) -> PObj;
    /// Parallel version of [`onto_opt`](Searchspace::onto_opt) using [rayon].
    fn par_onto_opt(&self, inp: &PObj) -> POpt;
    /// Parallel version of [`sample_obj`](Searchspace::sample_obj) using [rayon].
    fn par_sample_obj(&self, pid: u32, info: Arc<SInfo>) -> PObj;
    /// Parallel version of [`sample_opt`](Searchspace::sample_opt) using [rayon].
    fn par_sample_opt(&self, pid: u32, info: Arc<SInfo>) -> POpt;
    /// Parallel version of [`onto_vec_obj`](Searchspace::onto_vec_obj) using [rayon].
    fn par_vec_onto_obj(&self, inp: &[POpt]) -> Vec<PObj>;
    /// Parallel version of [`onto_vec_opt`](Searchspace::onto_vec_opt) using [rayon].
    fn par_vec_onto_opt(&self, inp: &[PObj]) -> Vec<POpt>;
    /// Parallel version of [`sample_vec_obj`](Searchspace::sample_vec_obj) using [rayon].
    fn par_vec_sample_obj(&self, pid: u32, size: usize, info: Arc<SInfo>) -> Vec<PObj>;
    /// Parallel version of [`sample_vec_opt`](Searchspace::sample_vec_opt) using [rayon].
    fn par_vec_sample_opt(&self, pid: u32, size: usize, info: Arc<SInfo>) -> Vec<POpt>;
    /// Parallel version of [`is_in_obj`](Searchspace::is_in_obj) using [rayon].
    fn par_is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<Obj, SInfo> + Send + Sync;
    /// Parallel version of [`is_in_opt`](Searchspace::is_in_opt) using [rayon].
    fn par_is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<Opt, SInfo> + Send + Sync;
    /// Parallel version of [`vec_is_in_obj`](Searchspace::vec_is_in_obj) using [rayon].
    fn par_vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<Obj, SInfo> + Send + Sync;
    /// Parallel version of [`vec_is_in_opt`](Searchspace::vec_is_in_opt) using [rayon].
    fn par_vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<Opt, SInfo> + Send + Sync;
}

pub mod spbase;
pub use spbase::Sp;