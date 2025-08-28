//! A [`Searchspace`] is a trait defining interactions with an actual searchspace made of [`Var`](tantale::core::Var).
//! The [`sp!`](../../../tantale/macros/macro.sp.html) handles the complex relationships between [`Domains`](tantale::core::Domain)
//! to create a [`Sp`] instance made of a [`Vec`] of [`Var`](tantale::core::Var).
//!
//! You can also define a [`Sp`] directly within the [`Objective`] function with [`objective!`](../../../tantale/macros/macro.objective.html).
//! It also handles [`Mixed`](tantale::core::Mixed) searchspaces, but it can also automatically disambiguate the mixed input vector of the function to
//! get the right value at the right place.
//!
//! # Example with [`sp!`](../../../tantale/macros/macro.sp.html).
//!
//! ```
//!     use tantale::core::{uniform_cat, uniform_nat, uniform_real,
//!                         Bool, Cat, Nat, Real, Searchspace,
//!                         EmptyInfo, Solution, SId};
//!     use tantale::macros::{sp,Outcome};
//!     use std::sync::Arc;
//!     use serde::{Serialize,Deserialize};
//!
//!     static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
//!     
//!     sp!(
//!         a | Real(0.0,1.0)                   |                               ;
//!         b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
//!         c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
//!         d | Bool()                          | Real(0.0,1.0)                 ;
//!     );
//!
//!     let mut rng =rand::rng();
//!     let sp = get_searchspace();
//!     let info = std::sync::Arc::new(EmptyInfo{});
//!
//!     let obj = sp.sample_obj(Some(&mut rng), info.clone());
//!     let opt = sp.onto_opt(obj.clone()); // Map obj => opt
//!     // Paired solutions have the same ID
//!     let id1 : SId = obj.get_id();
//!     let id2 : SId = opt.get_id();
//!     println!("Obj ID : {} <=> Opt ID : {}", id1.id, id2.id);
//!
//!     #[derive(Outcome,Serialize,Deserialize)]
//!     struct OutStruct{pub out:f64}
//!
//!     // _TantaleMixedObj is automatically created by sp!
//!     fn compute_obj(tantale_in : Arc::<[<_TantaleMixedObj as Domain >::TypeDom]>) -> OutStruct{
//!         let a = match tantale_in[0]{
//!             _TantaleMixedObjTypeDom::Real(value) => value,
//!             _ => unreachable!(""),
//!         };
//!         let b = match tantale_in[1]{
//!             _TantaleMixedObjTypeDom::Nat(value) => value,
//!             _ => unreachable!(""),
//!         };
//!         let c = match tantale_in[2]{
//!             _TantaleMixedObjTypeDom::Cat(ref value) => value,
//!             _ => unreachable!(""),
//!         };
//!         let d = match tantale_in[3]{
//!             _TantaleMixedObjTypeDom::Bool(value) => value,
//!             _ => unreachable!(""),
//!         };
//!         println!("a {}, b {}, c {}, d {}", a, b, c, d);
//!
//!         OutStruct{out:42.0}
//!     }
//!
//!     let o = compute_obj(obj.get_x());
//!     println!("OUT {}", o.out);
//!
//!
//! ```
//!
//! # Example with [`objective!`](../../../tantale/macros/macro.objective.html).
//!
//! ```
//! mod searchspace{
//!     use tantale::core::domain::{Real,Bool,Cat,Nat};
//!     use tantale::core::domain::sampler::{uniform_nat, uniform_cat, uniform_real};
//!     use tantale::macros::{objective,Outcome};
//!     use serde::{Serialize,Deserialize};
//!
//!     static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
//!     #[derive(Outcome,Serialize,Deserialize)]
//!     pub struct OutStruct{pub out:f64}
//!
//!     objective!(
//!         pub fn example() -> OutStruct {
//!             let a = [! a | Real(0.0,1.0)                   |                               !];
//!             let b = [! b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real !];
//!             let c = [! c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real !];
//!             let d = [! d | Bool()                          | Real(0.0,1.0)                 !];
//!                
//!             println!("a {}, b {}, c {}, d {}", a, b, c, d);
//!             OutStruct{out:42.0}
//!         }
//!     );
//! }
//!
//! use tantale::core::{EmptyInfo,Searchspace,Solution,SId};
//! let sp = searchspace::get_searchspace();
//! let info = std::sync::Arc::new(EmptyInfo{});
//! let mut rng = rand::rng();
//!
//! let sample = sp.sample_obj(Some(&mut rng),info);
//! let id1: SId = sample.get_id();
//! let out = searchspace::example(sample.get_x());
//! println!("ID : {} -- Out {}",id1.id,out.out);
//! ```

use crate::{
    domain::Domain,
    optimizer::ArcVecArc,
    solution::{Computed, Id, Partial, SolInfo, Solution},
    Codomain, Outcome,
};

use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

type ComputedOut<SolId, ADom, BDom, Cod, Out, Info> = (
    Arc<Computed<SolId, ADom, Cod, Out, Info>>,
    Arc<Computed<SolId, BDom, Cod, Out, Info>>,
);

/// The [`Searchspace`] handles the [`Domains`](Domain) of the [`Objective`], of the [`Optimizer`], and the [`Codomain`].
pub trait Searchspace<SolId, Obj, Opt, SInfo>
where
    SInfo: SolInfo,
    Obj: Domain,
    Opt: Domain,
    SolId: Id,
{
    /// Maps a [`Partial`] of type `Obj` onto an [`Partial`] of type `Opt`.
    /// It uses the [`onto_opt_fn`](tantale::core::Var::onto_opt_fn) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId};
    /// use std::sync::Arc;
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let obj : Arc<Partial<SId,_,_>> = sp.sample_obj(Some(&mut rng), info.clone());
    /// let opt : Arc<Partial<SId,_,_>> = sp.onto_opt(obj.clone()); // Map obj => opt
    ///
    /// for (i,o) in obj.get_x().iter().zip(opt.get_x().iter()){
    ///     println!("Obj: {} => Opt: {}", i, o);
    /// }
    ///
    /// ```
    fn onto_opt(&self, inp: Arc<Partial<SolId, Obj, SInfo>>) -> Arc<Partial<SolId, Opt, SInfo>>;
    /// Maps a [`Partial`] of type `Opt` onto an [`Partial`] of type `Obj`.
    /// It uses the [`onto_obj_fn`](tantale::core::Var::onto_obj_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId};
    /// use std::sync::Arc;
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let opt : Arc<Partial<SId,_,_>> = sp.sample_opt(Some(&mut rng), info.clone());
    /// let obj : Arc<Partial<SId,_,_>> = sp.onto_obj(opt.clone());
    ///
    /// for (i,o) in opt.get_x().iter().zip(obj.get_x().iter()){
    ///     println!("Opt: {} => Obj: {}", i, o);
    /// }
    ///
    /// ```
    fn onto_obj(&self, inp: Arc<Partial<SolId, Opt, SInfo>>) -> Arc<Partial<SolId, Obj, SInfo>>;
    /// Sample a random [`Partial`] of type `Obj`.
    /// It uses the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId};
    /// use std::sync::Arc;
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let obj : Arc<Partial<SId,_,_>> = sp.sample_obj(Some(&mut rng), info.clone());
    ///
    /// for i in obj.get_x().iter(){
    ///     println!("{}", i);
    /// }
    ///
    /// ```
    fn sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        info: Arc<SInfo>,
    ) -> Arc<Partial<SolId, Obj, SInfo>>;
    /// Sample a random [`Partial`] of type `Opt`.
    /// It uses the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace, SId};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId};
    /// use std::sync::Arc;
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let opt : Arc<Partial<SId,_,_>>= sp.sample_opt(Some(&mut rng), info.clone());
    ///
    /// for i in opt.get_x().iter(){
    ///     println!("{}", i);
    /// }
    ///
    /// ```
    fn sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        info: Arc<SInfo>,
    ) -> Arc<Partial<SolId, Opt, SInfo>>;
    /// Check if a given `Obj` [`Solution`] is within the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId};
    /// use std::sync::Arc;
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let obj : Arc<Partial<SId,_,_>> = sp.sample_obj(Some(&mut rng), info.clone());
    ///
    /// sp.is_in_obj(obj.clone());
    ///
    /// ```
    fn is_in_obj<S>(&self, inp: Arc<S>) -> bool
    where
        S: Solution<SolId, Obj, SInfo> + Send + Sync;
    /// Check if a given `Opt` [`Solution`] is within the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId};
    /// use std::sync::Arc;
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let opt : Arc<Partial<SId,_,_>> = sp.sample_opt(Some(&mut rng), info.clone());
    ///
    /// sp.is_in_opt(opt.clone());
    ///
    /// ```
    fn is_in_opt<S>(&self, inp: Arc<S>) -> bool
    where
        S: Solution<SolId, Opt, SInfo> + Send + Sync;
    /// Maps a [`Vec`] of [`Solution`] of type `Obj` onto a [`Vec`] [`Solution`] of type `Opt`.
    /// It uses the [`onto_opt_fn`](tantale::core::Var::onto_opt_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId,ArcVecArc};
    /// use std::sync::Arc;
    ///
    /// let mut rng = rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let vec_obj : ArcVecArc<Partial<SId,_,_>> = sp.vec_sample_obj(Some(&mut rng), 10, info.clone());
    /// let vec_opt : ArcVecArc<Partial<SId,_,_>> = sp.vec_onto_opt(vec_obj.clone()); // Map obj => opt
    ///
    /// for (obj,opt) in vec_obj.iter().zip(vec_opt.clone().iter()){
    ///     println!("[");
    ///     for (i,o) in obj.get_x().iter().zip(opt.get_x().iter()){
    ///         println!("Obj: {} => Opt: {}", i, o);
    ///     }
    ///     println!("]\n");
    /// }
    ///
    /// ```
    fn vec_onto_obj(
        &self,
        inp: ArcVecArc<Partial<SolId, Opt, SInfo>>,
    ) -> ArcVecArc<Partial<SolId, Obj, SInfo>>;
    /// Maps a [`Partial`] of type `Opt` onto an [`Partial`] of type `Obj`.
    /// It uses the [`onto_obj_fn`](tantale::core::Var::onto_obj_fn) from
    /// the corresponding [`variables`](Searchspace::variables). To main
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId,ArcVecArc};
    /// use std::sync::Arc;
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let vec_opt : ArcVecArc<Partial<SId,_,_>> = sp.vec_sample_opt(Some(&mut rng), 10, info.clone());
    /// let vec_obj : ArcVecArc<Partial<SId,_,_>> = sp.vec_onto_obj(vec_opt.clone());
    ///
    /// for (opt,obj) in vec_opt.iter().zip(vec_obj.clone().iter()){
    ///     println!("[");
    ///     for (i,o) in opt.get_x().iter().zip(obj.get_x().iter()){
    ///         println!("Opt: {} => Obj: {}", i, o);
    ///     }
    ///     println!("]\n");
    /// }
    ///
    /// ```
    fn vec_onto_opt(
        &self,
        inp: ArcVecArc<Partial<SolId, Obj, SInfo>>,
    ) -> ArcVecArc<Partial<SolId, Opt, SInfo>>;
    /// Sample a random [`Partial`] of type `Obj`.
    /// It uses the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId,ArcVecArc};
    /// use std::{fmt::Debug, sync::Arc};
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let vec_obj : ArcVecArc<Partial<SId,_,_>> = sp.vec_sample_obj(Some(&mut rng), 10, info.clone());
    ///
    /// for obj in vec_obj.clone().iter(){
    ///     println!("Obj: {:?}", obj.get_x());
    /// }
    ///
    /// ```
    ///
    fn vec_sample_obj(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> ArcVecArc<Partial<SolId, Obj, SInfo>>;
    /// Sample a random [`Partial`] of type `Opt`.
    /// It uses the [`sampler_obj`](tantale::core::Var::sampler_obj) from
    /// the corresponding [`variables`](Searchspace::variables).
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId,ArcVecArc};
    /// use std::sync::Arc;
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let vec_opt : ArcVecArc<Partial<SId,_,_>> = sp.vec_sample_opt(Some(&mut rng), 10, info.clone());
    ///
    /// for obj in vec_opt.clone().iter(){
    ///     println!("Obj: {:?}", obj.get_x());
    /// }
    ///
    /// ```
    fn vec_sample_opt(
        &self,
        rng: Option<&mut ThreadRng>,
        size: usize,
        info: Arc<SInfo>,
    ) -> ArcVecArc<Partial<SolId, Opt, SInfo>>;
    /// Check if all [`Solutions`](tantale::core::Solution) from a given [`Vec`] of `Opt` [`Solution`] is in the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId,ArcVecArc};
    /// use std::sync::Arc;
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let vobj : ArcVecArc<Partial<SId,_,_>> = sp.vec_sample_obj(Some(&mut rng), 10, info.clone());
    ///
    /// sp.vec_is_in_obj(vobj.clone());
    ///
    /// ```
    fn vec_is_in_obj<S>(&self, inp: ArcVecArc<S>) -> bool
    where
        S: Solution<SolId, Obj, SInfo> + Send + Sync;
    /// Check if all [`Solution`](tantale::core::Solution) from a given [`Vec`] of `Opt` [`Solution`] is in the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{uniform_cat, uniform_nat, uniform_real,
    /// #                            Bool, Cat, Nat, Real, Searchspace};
    /// #        use tantale::macros::sp;
    /// #
    /// #        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
    /// #
    /// #        sp!(
    /// #            a | Real(0.0,1.0)                   |                               ;
    /// #            b | Nat(0,100)       => uniform_nat | Real(0.0,1.0) => uniform_real ;
    /// #            c | Cat(&ACTIVATION) => uniform_cat | Real(0.0,1.0) => uniform_real ;
    /// #            d | Bool()                          | Real(0.0,1.0)                 ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Searchspace,Solution,Partial,EmptyInfo,SId,ArcVecArc};
    /// use std::sync::Arc;
    ///
    /// let mut rng =rand::rng();
    ///
    /// let sp = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let vopt : ArcVecArc<Partial<SId,_,_>> = sp.vec_sample_opt(Some(&mut rng), 10, info.clone());
    ///
    /// sp.vec_is_in_opt(vopt.clone());
    ///
    /// ```
    fn vec_is_in_opt<S>(&self, inp: ArcVecArc<S>) -> bool
    where
        S: Solution<SolId, Opt, SInfo> + Send + Sync;

    /// Creates a pair of [`Computed`] [`Solutions`](Solution) of [`Domain`] types `Dom` and `B`
    /// from a pair of [`twin`](Partial::twin) [`Partial`] of types `A` and `B`, and a shared [`Codomain`].
    fn computed<Cod, Out>(
        &self,
        xa: Arc<Partial<SolId, Obj, SInfo>>,
        xb: Arc<Partial<SolId, Opt, SInfo>>,
        y: Arc<Cod::TypeCodom>,
    ) -> ComputedOut<SolId, Obj, Opt, Cod, Out, SInfo>
    where
        Cod: Codomain<Out>,
        Out: Outcome,
        Cod::TypeCodom: Serialize + for<'a> Deserialize<'a>,
    {
        (
            Arc::new(Computed::new(xa, y.clone())),
            Arc::new(Computed::new(xb, y)),
        )
    }
}

pub mod spbase;
pub use spbase::Sp;

pub mod sppar;
pub use sppar::ParSp;
