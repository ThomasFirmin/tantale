//! # Searchspace
//!
//! This module provides the [`Searchspace`] trait and related types for defining the optimization
//! search domain. A searchspace specifies the valid variable values that an optimizer can explore
//! and an objective function can evaluate.
//!
//! ## Overview
//!
//! A [`Searchspace`] is composed of [`Var`](crate::Var) instances, each defining a variable with:
//! - An **Obj domain**: The domain for the objective function's input
//! - An **Opt domain**: The domain for the optimizer's representation
//! - Mapping functions between these two domains via [`Onto`](crate::Onto)
//!
//! ## Dual Domain Architecture
//!
//! Tantale uses a **dual domain** approach where each variable has two representations:
//!
//! - **Obj (Objective)**: The domain used by the objective function. This is typically the natural
//!   parameter space of the problem (e.g., categorical choices, specific numeric ranges).
//! - **Opt (Optimizer)**: The domain used by the optimizer for search. This may be a continuous
//!   relaxation or transformation suitable for the optimization algorithm.
//!
//! The searchspace handles automatic mapping between these domains, allowing optimizers to work
//! in their preferred representation while objectives receive values in their expected format.
//!
//! ## Construction
//!
//! Searchspaces are typically constructed using macros:
//! - [`hpo!`](../../../tantale/macros/macro.hpo.html) - Creates a standalone searchspace instance
//! - [`objective!`](../../../tantale/macros/macro.objective.html) - Defines a searchspace within the objective function
//!
//!
//! # Example with [`hpo!`](../../../tantale/macros/macro.hpo.html).
//!
//! This is a mock example to illustrate the usage of the searchspace and its mapping functions.
//! See concrete examples within the repository for more detailed use cases.
//!
//! ```rust,ignore
//!
//!     use tantale::core::{Bool, Cat, Nat, Real, Searchspace,
//!                         Uniform, Bernoulli,
//!                         EmptyInfo, Solution, SId};
//!     use tantale::macros::{hpo,Outcome};
//!     use std::sync::Arc;
//!     use serde::{Serialize,Deserialize};
//!
//!     
//!     hpo!(
//!         a | Real(0.0,1.0,Uniform)            |                       ;
//!         b | Nat(0,100,Uniform)               | Real(0.0,1.0,Uniform) ;
//!         c | Cat(["relu", "tanh", "sigmoid"]) | Real(0.0,1.0,Uniform) ;
//!         d | Bool()                           | Real(0.0,1.0,Uniform) ;
//!     );
//!
//!     let mut rng: rand::rngs::ThreadRng = rand::rng();
//!     let sp = get_searchspace();
//!     let info = std::sync::Arc::new(EmptyInfo{});
//!
//!     let obj = sp.sample_obj(&mut rng, info.clone());
//!     let opt = sp.onto_opt(obj.clone()); // Map obj => opt
//!     // Paired solutions have the same ID
//!     let id1 : SId = obj.get_id();
//!     let id2 : SId = opt.get_id();
//!     println!("Obj ID : {} <=> Opt ID : {}", id1.id, id2.id);
//!
//!     #[derive(Outcome,Serialize,Deserialize)]
//!     struct OutStruct{pub out:f64}
//!
//!     // _TantaleMixedObj is automatically created by hpo!
//!     fn compute_obj(tantale_in : Arc::<[<_TantaleMixedObj as Domain >::TypeDom]>) -> OutStruct{
//!         let a = match tantale_in[0]{
//!             _TantaleMixedObjTypeDom::Real(value) => value,
//!             _ !(""),
//!         };
//!         let b = match tantale_in[1]{
//!             _TantaleMixedObjTypeDom::Nat(value) => value,
//!             _ !(""),
//!         };
//!         let c = match tantale_in[2]{
//!             _TantaleMixedObjTypeDom::Cat(ref value) => value,
//!             _ !(""),
//!         };
//!         let d = match tantale_in[3]{
//!             _TantaleMixedObjTypeDom::Bool(value) => value,
//!             _ !(""),
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
//!     use tantale::core::{Bool, Cat, Nat, Real, Uniform, Bernoulli};
//!     use tantale::macros::{objective,Outcome};
//!     use serde::{Serialize,Deserialize};
//!
//!     #[derive(Outcome,Debug,Serialize,Deserialize)]
//!     pub struct OutStruct{pub out:f64}
//!
//!     objective!(
//!         pub fn example() -> OutStruct {
//!             let a = [! a | Real(0.0,1.0,Uniform)    |                       !];
//!             let b = [! b | Nat(0,100,Uniform)       | Real(0.0,1.0,Uniform) !];
//!             let c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform)         | Real(0.0,1.0,Uniform) !];
//!             let d = [! d | Bool(Bernoulli(0.5))                   | Real(0.0,1.0,Uniform) !];
//!                
//!             println!("a {}, b {}, c {}, d {}", a, b, c, d);
//!             OutStruct{out:42.0}
//!         }
//!     );
//! }
//!
//! use tantale::core::{Sp, BaseSol, EmptyInfo,Searchspace,Solution,SId, HasId};
//! use searchspace::{ObjType, OptType};
//!
//! let sp = searchspace::get_searchspace();
//! let info = std::sync::Arc::new(EmptyInfo{});
//! let mut rng: rand::rngs::ThreadRng = rand::rng();
//!
//! let sample: BaseSol<SId,ObjType,EmptyInfo> =
//! <Sp<ObjType, OptType> as Searchspace<BaseSol<_,OptType,_>, _, _>>::sample_obj(&sp, &mut rng, info.clone());
//! let id1: SId = sample.id();
//! let out = searchspace::example(sample.clone_x());
//! println!("ID : {} -- Out {}",id1.id,out.out);
//! ```

use crate::{
    domain::onto::Linked,
    solution::{
        HasId, HasSolInfo, Id, IntoComputed, SolInfo, Solution, SolutionShape, shape::RawObj,
    },
};

use rand::prelude::Rng;
use std::sync::Arc;

pub type CompShape<Scp, SolOpt, SolId, SInfo, Cod, Out> =
    <<Scp as Searchspace<SolOpt, SolId, SInfo>>::SolShape as IntoComputed>::Computed<Cod, Out>;

pub type OptionCompShape<Scp, SolOpt, SolId, SInfo, Cod, Out> = Option<
    <<Scp as Searchspace<SolOpt, SolId, SInfo>>::SolShape as IntoComputed>::Computed<Cod, Out>,
>;

/// Core trait defining the searchspace for optimization problems.
///
/// A [`Searchspace`] manages the relationship between objective and optimizer representations
/// of solutions. It provides methods for sampling, mapping, and validating solutions in both
/// domains. This trait is central to Tantale's type-safe optimization framework.
///
/// # Dual [`Domains`](crate::Domain)
///
/// Each searchspace maintains two domain representations:
/// - **Obj domain** ([`Self::Obj`](Linked::Obj)): Used by the objective function
/// - **Opt domain** ([`Self::Opt`](Linked::Opt)): Used by the optimizer
///
/// The searchspace provides bidirectional mapping between these domains via
/// [`onto_opt`](Searchspace::onto_opt) and [`onto_obj`](Searchspace::onto_obj).
///
/// # Associated Types
///
/// - [`SolShape`](Searchspace::SolShape) - The paired solution shape containing both Obj and Opt representations.
///   It can be :
///   * a [`Pair`](crate::solution::shape::Pair) when both `Obj` and `Opt` sides are defined
///   * a [`Lone`](crate::solution::shape::Lone) when only the `Obj` side is defined
///
/// # Key Operations
///
/// ## Sampling
/// - [`sample_obj`](Searchspace::sample_obj) - Generate random Obj domain solution
/// - [`sample_opt`](Searchspace::sample_opt) - Generate random Opt domain solution
/// - [`sample_pair`](Searchspace::sample_pair) - Generate paired Obj/Opt solution ([`SolutionShape`])
///
/// ## Mapping
/// - [`onto_opt`](Searchspace::onto_opt) - Map Obj → Opt domain
/// - [`onto_obj`](Searchspace::onto_obj) - Map Opt → Obj domain
///
/// ## Validation
/// - [`is_in_obj`](Searchspace::is_in_obj) - Check if solution is valid in Obj domain
/// - [`is_in_opt`](Searchspace::is_in_opt) - Check if solution is valid in Opt domain
///
/// # Usage
///
/// ```ignore
/// // Create searchspace (usually via macro)
/// let sp = get_searchspace();
/// let mut rng: rand::rngs::ThreadRng = rand::rng();
/// let info = Arc::new(EmptyInfo);
///
/// // Optimizer generates in Opt domain
/// let opt_solution = sp.sample_opt(&mut rng, info.clone());
///
/// // Map to Obj domain for evaluation
/// let obj_solution = sp.onto_obj(opt_solution);
///
/// // Evaluate objective function
/// let outcome = objective_fn(obj_solution.get_x());
/// ```
///
/// # See Also
///
/// - [`Sp`] - Concrete searchspace implementation
/// - [`Var`](crate::Var) - Individual variable definitions
/// - [`Linked`] - Trait linking Obj and Opt domains
/// - [`hpo!`](../../../tantale/macros/macro.hpo.html) - Macro for creating searchspaces
/// - [`objective!`](../../../tantale/macros/macro.objective.html) - Macro for defining searchspaces and wrap objective functions
pub trait Searchspace<SolOpt, SolId, SInfo>: Linked
where
    SolOpt: Solution<SolId, Self::Opt, SInfo>,
    SolOpt::Twin<Self::Obj>: Solution<SolId, Self::Obj, SInfo, Twin<Self::Opt> = SolOpt>,
    SolId: Id,
    SInfo: SolInfo,
{
    /// The paired solution shape containing both Obj and Opt [`twin`](crate::Solution::twin) [`Solution`] representations.
    type SolShape: SolutionShape<
            SolId,
            SInfo,
            Obj = Self::Obj,
            Opt = Self::Opt,
            SolObj = SolOpt::Twin<Self::Obj>,
            SolOpt = SolOpt,
        > + HasId<SolId>
        + HasSolInfo<SInfo>
        + IntoComputed;

    /// Maps a solution from the Obj domain to the Opt domain.
    ///
    /// This method transforms an objective function solution into the optimizer's representation
    /// using the [`Onto`](crate::Onto) mappings defined for each variable in the searchspace.
    /// The resulting solution maintains the same [`Id`] as the input.
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution in the Obj domain
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Bounded, Mixed, Real, Pair, Solution, EmptyInfo, SId, Searchspace, SolutionShape};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let obj: BaseSol<SId, Mixed, EmptyInfo> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::sample_obj(&sp, &mut rng, info.clone());
    /// let opt: Pair<BaseSol<SId, Mixed, EmptyInfo>, BaseSol<SId, Real, EmptyInfo>, SId, _, _, _> = sp.onto_opt(obj); // Map obj => opt
    ///
    /// for (i, o) in opt.get_sobj().get_x().iter().zip(opt.get_sopt().get_x().iter()) {
    ///     println!("Obj: {:?} => Opt: {}", i, o);
    /// }
    ///
    /// ```
    fn onto_opt(&self, inp: SolOpt::Twin<Self::Obj>) -> Self::SolShape;
    /// Maps a solution from the Opt domain to the Obj domain.
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution in the Opt domain
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Bounded, Mixed, Real, Solution, EmptyInfo, SId, Searchspace, SolutionShape};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let opt: BaseSol<SId, Bounded<f64>, EmptyInfo> = sp.sample_opt(&mut rng, info.clone());
    /// let obj = sp.onto_obj(opt); // Map opt => obj
    ///
    /// for (i, o) in obj.get_sopt().get_x().iter().zip(obj.get_sobj().get_x().iter()) {
    ///     println!("Opt: {} => Obj: {:?}", i, o);
    /// }
    ///
    /// ```
    fn onto_obj(&self, inp: SolOpt) -> Self::SolShape;
    /// Generates a random solution in the Obj (objective function) domain.
    ///
    /// Samples each variable using its Obj domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `info` - Shared solution metadata
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let obj: BaseSol<SId, Mixed, EmptyInfo> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::sample_obj(&sp, &mut rng, info.clone());
    ///
    /// for i in obj.get_x().iter(){
    ///     println!("{:?}", i);
    /// }
    ///
    /// ```
    fn sample_obj<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt::Twin<Self::Obj>;
    /// Generates a random solution in the Opt (optimizer) domain.
    ///
    /// Samples each variable using its Opt domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `info` - Shared solution metadata
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Bounded, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let opt: BaseSol<SId, Bounded<f64>, EmptyInfo> = sp.sample_opt(&mut rng, info.clone());
    ///
    /// for i in opt.get_x().iter(){
    ///     println!("{}", i);
    /// }
    ///
    /// ```
    fn sample_opt<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> SolOpt;
    /// Generates a random paired solution containing both Obj and Opt representations.
    ///
    /// Samples in the Obj domain and automatically creates the paired Opt representation
    /// via mapping. Both representations share the same [`Id`].
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `info` - Shared solution metadata
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Real, Mixed, Pair, Solution, EmptyInfo, SId, Searchspace, SolutionShape, HasId};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let pair: Pair<BaseSol<SId, Mixed, EmptyInfo>, BaseSol<SId, Real, EmptyInfo>, SId, _, _, _> = sp.sample_pair(&mut rng, info.clone());
    ///
    /// println!("Paired ID: {:?}", pair.id());
    /// println!("Obj: {:?}", pair.get_sobj().get_x());
    /// println!("Opt: {:?}", pair.get_sopt().get_x());
    ///
    /// ```
    ///
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<SInfo>) -> Self::SolShape;
    /// Validates that a solution belongs to the Obj domain of this searchspace.
    ///
    /// Checks each variable value against its Obj domain constraints using
    /// [`Domain::is_in`](crate::Domain::is_in).
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution to validate
    ///
    /// # Returns
    ///
    /// `true` if all variable values satisfy their Obj domain constraints, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let obj: BaseSol<SId, Mixed, EmptyInfo> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::sample_obj(&sp, &mut rng, info.clone());
    ///
    /// assert!(<Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::is_in_obj::<BaseSol<SId, Mixed, EmptyInfo>>(&sp, &obj));
    ///
    /// ```
    fn is_in_obj<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = RawObj<Self::SolShape, SolId, SInfo>>
            + Send
            + Sync;
    /// Validates that a solution belongs to the Opt domain of this searchspace.
    ///
    /// Checks each variable value against its Opt domain constraints using
    /// [`Domain::is_in`](crate::Domain::is_in).
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution to validate
    ///
    /// # Returns
    ///
    /// `true` if all variable values satisfy their Opt domain constraints, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Real, Mixed, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let opt: BaseSol<SId, Real, EmptyInfo> = sp.sample_opt(&mut rng, info.clone());
    ///
    /// assert!(<Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::is_in_opt::<BaseSol<SId, Real, EmptyInfo>>(&sp, &opt));
    ///
    /// ```
    fn is_in_opt<S>(&self, inp: &S) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = SolOpt::Raw> + Send + Sync;
    /// Maps a [`Solution`] of type `Opt` onto an [`Solution`] of type `Obj`.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace, Pair, SolutionShape};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let vec_opt: Vec<BaseSol<SId, Real, EmptyInfo>> = sp.vec_sample_opt(&mut rng, 10, info.clone());
    /// let vec_pair: Vec<Pair<BaseSol<SId, Mixed, EmptyInfo>, BaseSol<SId, Real, EmptyInfo>, _, _, _, _>> = sp.vec_onto_obj(vec_opt);
    ///
    /// for pair in vec_pair {
    ///     for (i, o) in pair.get_sopt().get_x().iter().zip(pair.get_sobj().get_x().iter()){
    ///         println!("Opt: {} => Obj: {:?}", i, o);
    ///     }
    /// }
    ///
    /// ```
    fn vec_onto_obj(&self, inp: Vec<SolOpt>) -> Vec<Self::SolShape>;
    /// Maps a vector of solutions from the Obj domain to the Opt domain.
    ///
    /// Batch version of [`onto_opt`](Searchspace::onto_opt) for efficient processing of
    /// multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `inp` - Vector of solutions in the Obj domain
    ///
    /// # Returns
    ///
    /// Vector of paired solutions with both Obj and Opt representations.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Real, Mixed, Solution, EmptyInfo, SId, Searchspace, Pair, SolutionShape};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let vec_obj: Vec<BaseSol<SId, Mixed, EmptyInfo>> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_sample_obj(&sp, &mut rng, 10, info.clone());
    /// let vec_pair: Vec<Pair<BaseSol<SId, Mixed, EmptyInfo>, BaseSol<SId, Real, EmptyInfo>, _, _, _, EmptyInfo>> = sp.vec_onto_opt(vec_obj); // Map obj => opt
    ///
    /// for pair in vec_pair.iter(){
    ///     for (i, o) in pair.get_sobj().get_x().iter().zip(pair.get_sopt().get_x().iter()){
    ///         println!("Obj: {:?} => Opt: {}", i, o);
    ///     }
    /// }
    ///
    /// ```
    fn vec_onto_opt(&self, inp: Vec<SolOpt::Twin<Self::Obj>>) -> Vec<Self::SolShape>;
    /// Generates multiple random solutions in the Obj domain.
    ///
    /// Batch version of [`sample_obj`](Searchspace::sample_obj) for efficient generation
    /// of multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of randomly sampled solutions in the Obj domain, each with a unique [`Id`].
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Real, Mixed, Solution, EmptyInfo, SId, Searchspace};
    /// use std::{fmt::Debug, sync::Arc};
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let vec_obj: Vec<BaseSol<SId, Mixed, EmptyInfo>> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_sample_obj(&sp, &mut rng, 10, info.clone());
    ///
    /// for obj in vec_obj.into_iter(){
    ///     println!("Obj: {:?}", obj.get_x());
    /// }
    ///
    /// ```
    ///
    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Self::Obj>>;
    /// Generates multiple random solutions in the Opt domain.
    ///
    /// Batch version of [`sample_opt`](Searchspace::sample_opt) for efficient generation
    /// of multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of randomly sampled solutions in the Opt domain, each with a unique [`Id`].
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let vec_opt: Vec<BaseSol<SId, Real, EmptyInfo>> = sp.vec_sample_opt(&mut rng, 10, info.clone());
    ///
    /// for opt in vec_opt.into_iter(){
    ///     println!("Opt: {:?}", opt.get_x());
    /// }
    ///
    /// ```
    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize, info: Arc<SInfo>) -> Vec<SolOpt>;
    /// Generates multiple random [`SolutionShape`] with both Obj and Opt representations.
    ///
    /// Batch version of [`sample_pair`](Searchspace::sample_pair) for efficient generation
    /// of multiple paired solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of paired solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of paired solutions ([`SolutionShape`]), each with both Obj and Opt representations sharing a unique [`Id`].
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Solution, Sp, BaseSol, Real, Mixed, Pair, EmptyInfo, HasId, SId, Searchspace, SolutionShape};
    /// use std::{fmt::Debug, sync::Arc};
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let vec_pair: Vec<Pair<BaseSol<SId, Mixed, EmptyInfo>, BaseSol<SId, Real, EmptyInfo>, SId, _, _, _>> = sp.vec_sample_pair(&mut rng, 10, info.clone());
    ///
    /// for pair in vec_pair.iter(){
    ///     println!("Paired: {:?}", pair.id());
    ///     println!("Obj: {:?}", pair.get_sobj().get_x());
    ///     println!("Opt: {:?}", pair.get_sopt().get_x());
    /// }
    ///
    /// ```
    ///
    fn vec_sample_pair<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape>;

    /// Generates multiple `Obj` solutions and applies a transformation to each.
    ///
    /// This is similar to [`vec_sample_obj`](Searchspace::vec_sample_obj), but allows a
    /// user-provided closure to post-process each sampled `Obj` solution before returning the vector.
    /// This is useful for adding custom metadata, mutating fields, or enforcing additional
    /// constraints on the generated `Obj` solutions.
    ///
    /// # Parameters
    ///
    /// * `f` - Closure applied to each sampled `Obj` solution
    /// * `rng` - Random number generator
    /// * `size` - Number of `Obj` solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of `Obj` solutions after applying `f` to each sample.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, Mixed, Real, EmptyInfo, Searchspace, SId, FidelitySol, HasFidelity};
    /// use std::sync::Arc;
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sols: Vec<FidelitySol<SId, Mixed, EmptyInfo>> = <Sp<Mixed, Real> as Searchspace<FidelitySol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_apply_obj(
    ///     &sp,
    ///     |mut sol: FidelitySol<SId, Mixed, EmptyInfo>| { sol.set_fidelity(1.0); sol },
    ///     &mut rng,
    ///     5,
    ///     info.clone(),
    /// );
    ///
    /// for sol in sols.iter() {
    ///     assert_eq!(sol.fidelity().0, 1.0);
    /// }
    /// ```
    fn vec_apply_obj<F, R>(
        &self,
        f: F,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<SolOpt::Twin<Self::Obj>>
    where
        F: Fn(SolOpt::Twin<Self::Obj>) -> SolOpt::Twin<Self::Obj> + Send + Sync,
        R: Rng;

    /// Generates multiple `Opt` solutions and applies a transformation to each.
    ///
    /// This is similar to [`vec_sample_opt`](Searchspace::vec_sample_opt), but allows a
    /// user-provided closure to post-process each sampled `Opt` solution before returning the vector.
    /// This is useful for adding custom metadata, mutating fields, or enforcing additional
    /// constraints on the generated `Opt` solutions.
    ///
    /// # Parameters
    ///
    /// * `f` - Closure applied to each sampled `Opt` solution
    /// * `rng` - Random number generator
    /// * `size` - Number of `Opt` solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of `Opt` solutions after applying `f` to each sample.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{Sp, Mixed, Real, FidelitySol, EmptyInfo, Searchspace, SId, HasFidelity};
    /// use std::sync::Arc;
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sols: Vec<FidelitySol<SId, Real, EmptyInfo>> = <Sp<Mixed, Real> as Searchspace<FidelitySol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_apply_opt(
    ///     &sp,
    ///     |mut sol: FidelitySol<SId, Real, EmptyInfo>| { sol.set_fidelity(1.0); sol },
    ///     &mut rng,
    ///     5,
    ///     info.clone(),
    /// );
    ///
    /// for sol in sols.iter() {
    ///     assert_eq!(sol.fidelity().0, 1.0);
    /// }
    /// ```
    fn vec_apply_opt<F, R>(&self, f: F, rng: &mut R, size: usize, info: Arc<SInfo>) -> Vec<SolOpt>
    where
        F: Fn(SolOpt) -> SolOpt + Send + Sync,
        R: Rng;

    /// Generates multiple paired solutions and applies a transformation to each.
    ///
    /// This is similar to [`vec_sample_pair`](Searchspace::vec_sample_pair), but allows a
    /// user-provided closure to post-process each sampled pair before returning the vector.
    /// This is useful for adding custom metadata, mutating fields, or enforcing additional
    /// constraints on the generated [`SolutionShape`].
    ///
    /// # Parameters
    ///
    /// * `f` - Closure applied to each sampled paired solution
    /// * `rng` - Random number generator
    /// * `size` - Number of paired solutions to generate
    /// * `info` - Shared solution metadata
    ///
    /// # Returns
    ///
    /// Vector of paired solutions after applying `f` to each sample.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{FidelitySol, EmptyInfo, Searchspace, Pair, Sp, Mixed, Real, SId, HasFidelity};
    /// use std::sync::Arc;
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let pairs: Vec<Pair<FidelitySol<SId, Mixed, EmptyInfo>, FidelitySol<SId, Real, EmptyInfo>, SId, _, _, _>> = sp.vec_apply_pair(
    ///     |mut pair: Pair<_, _, _, Mixed, Real, EmptyInfo>| { pair.set_fidelity(1.0); pair },
    ///     &mut rng,
    ///     5,
    ///     info.clone(),
    /// );
    ///
    /// for pair in pairs.iter() {
    ///     assert_eq!(pair.fidelity().0, 1.0);
    /// }
    /// ```
    fn vec_apply_pair<F, R>(
        &self,
        f: F,
        rng: &mut R,
        size: usize,
        info: Arc<SInfo>,
    ) -> Vec<Self::SolShape>
    where
        F: Fn(Self::SolShape) -> Self::SolShape + Send + Sync,
        R: Rng;

    /// Validates that all solutions in a vector belong to the Obj domain.
    ///
    /// Batch version of [`is_in_obj`](Searchspace::is_in_obj) that returns `true` only if
    /// all solutions satisfy their Obj domain constraints.
    ///
    /// # Parameters
    ///
    /// * `inp` - Slice of solutions to validate
    ///
    /// # Returns
    ///
    /// `true` if all solutions are valid in the Obj domain, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{BaseSol, Sp, Mixed, Real, Searchspace, Solution, EmptyInfo, SId};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let vobj: Vec<BaseSol<SId, Mixed, EmptyInfo>> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_sample_obj(&sp, &mut rng, 10, info.clone());
    ///
    /// assert!(<Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_is_in_obj::<BaseSol<SId, Mixed, EmptyInfo>>(&sp, &vobj));
    ///
    /// ```
    fn vec_is_in_obj<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Obj, SInfo, Raw = RawObj<Self::SolShape, SolId, SInfo>>
            + Send
            + Sync;
    /// Validates that all solutions in a vector belong to the Opt domain.
    ///
    /// Batch version of [`is_in_opt`](Searchspace::is_in_opt) that returns `true` only if
    /// all solutions satisfy their Opt domain constraints.
    ///
    /// # Parameters
    ///
    /// * `inp` - Slice of solutions to validate
    ///
    /// # Returns
    ///
    /// `true` if all solutions are valid in the Opt domain, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// # mod sp{
    /// #        use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
    /// #        use tantale::macros::hpo;
    /// #
    /// #
    /// #        hpo!(
    /// #            a | Real(0.0,1.0, Uniform)                    |                         ;
    /// #            b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)  ;
    /// #            c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)  ;
    /// #            d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)  ;
    /// #        );
    /// #    }
    ///
    /// use tantale::core::{BaseSol, Sp, Real, Mixed, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let vopt: Vec<BaseSol<SId, Real, EmptyInfo>> = sp.vec_sample_opt(&mut rng, 10, info.clone());
    ///
    /// assert!(<Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_is_in_opt::<BaseSol<SId, Real, EmptyInfo>>(&sp, &vopt));
    ///
    /// ```
    fn vec_is_in_opt<S>(&self, inp: &[S]) -> bool
    where
        S: Solution<SolId, Self::Opt, SInfo, Raw = SolOpt::Raw> + Send + Sync;
}

pub mod spbase;
pub use spbase::Sp;

pub mod sppar;
pub use sppar::SpPar;
