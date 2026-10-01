//! # Searchspace
//!
//! This module provides the [`Searchspace`] trait and related types for defining the optimization
//! search domain. A searchspace specifies the valid variable values that an optimizer can explore
//! and an objective function can evaluate.
//!
//! ## Overview
//!
//! A [`Searchspace`] is typed by :
//! - An **Obj domain**: The domain for the objective function's input
//! - An **Opt domain**: The domain for the optimizer's representation
//!   `Obj` and `Opt` types are related via [`Onto`](crate::Onto)
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
//! See actual examples within the repository for more detailed use cases.
//!
//! ```rust
//! mod searchspace{
//!     use tantale::core::{Bool, Cat, Nat, Real, Searchspace, Uniform, Bernoulli};
//!     use tantale::macros::{hpo, Outcome};
//!     use std::sync::Arc;
//!     use serde::{Serialize,Deserialize};
//!
//!     
//!     hpo!(
//!         a | Real(1e-4, 1.0,Uniform)       | Log                   ;
//!         b | Nat(0,100,Uniform)            | Real(0.0,1.0,Uniform) ;
//!         c | Cat(["a", "b", "c"], Uniform) | Real(0.0,1.0,Uniform) ;
//!         d | Bool(Bernoulli(0.5))          | Real(0.0,1.0,Uniform) ;
//!     );
//!
//!     #[derive(Outcome, Debug, Serialize, Deserialize)]
//!     pub struct OutStruct{#[maximize] pub out:f64}
//!
//!     // _TantaleMixedObj is automatically created by hpo!
//!     pub fn example(tantale_in : Arc::<[<ObjType as Domain>::TypeDom]>) -> OutStruct{
//!         let a = match tantale_in[0]{
//!             MixedTypeDom::Real(value) => value,
//!             _ => panic!("Unexpected type for a"),
//!         };
//!         let b = match tantale_in[1]{
//!             MixedTypeDom::Nat(value) => value,
//!             _ => panic!("Unexpected type for b"),
//!         };
//!         let c = match tantale_in[2]{
//!             MixedTypeDom::Cat(ref value) => value,
//!             _ => panic!("Unexpected type for c"),
//!         };
//!         let d = match tantale_in[3]{
//!             MixedTypeDom::Bool(value) => value,
//!             _ => panic!("Unexpected type for d"),
//!         };
//!         println!("a {}, b {}, c {}, d {}", a, b, c, d);
//!
//!         OutStruct{out:42.0}
//!     }
//! }
//!
//! use tantale::core::{HasX, Sp, BaseSol, EmptyInfo,Searchspace,Solution,SId, HasId};
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
//!     pub struct OutStruct{#[maximize] pub out:f64}
//!
//!     objective!(
//!         pub fn example() -> OutStruct {
//!             let a = [! a | Real(1e-4, 1.0,Uniform)  | Log                   !];
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
//! use tantale::core::{HasX, Sp, BaseSol, EmptyInfo,Searchspace,Solution,SId, HasId};
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
    FuncWrapper, Uncomputed, aliases::{GetId, GetRaw, GetSInfo, GetSObj, GetSOpt}, domain::onto::Linked, solution::{
        SolId, SolutionShape, shape::IntoComputedShape,
    },
};

use rand::prelude::Rng;
use std::sync::Arc;

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
/// - [`contains_obj`](Searchspace::contains_obj) - Check if solution is valid in Obj domain
/// - [`contains_opt`](Searchspace::contains_opt) - Check if solution is valid in Opt domain
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
/// let outcome = objective_fn(obj_solution.ref_x());
/// ```
///
/// # See Also
///
/// - [`Sp`] - Concrete searchspace implementation
/// - [`Var`](crate::Var) - Individual variable definitions
/// - [`Linked`] - Trait linking Obj and Opt domains
/// - [`hpo!`](../../../tantale/macros/macro.hpo.html) - Macro for creating searchspaces
/// - [`objective!`](../../../tantale/macros/macro.objective.html) - Macro for defining searchspaces and wrap objective functions
pub trait Searchspace<F: FuncWrapper<Dom=Self::Obj>>: Linked
{
    /// The paired solution shape containing both Obj and Opt [`twin`](crate::Solution::twin) [`Solution`](crate::Solution) representations.
    type Shape: SolutionShape<
        ObjSol = F::Sol<Self::Obj>,
        OptSol = F::Sol<Self::Opt>, 
        Obj=Self::Obj, 
        Opt=Self::Opt,
    > + IntoComputedShape;

    /// Create a new solution from a raw solution in the Obj domain.
    fn new_obj<T: Into<GetRaw<F::Sol<Self::Obj>>>>(&self, x: T, info: Arc<GetSInfo<F::Sol<Self::Obj>>>) -> Self::Shape {
        let id = GetId::<F::Sol<Self::Obj>>::generate();
        let sol = F::Sol::<Self::Obj>::new(id, x, info);
        self.onto_opt(sol)
    }

    /// Create a new solution from a raw solution in the Opt domain.
    fn new_opt<T: Into<GetRaw<F::Sol<Self::Opt>>>>(&self, x: T, info: Arc<GetSInfo<F::Sol<Self::Opt>>>) -> Self::Shape {
        let id = GetId::<F::Sol<Self::Opt>>::generate();
        let sol = F::Sol::<Self::Opt>::new(id, x, info);
        self.onto_obj(sol)
    }

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
    /// use tantale::core::{HasX, Sp, BaseSol, Bounded, Mixed, Real, Pair, Solution, EmptyInfo, SId, Searchspace, SolutionShape};
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
    /// for (i, o) in opt.get_sobj().ref_x().iter().zip(opt.get_sopt().ref_x().iter()) {
    ///     println!("Obj: {:?} => Opt: {}", i, o);
    /// }
    /// ```
    fn onto_opt(&self, inp: GetSObj<Self::Shape>) -> Self::Shape;
    /// Maps a solution from the Opt domain to the Obj domain.
    ///
    /// # Parameters
    ///
    /// * `inp` - Solution in the Opt domain
    ///
    /// # Example
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
    /// use tantale::core::{HasX, Sp, BaseSol, Bounded, Mixed, Real, Solution, EmptyInfo, SId, Searchspace, SolutionShape};
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
    /// for (i, o) in obj.get_sopt().ref_x().iter().zip(obj.get_sobj().ref_x().iter()) {
    ///     println!("Opt: {} => Obj: {:?}", i, o);
    /// }
    /// ```
    fn onto_obj(&self, inp: GetSOpt<Self::Shape>) -> Self::Shape;
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
    /// use tantale::core::{HasX, Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let obj: BaseSol<SId, Mixed, EmptyInfo> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::sample_obj(&sp, &mut rng, info.clone());
    ///
    /// for i in obj.ref_x().iter(){
    ///     println!("{:?}", i);
    /// }
    /// ```
    fn sample_obj<R: Rng>(&self, rng: &mut R, info: Arc<GetSInfo<GetSObj<Self::Shape>>>) -> GetSObj<Self::Shape>{
        Uncomputed::new(GetId::<GetSObj<Self::Shape>>::generate(), self.sample_raw_obj(rng), info)
    }
    /// Generates a [`Raw`](crate::Solution::Raw) solution in the Obj (objective function) domain.
    ///
    /// Samples each variable using its Obj domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `info` - Shared solution metadata
    ///
    /// # Example
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
    /// use tantale::core::{HasX, Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    ///
    /// let obj: Arc<[Mixed]> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::sample_raw_obj(&sp, &mut rng);
    ///
    /// for i in obj.iter(){
    ///     println!("{:?}", i);
    /// }
    /// ```
    fn sample_raw_obj<R: Rng>(&self, rng: &mut R) -> GetRaw<GetSObj<Self::Shape>>;
    /// Generates a random solution in the Opt (optimizer) domain.
    ///
    /// Samples each variable using its Opt domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    ///
    /// # Example
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
    /// use tantale::core::{HasX, Sp, BaseSol, Bounded, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    ///
    /// let opt: BaseSol<SId, Bounded<f64>, EmptyInfo> = sp.sample_opt(&mut rng, info.clone());
    ///
    /// for i in opt.ref_x().iter(){
    ///     println!("{}", i);
    /// }
    /// ```
    fn sample_opt<R: Rng>(&self, rng: &mut R, info: Arc<GetSInfo<GetSOpt<Self::Shape>>>) -> GetSOpt<Self::Shape>{
        Uncomputed::new(GetId::<GetSOpt<Self::Shape>>::generate(), self.sample_raw_opt(rng), info)
    }
    /// Generates a [`Raw`](crate::Solution::Raw) solution in the Opt (objective function) domain.
    ///
    /// Samples each variable using its Opt domain sampler.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `info` - Shared solution metadata
    ///
    /// # Example
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
    /// use tantale::core::{HasX, Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
    /// use std::sync::Arc;
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    ///
    /// let opt: Arc<[Real]> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::sample_raw_opt(&sp, &mut rng);
    ///
    /// for i in opt.iter(){
    ///     println!("{:?}", i);
    /// }
    /// ```
    fn sample_raw_opt<R: Rng>(&self, rng: &mut R) -> GetRaw<GetSOpt<Self::Shape>>;
    /// Generates a random paired solution containing both Obj and Opt representations.
    ///
    /// Samples in the Obj domain and automatically creates the paired Opt representation
    /// via mapping. Both representations share the same [`Id`].
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    ///
    /// # Example
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
    /// use tantale::core::{HasX,Sp, BaseSol, Real, Mixed, Pair, Solution, EmptyInfo, SId, Searchspace, SolutionShape, HasId};
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
    /// println!("Obj: {:?}", pair.get_sobj().ref_x());
    /// println!("Opt: {:?}", pair.get_sopt().ref_x());
    /// ```
    ///
    fn sample_pair<R: Rng>(&self, rng: &mut R, info: Arc<GetSInfo<Self::Shape>>) -> Self::Shape;
    /// Validates that a solution belongs to the Obj domain of this searchspace.
    ///
    /// Checks each variable value against its Obj domain constraints using
    /// [`Domain::contains`](crate::Domain::contains).
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
    /// assert!(<Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::contains_obj::<BaseSol<SId, Mixed, EmptyInfo>>(&sp, &obj));
    /// ```
    fn contains_obj(&self, inp: &GetRaw<GetSObj<Self::Shape>>) -> bool;

    /// Validates that a solution belongs to the Opt domain of this searchspace.
    ///
    /// Checks each variable value against its Opt domain constraints using
    /// [`Domain::contains`](crate::Domain::contains).
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
    /// assert!(<Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::contains_opt::<BaseSol<SId, Real, EmptyInfo>>(&sp, &opt));
    /// ```
    fn contains_opt(&self, inp: &GetRaw<GetSOpt<Self::Shape>>) -> bool;
    /// Maps a [`Solution`](crate::Solution) of type `Opt` onto an [`Solution`](crate::Solution) of type `Obj`.
    ///
    /// # Example
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
    /// use tantale::core::{HasX, Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace, Pair, SolutionShape};
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
    ///     for (i, o) in pair.get_sopt().ref_x().iter().zip(pair.get_sobj().ref_x().iter()){
    ///         println!("Opt: {} => Obj: {:?}", i, o);
    ///     }
    /// }
    /// ```
    fn vec_onto_obj(&self, inp: Vec<GetSOpt<Self::Shape>>) -> Vec<Self::Shape>;
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
    /// use tantale::core::{HasX, Sp, BaseSol, Real, Mixed, Solution, EmptyInfo, SId, Searchspace, Pair, SolutionShape};
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
    ///     for (i, o) in pair.get_sobj().ref_x().iter().zip(pair.get_sopt().ref_x().iter()){
    ///         println!("Obj: {:?} => Opt: {}", i, o);
    ///     }
    /// }
    /// ```
    fn vec_onto_opt(&self, inp: Vec<GetSObj<Self::Shape>>) -> Vec<Self::Shape>;
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
    /// use tantale::core::{HasX, Sp, BaseSol, Real, Mixed, Solution, EmptyInfo, SId, Searchspace};
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
    ///     println!("Obj: {:?}", obj.ref_x());
    /// }
    /// ```
    ///
    fn vec_sample_obj<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<GetSInfo<Self::Shape>>,
    ) -> Vec<GetSObj<Self::Shape>>;
    /// Generates multiple random raw solutions in the Obj domain.
    ///
    /// Batch version of [`sample_raw_obj`](Searchspace::sample_raw_obj) for efficient generation
    /// of multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of solutions to generate
    ///
    /// # Returns
    ///
    /// Vector of randomly sampled solutions in the Obj domain, each with a unique [`Id`].
    ///
    /// # Example
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
    /// use tantale::core::{HasX, Sp, BaseSol, Real, Mixed, Solution, EmptyInfo, SId, Searchspace};
    /// use std::{fmt::Debug, sync::Arc};
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    ///
    /// let vec_obj: Vec<Arc<[Mixed]>> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_sample_raw_obj(&sp, &mut rng, 10);
    ///
    /// for obj in vec_obj.into_iter(){
    ///     println!("Obj: {:?}", obj);
    /// }
    /// ```
    ///
    fn vec_sample_raw_obj<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<GetSObj<Self::Shape>>>;
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
    /// use tantale::core::{HasX, Sp, BaseSol, Mixed, Real, Solution, EmptyInfo, SId, Searchspace};
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
    ///     println!("Opt: {:?}", opt.ref_x());
    /// }
    /// ```
    fn vec_sample_opt<R: Rng>(&self, rng: &mut R, size: usize, info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSOpt<Self::Shape>>;
    /// Generates multiple random raw solutions in the Opt domain.
    ///
    /// Batch version of [`sample_raw_obj`](Searchspace::sample_raw_obj) for efficient generation
    /// of multiple solutions.
    ///
    /// # Parameters
    ///
    /// * `rng` - Random number generator
    /// * `size` - Number of solutions to generate
    ///
    /// # Returns
    ///
    /// Vector of randomly sampled solutions in the Opt domain, each with a unique [`Id`].
    ///
    /// # Example
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
    /// use tantale::core::{HasX, Sp, BaseSol, Real, Mixed, Solution, EmptyInfo, SId, Searchspace};
    /// use std::{fmt::Debug, sync::Arc};
    ///
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    ///
    /// let vec_opt: Vec<Arc<[Real]>> = <Sp<Mixed, Real> as Searchspace<BaseSol<SId, Real, EmptyInfo>, SId, EmptyInfo>>::vec_sample_raw_opt(&sp, &mut rng);
    ///
    /// for opt in vec_opt.into_iter(){
    ///     println!("Opt: {:?}", opt);
    /// }
    /// ```
    ///
    fn vec_sample_raw_opt<R: Rng>(&self, rng: &mut R, size: usize) -> Vec<GetRaw<GetSOpt<Self::Shape>>>;
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
    /// use tantale::core::{HasX, Solution, Sp, BaseSol, Real, Mixed, Pair, EmptyInfo, HasId, SId, Searchspace, SolutionShape};
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
    ///     println!("Obj: {:?}", pair.get_sobj().ref_x());
    ///     println!("Opt: {:?}", pair.get_sopt().ref_x());
    /// }
    /// ```
    ///
    fn vec_sample_pair<R: Rng>(
        &self,
        rng: &mut R,
        size: usize,
        info: Arc<GetSInfo<Self::Shape>>,
    ) -> Vec<Self::Shape>;

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
    /// use tantale::core::{Sp, Mixed, Real, EmptyInfo, Searchspace, StepSId, FidelitySol, HasFidelity};
    /// use std::sync::Arc;
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sols: Vec<FidelitySol<StepSId, Mixed, EmptyInfo>> = <Sp<Mixed, Real> as Searchspace<FidelitySol<StepSId, Real, EmptyInfo>, StepSId, EmptyInfo>>::vec_apply_obj(
    ///     &sp,
    ///     |mut sol: FidelitySol<StepSId, Mixed, EmptyInfo>| { sol.set_fidelity(1.0); sol },
    ///     &mut rng,
    ///     5,
    ///     info.clone(),
    /// );
    ///
    /// for sol in sols.iter() {
    ///     assert_eq!(sol.fidelity().0, 1.0);
    /// }
    /// ```
    fn vec_apply_obj<A, R>(
        &self,
        f: A,
        rng: &mut R,
        size: usize,
        info: Arc<GetSInfo<Self::Shape>>,
    ) -> Vec<GetSObj<Self::Shape>>
    where
        A: Fn(GetSObj<Self::Shape>) -> GetSObj<Self::Shape> + Send + Sync,
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
    /// use tantale::core::{Sp, Mixed, Real, FidelitySol, EmptyInfo, Searchspace, StepSId, HasFidelity};
    /// use std::sync::Arc;
    ///
    /// let sp: Sp<Mixed, Real> = sp::get_searchspace();
    /// let info = Arc::new(EmptyInfo);
    /// let mut rng: rand::rngs::ThreadRng = rand::rng();
    ///
    /// let sols: Vec<FidelitySol<StepSId, Real, EmptyInfo>> = <Sp<Mixed, Real> as Searchspace<FidelitySol<StepSId, Real, EmptyInfo>, StepSId, EmptyInfo>>::vec_apply_opt(
    ///     &sp,
    ///     |mut sol: FidelitySol<StepSId, Real, EmptyInfo>| { sol.set_fidelity(1.0); sol },
    ///     &mut rng,
    ///     5,
    ///     info.clone(),
    /// );
    ///
    /// for sol in sols.iter() {
    ///     assert_eq!(sol.fidelity().0, 1.0);
    /// }
    /// ```
    fn vec_apply_opt<A, R>(&self, f: A, rng: &mut R, size: usize, info: Arc<GetSInfo<Self::Shape>>) -> Vec<GetSOpt<Self::Shape>>
    where
        A: Fn(GetSOpt<Self::Shape>) -> GetSOpt<Self::Shape> + Send + Sync,
        R: Rng;
}

pub mod spbase;
pub use spbase::Sp;

pub mod sppar;
pub use sppar::SpPar;
