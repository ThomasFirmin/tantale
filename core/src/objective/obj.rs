//! Objective wrappers for user-defined functions.
//!
//! An objective is a thin wrapper around the function the user wants to optimize.
//! The raw function must return an [`Outcome`], from which
//! a [`Codomain`](crate::domain::codomain::Codomain) is extracted.
//!
//! # Examples
//! ## Single-shot objective
//! Wrap the following code inside a module, to use effortlessly with Tantale.
//! The `objective!` macro creates various functions
//! generating the searchspace, and the objective itself.
//! ```
//!  use serde::{Deserialize, Serialize};
//!  use tantale::core::{Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
//!  use tantale::macros::{Outcome, objective};
//!
//!  #[derive(Outcome, Debug, Serialize, Deserialize)]
//!  pub struct OutExample {
//!      #[maximize]
//!      pub obj: f64,
//!      pub fid: f64,
//!      pub con: f64,
//!      pub more: f64,
//!      pub info: f64,
//!      pub intinfo: i64,
//!      pub boolinfo: bool,
//!      pub natinfo: u64,
//!      pub catinfo: String,
//!  }
//!
//!  fn plus_one_int(x: i64) -> i64 {
//!      x + 1
//!  }
//!  objective!(
//!      pub fn example<'a>() -> OutExample {
//!          let a = [! a | Real(0.0,5.0,Uniform) | !];
//!          let aa = [! aa_{10} | Real(-5.0,0.0,Uniform) | Int(0,100,Uniform) !];
//!          let aaa = [! aaa | Real(100.0,200.0,Uniform) | !];
//!          let some_bool = [! boolvar | Bool(Bernoulli(0.5)) | !];
//!          let some_nat = [! natvar | Nat(0,10,Uniform) | !];
//!          let some_cat = [! catvar | Cat(["relu", "tanh", "sigmoid"],Uniform) |!];
//!          let some_int = plus_one_int([! intvar | Int(-10,0,Uniform) | !]);
//!          OutExample{
//!              obj: a,
//!              fid: aa[0],
//!              con: aa[1],
//!              more: aa[2],
//!              info: aaa,
//!              intinfo: some_int,
//!              boolinfo: some_bool,
//!              natinfo: some_nat,
//!              catinfo: some_cat,
//!          }
//!      }
//!  );
//! ```
//!
//! ## Multi-steps objective
//! Wrap the following code inside a module, to use effortlessly with Tantale.
//! The `objective!` macro creates various functions
//! generating the searchspace, and the objective itself.
//! Pay attention to the returned [`FuncState`], and [`EvalStep`](crate::objective::EvalStep).
//! ```
//! use tantale::core::{FuncState,Step, Bool, Cat, Int, Nat, Real, Bernoulli, Uniform};
//! use tantale::macros::{Outcome, objective};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Outcome, Debug, Serialize, Deserialize)]
//! pub struct FidOutExample {
//!     #[maximize]
//!     pub obj: f64,
//!     pub fid: f64,
//!     pub con: f64,
//!     pub more: f64,
//!     pub info: f64,
//!     pub intinfo: i64,
//!     pub boolinfo: bool,
//!     pub natinfo: u64,
//!     pub catinfo: String,
//!     #[step]
//!     pub step: Step,
//! }
//!
//! #[derive(Serialize, Deserialize)]
//! pub struct FnState {
//!     pub state: isize,
//! }
//!
//! impl FuncState for FnState {
//!     fn save(&self, path: std::path::PathBuf) -> std::io::Result<()>{
//!         let mut file = std::fs::File::create(path.join("fn_state.mp"))?;
//!         rmp_serde::encode::write(&mut file, &self).unwrap();
//!         Ok(())
//!     }
//!     fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
//!         let file_path = path.join("fn_state.mp");
//!         let file = std::fs::File::open(file_path)?;
//!         let state = rmp_serde::decode::from_read(file).unwrap();
//!         Ok(state)
//!     }
//! }
//!
//! fn plus_one_int(x: i64) -> i64 {
//!     x + 1
//! }
//!
//! objective!(
//!     pub fn example() -> (FidOutExample,FnState) {
//!        let a = [! a | Real(0.0,5.0,Uniform) | !];
//!        let aa = [! aa_{10} | Real(-5.0,0.0,Uniform) | Int(0,100,Uniform) !];
//!        let aaa = [! aaa | Real(100.0,200.0,Uniform) | !];
//!        let some_bool = [! boolvar | Bool(Bernoulli(0.5)) | !];
//!        let some_nat = [! natvar | Nat(0,10,Uniform) | !];
//!        let some_cat = [! catvar | Cat(["relu", "tanh", "sigmoid"],Uniform) |!];
//!        let some_int = plus_one_int([! intvar | Int(-10,0,Uniform) | !]);
//!
//!        let mut state = match [! STATE !] {
//!             Some(s) => s,
//!             None => FnState { state: 0 },
//!         };
//!         state.state += 1;
//!         let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
//!         (
//!            FidOutExample{
//!                obj: a,
//!                fid: aa[0],
//!                con: aa[1],
//!                more: aa[2],
//!                info: aaa,
//!                intinfo: some_int,
//!                boolinfo: some_bool,
//!                natinfo: some_nat,
//!                catinfo: some_cat,
//!                step: evalstate,
//!            },
//!            state
//!         )
//!     }
//! );
//!
//! fn main() {
//!     
//! }
//! ```

use crate::{
    BaseSol, Domain, FidOutcome, FidelitySol, HasX, SolId, SolInfo, Solution, StepId, Uncomputed, objective::outcome::{FuncState, Outcome}, solution::partial::Fidelity,
};
use std::sync::Arc;

type OptimFn<Raw, Out> = fn(Raw) -> Out;
type SteppFn<Raw, Out, FnState> = fn(Raw, Fidelity, Option<FnState>) -> (Out, FnState);

/// Marker trait for wrappers around user-defined objective functions.
///
/// Both [`Objective`] (single-shot evaluation) and [`Stepped`] (multi-[`Step`](crate::objective::Step))
/// implement this trait so they can be used interchangeably by higher-level
/// components.
///
/// # Associated Derive Macro
///
/// The `FuncWrapper` derive macro automatically implements the trait for any struct
/// satisfying the required trait bounds.
pub trait FuncWrapper
{
    type Context;
    type Out: Outcome;
    type OutShape;
    type Dom: Domain;
    type Sol<Dom:Domain>: Uncomputed<Dom=Dom>; 
    /// Evaluate the objective at a given `x` [`Raw`](crate::Solution::Raw) with a given `context`.
    fn compute(&self, x: <Self::Sol::<Self::Dom> as HasX>::X , context: Self::Context) -> Self::OutShape;
}

/// Minimal wrapper for a single-shot objective function.
///
/// The wrapped function maps a [`Raw`](crate::Solution::Raw) solution `x` to an [`Outcome`]. The [`Raw`](crate::Solution::Raw)
/// input is typically the searchspace solution representation.
///
/// # Type parameters
/// - `S`: [`Solution`] type, which provides the [`Raw`](HasX::Raw) input to the objective function. 
/// - `Out`: Output implementing [`Outcome`].
pub struct OneStep<S:Solution, Out: Outcome>(pub OptimFn<S::X, Out>);

impl<S:Solution, Out: Outcome> OneStep<S, Out>
{
    /// Creates a new [`Objective`] from a raw function.
    ///
    /// # Parameters
    /// - `func`: The objective function to optimize.
    ///
    /// # See also
    /// - [`Searchspace`](crate::searchspace::Searchspace)
    /// - `objective!`
    pub fn new(func: OptimFn<S::X, Out>) -> Self {
        OneStep(func)
    }
}

impl<S:Solution, Out: Outcome> Clone for OneStep<S, Out>
{
    fn clone(&self) -> Self {
        OneStep(self.0)
    }
}

impl<Id, Obj, Info, Out> FuncWrapper for OneStep<BaseSol<Id, Obj, Info>, Out>
where
    Id: SolId,
    Obj: Domain,
    Info: SolInfo,
    Out: Outcome,
{
    type Context = ();
    type OutShape = Out;
    type Out = Out;
    type Dom = Obj;
    type Sol<Dom:Domain> = BaseSol<Id, Dom, Info>;

    fn compute(&self, x: Arc<[Obj::Element]>, _context: Self::Context) -> Self::OutShape {
        (self.0)(x)
    }
}

/// Wrapper for objectives evaluated by [`Step`](crate::objective::Step).
///
/// The wrapped function receives an input `x`, a [`Fidelity`], and an optional
/// [`FuncState`] from previous [`Step`](crate::objective::Step)s. It returns the [`Outcome`] and the updated
/// [`FuncState`].
///
/// # Type parameters
/// - `Raw`: [`Raw`](crate::Solution::Raw) input to the objective function.
/// - `Out`: Output implementing [`FidOutcome`].
/// - `FnState`: Internal function state implementing [`FuncState`].
pub struct MultiStep<S, Out, FnState>(pub SteppFn<S::X, Out, FnState>)
where
    S: Solution,
    Out: FidOutcome,
    FnState: FuncState;

impl<S, Out, FnState> Clone for MultiStep<S, Out, FnState>
where
    S: Solution,
    Out: FidOutcome,
    FnState: FuncState,
{
    fn clone(&self) -> Self {
        MultiStep(self.0)
    }
}

impl<S, Out, FnState> MultiStep<S, Out, FnState>
where
    S: Solution,
    Out: FidOutcome,
    FnState: FuncState,
{
    /// Creates a new [`Stepped`] from a raw multi-[`Step`](crate::objective::Step) function.
    ///
    /// # Parameters
    /// - `func`: The stepped objective function to optimize.
    ///
    /// # See also
    /// - [`Searchspace`](crate::searchspace::Searchspace)
    /// - `objective!` procedural macro
    pub fn new(func: SteppFn<S::X, Out, FnState>) -> Self {
        MultiStep(func)
    }
}

impl<Id, Obj, Info, Out, FnState> FuncWrapper for MultiStep<FidelitySol<Id, Obj, Info>, Out, FnState>
where
    Id: StepId,
    Obj: Domain,
    Info: SolInfo,
    Out: FidOutcome,
    FnState: FuncState,
{
    type Context = (Fidelity, Option<FnState>);
    type OutShape = (Out, FnState);
    type Out = Out;
    type Dom = Obj;
    type Sol<Dom:Domain> = FidelitySol<Id, Dom, Info>;

    fn compute(&self, x: Arc<[Obj::Element]>, context: Self::Context) -> Self::OutShape {
        (self.0)(x, context.0, context.1)
    }
}
