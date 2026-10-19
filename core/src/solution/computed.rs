//! Computed (evaluated) solutions.
//!
//! A [`Computed`] solution pairs an [`Uncomputed`] solution with the
//! corresponding codomain value (a [`TypeCodom`]) produced by evaluating the raw
//! solution with an [`Objective`](crate::Objective). This is the evaluated form used by
//! optimizers and recorders.
//!
//! # Note
//!
//! For [`TypeCodom`] that are [`Ord`] the
//! [`Computed`] solution implements [`Ord`] and can be used in sorted collections.
//! Same applies for [`PartialOrd`], [`PartialEq`] and [`Eq`].
//! For [`TypeCodom`] that are [`Dominate`] the
//! [`Computed`] solution implements [`Dominate`].
//!
//! # Examples
//! ```
//! use tantale::core::{HasX, Solution, BaseSol, Outcome, Computed, Codomain, EmptyInfo, Real, Id, SId, Uncomputed};
//! use tantale::macros::Outcome;
//! use std::sync::Arc;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Outcome, Serialize, Deserialize, Debug)]
//! struct Out { #[maximize] value: f64 }
//!
//! let cod = Out::codomain();
//! let x = Arc::from(vec![0.1, 0.2]);
//! let info = Arc::new(EmptyInfo {});
//! let sol = BaseSol::<SId, Real, _>::new(SId::generate(), x, info);
//! let y = Arc::new(cod.get_elem(&Out { value: 0.5 }));
//!
//! let computed: Computed<_, SId, Real, Out, _> = Computed::new(sol, y);
//! assert_eq!(computed.ref_x().len(), 2);
//! ```
//!
//! ```
//! use tantale::core::{HasX, Solution, BaseSol, Outcome, Codomain, EmptyInfo, IntoComputed, Real, Id, SId, Uncomputed};
//! use tantale::macros::Outcome;
//! use std::sync::Arc;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Outcome, Serialize, Deserialize, Debug)]
//! struct Out { #[maximize] value: f64 }
//!
//! let cod = Out::codomain();
//! let x = Arc::from(vec![0.4, 0.8]);
//! let info = Arc::new(EmptyInfo {});
//! let sol = BaseSol::<SId, Real, _>::new(SId::generate(), x, info);
//! let y = Arc::new(cod.get_elem(&Out { value: 1.2 }));
//!
//! let computed = sol.into_computed::<Out>(y);
//! assert_eq!(computed.ref_x().len(), 2);
//! ```

use crate::{
    Dominate, EvalStep, Fidelity, HasFidelity, HasId, HasInfo, HasStep, HasStepId, HasUncomputed, HasX, HasY, StepId, Xy, aliases::TypeCodom, domain::codomain::ElemMulti, 
    objective::{Outcome, Step}, solution::{Solution, Uncomputed}, utils::orderable::Orderable,
};

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::{fmt::Debug, sync::Arc};

/// A solution of the [`Objective`](crate::Objective) or the [`Optimizer`](crate::Optimizer)
/// [`Domain`], enriched with the computed [`Codomain`](crate::Codomain) value [`TypeCodom`].
///
/// # Attributes
/// * `sol` : [`Uncomputed`] - An already created partial solution.
/// * `y` : [`Arc<TypeCodom<Out>>`](TypeCodom) - The computed codomain value.
///
/// # Note
///
/// A [`Computed`] is obtained by evaluating the raw solution from an [`Uncomputed`] with the
/// [`Objective`](crate::Objective) and then extracting a [`TypeCodom`].
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound = "S: Uncomputed, Out: Outcome")]
pub struct Computed<S: Uncomputed, Out: Outcome>
{
    pub sol: S,
    pub y: Arc<TypeCodom<Out>>,
}

impl<S: Uncomputed, Out: Outcome> Clone for Computed<S, Out>
{
    fn clone(&self) -> Self {
        Self {
            sol: self.sol.clone(),
            y: self.y.clone(),
        }
    }
}

impl<S: Uncomputed, Out: Outcome> HasX for Computed<S, Out>
{
    type X = S::X;
    fn ref_x(&self) -> &S::X {
        self.sol.ref_x()
    }

    fn clone_x(&self) -> S::X {
        self.sol.clone_x()
    }
    
    fn set_x(&mut self, x: Self::X) {
        self.sol.set_x(x);
    }
    
}

impl<S: Uncomputed, Out: Outcome> HasId for Computed<S, Out>
{
    type Id = S::Id;

    /// Return the identifier of the underlying uncomputed solution.
    fn id(&self) -> S::Id {
        self.sol.id()
    }

    fn ref_id(&self) -> &S::Id {
        self.sol.ref_id()
    }

    fn mut_id(&mut self) -> &mut S::Id {
        self.sol.mut_id()
    }
}

impl<S: Uncomputed, Out: Outcome> HasStepId for Computed<S, Out>
where
    Self::Id: StepId,
{
    fn increment(&mut self) {
        self.sol.mut_id().increment()
    }

    fn id_step(&self) -> usize {
        self.sol.ref_id().id_step()
    }

    fn previous_id(&self) -> Self::Id {
        self.sol.ref_id().previous_id()
    }
}

impl<S: Uncomputed, Out: Outcome> HasInfo for Computed<S, Out>
{
    type Info = S::Info;
    /// Return the [`SolInfo`] associated with the underlying solution.
    fn sinfo(&self) -> Arc<S::Info> {
        self.sol.sinfo()
    }
    
    fn set_info(&mut self, info: Arc<Self::Info>) {
        self.sol.set_info(info);
    }
}

impl<S: Uncomputed + HasStep, Out: Outcome> HasStep for Computed<S, Out>
{
    /// Return the current evaluation [`Step`] if the underlying solution supports it.
    fn step(&self) -> Step {
        self.sol.step()
    }

    fn raw_step(&self) -> crate::EvalStep {
        self.sol.raw_step()
    }

    fn pending(&mut self) {
        self.sol.pending();
    }

    fn partially(&mut self, value: isize) {
        self.sol.partially(value);
    }

    fn discard(&mut self) {
        self.sol.discard();
    }

    fn evaluated(&mut self) {
        self.sol.evaluated();
    }

    fn error(&mut self) {
        self.sol.error();
    }

    fn set_raw_step(&mut self, value: EvalStep) {
        self.sol.set_raw_step(value);
    }
}

impl<S: Uncomputed + HasFidelity, Out: Outcome> HasFidelity for Computed<S, Out>
{
    /// Return the [`Fidelity`] of the underlying solution.
    fn fidelity(&self) -> Fidelity {
        self.sol.fidelity()
    }

    /// Update the [`Fidelity`] of the underlying solution.
    fn set_fidelity(&mut self, fidelity: f64) {
        self.sol.set_fidelity(fidelity);
    }
}

impl<S: Uncomputed, Out: Outcome> HasY for Computed<S, Out>
{
    type Y = TypeCodom<Out>;
    /// Returns the computed [`TypeCodom`](crate::Codomain::TypeCodom) for this solution.
    fn y(&self) -> Arc<TypeCodom<Out>> {
        self.y.clone()
    }
    
    fn set_y(&mut self, y: Arc<Self::Y>) {
        self.y = y;
    }
}

impl<S: Uncomputed, Out: Outcome> HasUncomputed for Computed<S, Out>
{
    type Uncomputed = S;
    /// Returns the underlying [`Uncomputed`] solution.
    fn get_uncomputed(&self) -> &Self::Uncomputed {
        &self.sol
    }
    
    fn set_uncomputed(&mut self, uncomputed: Self::Uncomputed) {
        self.sol = uncomputed;
    }
}

impl<S: Uncomputed, Out: Outcome> Solution for Computed<S, Out>
{
    type Dom = S::Dom;
}

impl<S: Uncomputed, Out: Outcome> PartialEq for Computed<S, Out>
where
    TypeCodom<Out>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.y == other.y
    }
}

impl<S: Uncomputed, Out: Outcome> Eq for Computed<S, Out>
where
    TypeCodom<Out>: Eq,
{
}

impl<S: Uncomputed, Out: Outcome> PartialOrd for Computed<S, Out>
where
    TypeCodom<Out>: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y.partial_cmp(&other.y())
    }
}

impl<S: Uncomputed, Out: Outcome> Ord for Computed<S, Out>
where
    TypeCodom<Out>: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.y.cmp(&other.y())
    }
}

impl<S: Uncomputed, Out: Outcome> Orderable for Computed<S, Out>
where
    TypeCodom<Out>: Orderable,
{
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y.ord_cmp(&other.y())
    }
}

impl<S: Uncomputed, Out: Outcome> Dominate for Computed<S, Out>
where
    TypeCodom<Out>: ElemMulti,
{
    fn dominates(&self, other: &Self) -> bool {
        self.y.dominates(&other.y())
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self.y.get_objective_by_index(idx)
    }

    fn len_objectives(&self) -> usize {
        self.y.len_objectives()
    }

    fn get_objectives(&self) -> &[f64] {
        self.y.get_objectives()
    }
}

impl<S: Uncomputed, Out: Outcome> Computed<S, Out>
{
    /// Creates a new [`Computed`] from a [`BaseSol`](crate::solution::BaseSol) and a [`TypeCodom`](crate::Codomain::TypeCodom).
    pub fn new(sol: S, y: Arc<TypeCodom<Out>>) -> Self {
        Computed {
            sol,
            y,
        }
    }

    /// Creates a vec of [`Computed`] from an iterator of [`BaseSol`](crate::solution::BaseSol)s,
    /// and an iterator of [`Arc<`TypeCodom`>](crate::Codomain::TypeCodom).
    pub fn new_vec<I, J>(sol: I, y: J) -> Vec<Self>
    where
        I: IntoIterator<Item = S>,
        J: IntoIterator<Item = Arc<TypeCodom<Out>>>,
    {
        sol.into_iter()
            .zip(y)
            .map(|(s, cod)| Self::new(s, cod))
            .collect()
    }

    /// Returns the x and y values of the computed solution.
    pub fn xy(&self) -> Xy<S::X, TypeCodom<Out>> {
        Xy {
            x: self.clone_x(),
            y: self.y(),
        }
    }

    /// Extracts the uncomputed solution and objective value from a computed solution.
    ///
    /// This is the inverse operation of [`into_computed`](IntoComputed::into_computed),
    /// decomposing a computed solution back into its constituent parts.
    ///
    /// # Type Parameters
    ///
    /// * `Cod` - The [`crate::Codomain`] of the objective value
    /// * `Out` - The [`Outcome`] type
    ///
    /// # Parameters
    ///
    /// * `comp` - The computed solution to decompose
    ///
    /// # Returns
    ///
    /// A tuple of `(uncomputed_solution, objective_value)`.
    pub fn extract(self) -> (S, Arc<TypeCodom<Out>>){
        (self.sol, self.y)
    }
}
