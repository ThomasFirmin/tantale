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
    HasX, Dominate, EvalStep, Fidelity, HasFidelity, HasId, HasSolInfo, HasStep, HasStepId, HasUncomputed, HasY, Multi, StepId, Xy, 
    domain::{Domain, codomain::TypeCodom}, 
    objective::{Outcome, Step}, 
    solution::{Id, IntoComputed, SolInfo, Solution, Uncomputed}, 
    utils::orderable::Orderable
};

use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, marker::PhantomData};
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
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize, TypeCodom<Out>: Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>, TypeCodom<Out>: for<'a> Deserialize<'a>",
))]
pub struct Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    pub sol: PSol,
    pub y: Arc<TypeCodom<Out>>,
    _id: PhantomData<SolId>,
    _dom: PhantomData<Dom>,
    _info: PhantomData<Info>,
}

impl<PSol, SolId, Dom, Out, Info> HasX<PSol::Raw> for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    fn ref_x(&self) -> &PSol::Raw {
        self.sol.ref_x()
    }

    fn clone_x(&self) -> PSol::Raw {
        self.sol.clone_x()
    }
}

impl<PSol, SolId, Dom, Out, Info> HasId<SolId> for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    /// Return the identifier of the underlying uncomputed solution.
    fn id(&self) -> SolId {
        self.sol.id()
    }

    fn ref_id(&self) -> &SolId {
        self.sol.ref_id()
    }

    fn mut_ref_id(&mut self) -> &mut SolId {
        self.sol.mut_ref_id()
    }
}

impl<PSol, SolId, Dom, Out, Info> HasStepId<SolId> for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + HasStepId<SolId> + HasStep,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: StepId,
{
    fn increment(&mut self) {
        self.sol.increment()
    }

    fn id_step(&self) -> usize {
        self.sol.id_step()
    }

    fn previous_id(&self) -> SolId {
        self.sol.previous_id()
    }
}

impl<PSol, SolId, Dom, Out, Info> HasSolInfo<Info> for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    /// Return the [`SolInfo`] associated with the underlying solution.
    fn sinfo(&self) -> Arc<Info> {
        self.sol.sinfo()
    }
}

impl<PSol, SolId, Dom, Out, Info> HasStep for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + HasStep,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
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

impl<PSol, SolId, Dom, Out, Info> HasFidelity for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + HasFidelity,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
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

impl<PSol, SolId, Dom, Out, Info> HasY<Out> for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + IntoComputed,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    /// Returns the computed [`TypeCodom`](crate::Codomain::TypeCodom) for this solution.
    fn y(&self) -> Arc<TypeCodom<Out>> {
        self.y.clone()
    }
}

impl<PSol, SolId, Dom, Out, Info> HasUncomputed<SolId, Dom, Info>
    for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + IntoComputed,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    type Uncomputed = PSol;
    /// Returns the underlying [`Uncomputed`] solution.
    fn get_uncomputed(&self) -> &Self::Uncomputed {
        &self.sol
    }
}

impl<PSol, SolId, Dom, Out, Info> Solution<SolId, Dom, Info>
    for Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    SolId: Id,
    Dom: Domain,
    Out: Outcome,
    Info: SolInfo,
{
    type Raw = PSol::Raw;
    type Twin<B: Domain> = Computed<PSol::TwinUC<B>, SolId, B, Out, Info>;

    fn _clone_sol(&self) -> Self {
        Self::new(self.sol._clone_sol(), self.y.clone())
    }

    fn twin<B: Domain>(
        &self,
        x: <Self::Twin<B> as Solution<SolId, B, Info>>::Raw,
    ) -> Self::Twin<B> {
        Computed::new(Uncomputed::twin(&self.sol, x), self.y.clone())
    }
}

impl<PSol, SolId, Dom, SInfo, Out> PartialEq for Computed<PSol, SolId, Dom, Out, SInfo>
where
    Self: HasY<Out>,
    TypeCodom<Out>: PartialEq,
    Out: Outcome,
    PSol: Uncomputed<SolId, Dom, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
{
    fn eq(&self, other: &Self) -> bool {
        self.y == other.y
    }
}

impl<PSol, SolId, Dom, SInfo, Out> Eq for Computed<PSol, SolId, Dom, Out, SInfo>
where
    Self: HasY<Out>,
    TypeCodom<Out>: Eq,
    Out: Outcome,
    PSol: Uncomputed<SolId, Dom, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
{
}

impl<PSol, SolId, Dom, SInfo, Out> PartialOrd for Computed<PSol, SolId, Dom, Out, SInfo>
where
    Self: HasY<Out>,
    TypeCodom<Out>: PartialOrd,
    Out: Outcome,
    PSol: Uncomputed<SolId, Dom, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y.partial_cmp(&other.y())
    }
}

impl<PSol, SolId, Dom, SInfo, Out> Ord for Computed<PSol, SolId, Dom, Out, SInfo>
where
    Self: HasY<Out>,
    TypeCodom<Out>: Ord,
    Out: Outcome,
    PSol: Uncomputed<SolId, Dom, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.y.cmp(&other.y())
    }
}

impl<PSol, SolId, Dom, SInfo, Out> Orderable for Computed<PSol, SolId, Dom, Out, SInfo>
where
    Self: HasY<Out>,
    TypeCodom<Out>: Orderable,
    Out: Outcome,
    PSol: Uncomputed<SolId, Dom, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
{
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.y.ord_cmp(&other.y())
    }
}

impl<PSol, SolId, Dom, SInfo, Out> Dominate for Computed<PSol, SolId, Dom, Out, SInfo>
where
    Self: HasY<Out>,
    Out::Cod: Multi<Out>,
    TypeCodom<Out>: Dominate,
    Out: Outcome,
    PSol: Uncomputed<SolId, Dom, SInfo>,
    SolId: Id,
    SInfo: SolInfo,
    Dom: Domain,
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

impl<PSol, SolId, Dom, Info, Out> Computed<PSol, SolId, Dom, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    /// Creates a new [`Computed`] from a [`BaseSol`](crate::solution::BaseSol) and a [`TypeCodom`](crate::Codomain::TypeCodom).
    pub fn new(sol: PSol, y: Arc<TypeCodom<Out>>) -> Self {
        Computed {
            sol,
            y,
            _id: PhantomData,
            _dom: PhantomData,
            _info: PhantomData,
        }
    }

    /// Creates a vec of [`Computed`] from an iterator of [`BaseSol`](crate::solution::BaseSol)s,
    /// and an iterator of [`Arc<`TypeCodom`>](crate::Codomain::TypeCodom).
    pub fn new_vec<I, J>(sol: I, y: J) -> Vec<Self>
    where
        I: IntoIterator<Item = PSol>,
        J: IntoIterator<Item = Arc<TypeCodom<Out>>>,
    {
        sol.into_iter()
            .zip(y)
            .map(|(s, cod)| Self::new(s, cod))
            .collect()
    }

    pub fn xy(&self) -> Xy<PSol::Raw, TypeCodom<Out>> {
        Xy {
            x: self.clone_x(),
            y: self.y(),
        }
    }
}
