//! Computed (evaluated) solutions.
//!
//! A [`Computed`] solution pairs an [`Uncomputed`](crate::solution::Uncomputed) solution with the
//! corresponding codomain value (a [`Codomain::TypeCodom`]) produced by evaluating the raw
//! solution with an [`Objective`](crate::Objective). This is the evaluated form used by
//! optimizers and recorders.
//!
//! # Note
//!
//! For [`TypeCodom`](crate::domain::codomain::TypeCodom) that are [`Ord`] the
//! [`Computed`] solution implements [`Ord`] and can be used in sorted collections.
//! Same applies for [`PartialOrd`], [`PartialEq`] and [`Eq`].
//!
//! # Examples
//! ```
//! use tantale::core::{BasePartial, Computed, SingleCodomain, EmptyInfo, Id, Outcome, Real, SId};
//! use std::sync::Arc;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Outcome, Serialize, Deserialize)]
//! struct Out { value: f64 }
//!
//! let cod = SingleCodomain::new(|o: &Out| o.value);
//! let x = Arc::from(vec![0.1, 0.2]);
//! let info = Arc::new(EmptyInfo {});
//! let sol = BasePartial::<SId, Real, _>::new(SId::generate(), x, info);
//! let y = Arc::new(cod.get_elem(&Out { value: 0.5 }));
//!
//! let computed: Computed<_, SId, Real, _, Out, _> = Computed::new(sol, y);
//! assert_eq!(computed.get_x().len(), 2);
//! ```
//!
//! ```
//! use tantale::core::{BasePartial, SingleCodomain, EmptyInfo, Id, IntoComputed, Outcome, Real, SId};
//! use std::sync::Arc;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Outcome, Serialize, Deserialize)]
//! struct Out { value: f64 }
//!
//! let cod = SingleCodomain::new(|o: &Out| o.value);
//! let x = Arc::from(vec![0.4, 0.8]);
//! let info = Arc::new(EmptyInfo {});
//! let sol = BasePartial::<SId, Real, _>::new(SId::generate(), x, info);
//! let y = Arc::new(cod.get_elem(&Out { value: 1.2 }));
//!
//! let computed = sol.into_computed::<SingleCodomain<Out>, Out>(y);
//! assert_eq!(computed.get_x().len(), 2);
//! ```

use crate::{
    EvalStep, Fidelity,
    domain::{Codomain, Domain},
    objective::{Outcome, Step},
    solution::{
        HasFidelity, HasId, HasSolInfo, HasStep, HasUncomputed, HasY, Id, IntoComputed, SolInfo,
        Solution, Uncomputed,
    },
};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{fmt::Debug, sync::Arc};

/// A solution of the [`Objective`](crate::Objective) or the [`Optimizer`](crate::Optimizer)
/// [`Domains`](Domain), enriched with the computed [`Codomain`] value [`TypeCodom`](Codomain::TypeCodom).
///
/// # Attributes
/// * `sol` : [`Uncomputed`] - An already created partial solution.
/// * `y` : [`Arc`]`<Cod::TypeCodom>` - The computed codomain value.
///
/// # Note
///
/// A [`Computed`] is obtained by evaluating the raw solution from an [`Uncomputed`] with the
/// [`Objective`](crate::Objective) and then extracting a [`Codomain::TypeCodom`].
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize, Cod::TypeCodom: Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>, Cod::TypeCodom: for<'a> Deserialize<'a>",
))]
pub struct Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolId: Id,
{
    pub sol: PSol,
    pub y: Arc<Cod::TypeCodom>,
    _id: PhantomData<SolId>,
    _dom: PhantomData<Dom>,
    _info: PhantomData<Info>,
}

impl<PSol, SolId, Dom, Cod, Out, Info> HasId<SolId> for Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolId: Id,
{
    /// Return the identifier of the underlying uncomputed solution.
    fn get_id(&self) -> SolId {
        self.sol.get_id()
    }
}

impl<PSol, SolId, Dom, Cod, Out, Info> HasSolInfo<Info>
    for Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolId: Id,
{
    /// Return the [`SolInfo`](crate::SolInfo) associated with the underlying solution.
    fn get_sinfo(&self) -> Arc<Info> {
        self.sol.get_sinfo()
    }
}

impl<PSol, SolId, Dom, Cod, Out, Info> HasStep for Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + HasStep,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
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

    fn set_step(&mut self, value: EvalStep) {
        self.sol.set_step(value);
    }
}

impl<PSol, SolId, Dom, Cod, Out, Info> HasFidelity for Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + HasFidelity,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
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

impl<PSol, SolId, Dom, Cod, Out, Info> HasY<Cod, Out> for Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + IntoComputed,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolId: Id,
{
    /// Returns the computed [`TypeCodom`](Codomain::TypeCodom) for this solution.
    fn get_y(&self) -> Arc<Cod::TypeCodom> {
        self.y.clone()
    }
}

impl<PSol, SolId, Dom, Cod, Out, Info> HasUncomputed<SolId, Dom, Info>
    for Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info> + IntoComputed,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolId: Id,
{
    type Uncomputed = PSol;
    /// Returns the underlying [`Uncomputed`] solution.
    fn get_uncomputed(&self) -> &Self::Uncomputed {
        &self.sol
    }
}

impl<PSol, SolId, Dom, Cod, Out, Info> Solution<SolId, Dom, Info>
    for Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    SolId: Id,
    Dom: Domain,
    Cod: Codomain<Out>,
    Out: Outcome,
    Info: SolInfo,
{
    type Raw = PSol::Raw;
    type Twin<B: Domain> = Computed<PSol::TwinUC<B>, SolId, B, Cod, Out, Info>;

    /// Return the raw representation of the underlying solution.
    fn get_x(&self) -> Self::Raw {
        self.sol.get_x()
    }

    fn twin<B: Domain>(
        &self,
        x: <Self::Twin<B> as Solution<SolId, B, Info>>::Raw,
    ) -> Self::Twin<B> {
        Computed::new(Uncomputed::twin(&self.sol, x), self.y.clone())
    }
}

impl<PSol, SolId, Dom, Info, Cod, Out> Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Uncomputed<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolId: Id,
{
    /// Creates a new [`Computed`] from a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
    pub fn new(sol: PSol, y: Arc<Cod::TypeCodom>) -> Self {
        Computed {
            sol,
            y,
            _id: PhantomData,
            _dom: PhantomData,
            _info: PhantomData,
        }
    }

    /// Creates a vec of [`Computed`] from an iterator of [`Arc`] [`Partial`]
    /// and an iterator of [`Arc`] [`TypeCodom`](Codomain::TypeCodom).
    pub fn new_vec<I, J>(sol: I, y: J) -> Vec<Self>
    where
        I: IntoIterator<Item = PSol>,
        J: IntoIterator<Item = Arc<<Cod as Codomain<Out>>::TypeCodom>>,
    {
        sol.into_iter()
            .zip(y)
            .map(|(s, cod)| Self::new(s, cod))
            .collect()
    }
}
