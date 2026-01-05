use crate::{
    domain::{Codomain, Domain},
    objective::{Outcome, Step},
    solution::{
        HasFidelity, HasId, HasSolInfo, HasStep, HasUncomputed, HasY, Id, IntoComputed, SolInfo,
        Solution, Uncomputed,
    },
    EvalStep, Fidelity,
};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{fmt::Debug, sync::Arc};

/// A solution of the [`Objective`](tantale::core::Objective) or of the [`Optimizer`](tantale::core::Optimizer)
/// [`Domains`](Domain). The solution is defined by a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
///
/// # Attributes
/// * `sol` : [`Partial`]`<Dom,Info,N>` - An already created partial solution.
/// * `y` : `[`Arc`]`<Cod<Out>>` - State of the evaluation of a solution. This a [`TypeCodom`](tantale::core::objective::comdomain::Codomain::TypeCodom),
///
/// # Note
///
/// A [`Computed`] can only be created from a pair of [`Partial`] of respectively the [`Opt`](Optimizer) and the [`Obj`](Objective)
/// [`Domain`] type.
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
    fn fidelity(&self) -> Fidelity {
        self.sol.fidelity()
    }

    fn set_fidelity(&mut self, fidelity: Fidelity) {
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
    /// Returns the [`TypeCodom`](Codomain::TypeCodom), i.e. result from the computation of [`Partial`].
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
    /// Returns the [`Uncomputed`] the [`Computed`].
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
