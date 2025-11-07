use crate::domain::{Domain, TypeDom};
use crate::objective::Outcome;
use crate::solution::{Id, Partial, SolInfo, Solution};
use crate::{Codomain, Computed};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{fmt::Debug, sync::Arc};

/// A [`RawSol`] describes a [`Partial`] linked to a computed [`Outcome`].
///
/// # Attributes
/// * `sol` : [`Partial`]`<Dom,Info,N>` - A partial solution.
/// * `out` : `[`Arc`]`<Out>` - An [`Outcome`] from the evaluation of `sol` by the [`Objective`] function,
///
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize, Out: Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>, Out: for<'a> Deserialize<'a>",
))]
pub struct RawSol<PSol, SolId, Dom, Out, Info>
where
    PSol: Partial<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    pub sol: Arc<PSol>,
    pub out: Arc<Out>,
    _id: PhantomData<SolId>,
    _dom: PhantomData<Dom>,
    _info: PhantomData<Info>,
}

impl<PSol, SolId, Dom, Out, Info> Solution<SolId, Dom, Info> for RawSol<PSol, SolId, Dom, Out, Info>
where
    PSol: Partial<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    fn get_id(&self) -> SolId {
        self.sol.get_id()
    }

    fn get_x(&self) -> Arc<[TypeDom<Dom>]> {
        self.sol.get_x()
    }

    fn get_info(&self) -> Arc<Info> {
        self.sol.get_info()
    }
}

impl<PSol, SolId, Dom, Info, Out> RawSol<PSol, SolId, Dom, Out, Info>
where
    PSol: Partial<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    /// Creates a new [`RawSol`] from a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
    pub fn new(sol: Arc<PSol>, out: Arc<Out>) -> Self {
        RawSol {
            sol,
            out,
            _id: PhantomData,
            _dom: PhantomData,
            _info: PhantomData,
        }
    }

    /// Creates a vec of [`RawSol`] from an iterator of [`Arc`] [`Partial`]
    /// and an iterator of [`Arc`] [`TypeCodom`](Codomain::TypeCodom).
    pub fn new_vec<I, J>(sol: I, y: J) -> Vec<Arc<Self>>
    where
        I: IntoIterator<Item = Arc<PSol>>,
        J: IntoIterator<Item = Arc<Out>>,
    {
        sol.into_iter()
            .zip(y)
            .map(|(s, cod)| Arc::new(Self::new(s.clone(), cod)))
            .collect()
    }

    /// Returns the [`Partial`] [`Solution`].
    pub fn get_sol(&self) -> Arc<PSol> {
        self.sol.clone()
    }

    /// Returns the [`TypeCodom`](Codomain::TypeCodom), i.e. result from the computation of [`Partial`].
    pub fn get_out(&self) -> Arc<Out> {
        self.out.clone()
    }

    pub fn get_computed<Cod: Codomain<Out>>(
        &self,
        cod: Arc<Cod>,
    ) -> Computed<PSol, SolId, Dom, Cod, Out, Info> {
        let y = Arc::new(cod.get_elem(&self.out));
        Computed::new(self.sol.clone(), y)
    }
}
