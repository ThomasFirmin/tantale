use crate::domain::{Domain, TypeDom};
use crate::objective::{Codomain, Outcome};
use crate::solution::{Id, Partial, SolInfo, Solution};

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
    PSol: Partial<SolId, Dom, Info>,
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

impl<PSol, SolId, Dom, Cod, Out, Info> Solution<SolId, Dom, Info>
    for Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Partial<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
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

impl<PSol, SolId, Dom, Info, Cod, Out> Computed<PSol, SolId, Dom, Cod, Out, Info>
where
    PSol: Partial<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolId: Id,
{
    /// Creates a new [`Computed`] from a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
    pub fn new(sol: PSol, y: Arc<<Cod as Codomain<Out>>::TypeCodom>) -> Self {
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

    /// Returns the [`Partial`] [`Solution`].
    pub fn get_sol(&self) -> &PSol {
        &self.sol
    }

    /// Returns the [`TypeCodom`](Codomain::TypeCodom), i.e. result from the computation of [`Partial`].
    pub fn get_y(&self) -> &<Cod as Codomain<Out>>::TypeCodom {
        &self.y
    }
}
