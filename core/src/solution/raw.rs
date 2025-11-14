use crate::domain::Domain;
use crate::objective::Outcome;
use crate::solution::{Id, Partial, SolInfo};

use std::marker::PhantomData;
use std::sync::Arc;

/// A [`RawSol`] describes a [`Partial`] linked to a computed [`Outcome`].
///
/// # Attributes
/// * `sol` : [`Partial`]`<Dom,Info,N>` - A partial solution.
/// * `out` : `Out` - An [`Outcome`] from the evaluation of `sol` by the [`Objective`] function,
///
#[derive(Debug)]
pub struct RawSol<'a, PSol, SolId, Dom, Out, Info>
where
    PSol: Partial<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    pub sol: &'a PSol,
    pub out: Arc<Out>,
    _id: PhantomData<SolId>,
    _dom: PhantomData<Dom>,
    _info: PhantomData<Info>,
}

impl<'a, PSol, SolId, Dom, Info, Out> RawSol<'a, PSol, SolId, Dom, Out, Info>
where
    PSol: Partial<SolId, Dom, Info>,
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    /// Creates a new [`RawSol`] from a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
    pub fn new(sol: &'a PSol, out: Arc<Out>) -> Self {
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
    pub fn new_vec<I, J>(sol: I, y: J) -> Vec<Self>
    where
        I: IntoIterator<Item = &'a PSol>,
        J: IntoIterator<Item = Arc<Out>>,
    {
        sol.into_iter()
            .zip(y)
            .map(|(s, cod)| Self::new(s, cod))
            .collect()
    }

    /// Returns the [`Partial`] [`Solution`].
    pub fn get_sol(&self) -> &PSol {
        self.sol
    }

    /// Returns a reference to the [`Outcome`], i.e. result from the computation of [`Partial`].
    pub fn get_out(&self) -> &Out {
        &self.out
    }
}
