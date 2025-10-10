use crate::{Codomain, Computed};
use crate::domain::{Domain, TypeDom};
use crate::objective::Outcome;
use crate::solution::{Id, Partial, SolInfo, Solution};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{
    fmt::Debug,
    sync::Arc,
};

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
pub struct RawSol<SolId, Dom, Out, Info>
where
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    pub sol: Arc<Partial<SolId, Dom, Info>>,
    pub out: Arc<Out>,
    _id: PhantomData<SolId>,
    _dom: PhantomData<Dom>,
    _info: PhantomData<Info>,
}

impl<SolId, Dom, Out, Info> Solution<SolId, Dom, Info> for RawSol<SolId, Dom, Out, Info>
where
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

impl<SolId, Dom, Info, Out> RawSol<SolId, Dom, Out, Info>
where
    Dom: Domain,
    Info: SolInfo,
    Out: Outcome,
    SolId: Id,
{
    /// Creates a new [`RawSol`] from a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
    pub fn new(
        sol: Arc<Partial<SolId, Dom, Info>>,
        out:Arc<Out>,
    ) -> Self {
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
        I: IntoIterator<Item = Arc<Partial<SolId, Dom, Info>>>,
        J: IntoIterator<Item = Arc<Out>>,
    {
        sol.into_iter()
            .zip(y)
            .map(|(s, cod)| Arc::new(Self::new(s.clone(), cod)))
            .collect()
    }

    /// Returns the [`Partial`] [`Solution`].
    pub fn get_sol(&self) -> Arc<Partial<SolId, Dom, Info>> {
        self.sol.clone()
    }

    /// Returns the [`TypeCodom`](Codomain::TypeCodom), i.e. result from the computation of [`Partial`].
    pub fn get_out(&self) -> Arc<Out>{
        self.out.clone()
    }

    /// Given a [`RawSol`] of type [`Self`] and a slice of type [`TypeDom`]`<B>`,
    /// creates the twin [`RawSol`] of type [`B`].
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used in [`onto_opt`](tantale::core::searchspace::onto_opt)
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,RawSol,Partial,Real,Int,EmptyInfo,SId,Id};
    /// use std::sync::Arc;
    ///
    /// # use tantale::core::Outcome;
    /// # use serde::{Serialize,Deserialize};
    /// # #[derive(Serialize,Deserialize)]
    /// # struct OutExample(i32);
    /// # impl Outcome for OutExample{}
    ///
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0].into_boxed_slice();
    /// let x_2 = vec![5,6,7,8].into_boxed_slice();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let partial = Arc::new(Partial::new(SId::generate(),x_1,info));
    /// let out = Arc::new(OutExample(10));
    ///
    /// let real_sol = RawSol::<_,Real,OutExample,_>::new(partial,out);
    /// let int_sol : RawSol<_,Int,OutExample,_> = real_sol.twin(x_2);
    ///
    /// let id_r = real_sol.get_id();
    /// let id_i = int_sol.get_id();
    ///
    /// println!("REAL ID : {}", id_r.id);
    /// println!("INT ID : {}", id_i.id);
    ///
    /// for (elem1, elem2) in real_sol.get_x().iter().zip(int_sol.get_x().iter()){
    ///     println!("{},{}", elem1, elem2);
    /// }
    ///
    /// ```
    pub fn twin<B, T>(&self, x: T) -> RawSol<SolId, B, Out, Info>
    where
        B: Domain,
        T: AsRef<[TypeDom<B>]>,
    {
        let info = self.get_sol().get_info();
        let partial = Arc::new(Partial::new(self.get_sol().get_id(), x, info));
        RawSol::new(partial, self.get_out())
    }


    pub fn get_computed<Cod:Codomain<Out>>(&self, cod:Arc<Cod>) -> Computed<SolId,Dom,Cod,Out,Info>{
        let y = Arc::new(cod.get_elem(&self.out));
        Computed::new(self.get_sol(), y)
    }
}
