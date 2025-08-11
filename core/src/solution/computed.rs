use crate::domain::{Domain, TypeDom};
use crate::objective::{Codomain, Outcome};
use crate::solution::{Id, Partial, SolInfo, Solution};

use std::marker::PhantomData;
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

/// A solution of the [`Objective`](tantale::core::Objective) or of the [`Optimizer`](tantale::core::Optimizer)
/// [`Domains`](Domain). The solution is defined by a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
///
/// # Attributes
/// * `sol` : [`PartialSol`]`<Dom,Info,N>` - An already created partial solution.
/// * `y` : `[`Arc`]`<Cod<Out>>` - State of the evaluation of a solution. This a [`TypeCodom`](tantale::core::objective::comdomain::Codomain::TypeCodom),
///
/// # Note
///
/// A [`ComputedSol`] can only be created from a pair of [`PartialSol`] of respectively the [`Opt`](Optimizer) and the [`Obj`](Objective)
/// [`Domain`] type.
pub struct Computed<SolId, P, Dom, Cod, Out, Info>
where
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    SolId: Id + PartialEq + Clone + Copy,
    P: Partial<SolId, Dom, Info>,
{
    pub sol: Arc<P>,
    pub y: Arc<Cod::TypeCodom>,
    _id: PhantomData<SolId>,
    _dom: PhantomData<Dom>,
    _info: PhantomData<Info>,
}

impl<SolId, P, Dom, Cod, Out, Info> Solution<SolId, Dom, Info>
    for Computed<SolId, P, Dom, Cod, Out, Info>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out: Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    SolId: Id + PartialEq + Clone + Copy,
    P: Partial<SolId, Dom, Info>,
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

impl<SolId, P, Dom, Info, Cod, Out> Computed<SolId, P, Dom, Cod, Out, Info>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out: Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    SolId: Id + PartialEq + Clone + Copy,
    P: Partial<SolId, Dom, Info>,
{
    /// Creates a new [`Computed`] from a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
    pub fn new(sol: Arc<P>, y: Arc<<Cod as Codomain<Out>>::TypeCodom>) -> Self {
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
    pub fn new_vec<I, J>(sol: I, y: J) -> Vec<Arc<Self>>
    where
        I: IntoIterator<Item = Arc<P>>,
        J: IntoIterator<Item = Arc<<Cod as Codomain<Out>>::TypeCodom>>,
    {
        sol.into_iter()
            .zip(y)
            .map(|(s, cod)| Arc::new(Self::new(s.clone(), cod)))
            .collect()
    }

    /// Returns the [`Partial`] [`Solution`].
    pub fn get_sol(&self) -> Arc<P> {
        self.sol.clone()
    }

    /// Returns the [`TypeCodom`](Codomain::TypeCodom), i.e. result from the computation of [`Partial`].
    pub fn get_y(&self) -> Arc<<Cod as Codomain<Out>>::TypeCodom> {
        self.y.clone()
    }

    /// Given a [`Computed`] of type [`Self`] and a slice of type [`TypeDom`]`<B>`,
    /// creates the twin [`Computed`] of type [`B`].
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used in [`onto_opt`](tantale::core::searchspace::onto_opt)
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Computed,Partial,PartialSol,Real,Int,SingleCodomain,HashOut,EmptyInfo,SId,Id};
    /// use tantale::core::objective::codomain::ElemSingleCodomain;
    /// use std::sync::Arc;
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0].into_boxed_slice();
    /// let x_2 = vec![5,6,7,8].into_boxed_slice();
    /// let info = Arc::new(EmptyInfo{});
    ///
    /// let partial = Arc::new(PartialSol::new(SId::generate(),x_1,info));
    /// let y = Arc::new(ElemSingleCodomain{value:1.0});
    ///
    /// let real_sol = Computed::<_,_,Real,SingleCodomain<HashOut>,HashOut,_>::new(partial,y);
    /// let int_sol : Computed<_,PartialSol<_,Int,_>,_,SingleCodomain<HashOut>,HashOut,_> = real_sol.twin(x_2);
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
    pub fn twin<B, T, Pb>(&self, x: T) -> Computed<SolId, Pb, B, Cod, Out, Info>
    where
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        Pb: Partial<SolId, B, Info>,
        T: AsRef<[TypeDom<B>]>,
    {
        let info = self.get_sol().get_info();
        let partial = Arc::new(Pb::new(self.get_sol().get_id(), x, info));
        Computed::new(partial, self.get_y())
    }
}
