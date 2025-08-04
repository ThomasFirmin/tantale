use crate::domain::{Domain, TypeDom};
use crate::objective::{Codomain, Outcome};
use crate::solution::{Partial, PartialSol, SolInfo, Solution};

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

pub trait Computed<P, Dom, Info, Cod, Out>
where
    Self: Sized,
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    Info: SolInfo,
    P: Partial<Dom, Info>,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    /// Creates a new [`Computed`] from a [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
    ///
    /// # Notes
    ///
    /// When created, consumes the [`Partial`].
    fn new(sol: P, y: Arc<Cod::TypeCodom>) -> Self;

    /// Creates a vec of [`Computed`] from a vec of [`Partial`] and a vec of [`TypeCodom`](Codomain::TypeCodom).
    ///
    /// # Notes
    ///
    /// When created, consumes the [`Partial`].
    fn new_vec<I, J>(sol: I, y: J) -> Vec<Self>
    where
        I: IntoIterator<Item = P>,
        J: IntoIterator<Item = Arc<Cod::TypeCodom>>;

    /// Returns the [`Partial`] [`Solution`].
    fn get_sol(&self) -> Arc<P>;

    /// Returns the [`TypeCodom`](Codomain::TypeCodom), i.e. result from the computation of [`Partial`].
    fn get_y(&self) -> Arc<Cod::TypeCodom>;

    /// Given a [`Computed`] of type [`Self`] and a slice of type [`TypeDom`]`<B>`,
    /// creates the twin [`Computed`] of type [`B`].
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used in [`onto_opt`](tantale::core::searchspace::onto_opt)
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Computed,Partial,PartialSol,ComputedSol,Real,Int,SingleCodomain,HashOut,EmptyInfo};
    /// use tantale::core::objective::codomain::ElemSingleCodomain;
    ///
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0].into_boxed_slice();
    /// let x_2 = vec![5,6,7,8].into_boxed_slice();
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let partial = PartialSol::new(std::process::id(),x_1,info);
    /// let y = std::sync::Arc::new(ElemSingleCodomain{value:1.0});
    ///
    /// let real_sol = ComputedSol::<Real,SingleCodomain<HashOut>,HashOut,_>::new(partial,y);
    /// let int_sol : ComputedSol<Int,SingleCodomain<HashOut>,HashOut,_> = real_sol.twin(x_2);
    ///
    /// let id_r = real_sol.get_id();
    /// let id_i = int_sol.get_id();
    ///
    /// println!("REAL ID : {},{}", id_r.0,id_r.1);
    /// println!("INT ID : {},{}", id_i.0,id_i.1);
    ///
    /// for (elem1, elem2) in real_sol.get_x().iter().zip(int_sol.get_x().iter()){
    ///     println!("{},{}", elem1, elem2);
    /// }
    ///
    /// ```
    fn twin<Twin, B, T, Pb>(&self, x: T) -> Twin
    where
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        Pb: Partial<B, Info>,
        Twin: Computed<Pb, B, Info, Cod, Out>,
        T: AsRef<[TypeDom<B>]>,
    {
        let id = self.get_sol().get_id();
        let info = self.get_sol().get_info();
        let partial = Pb::build(id.0, id.1, x, info);
        Twin::new(partial, self.get_y())
    }
}

/// A solution of the [`Objective`](tantale::core::Objective) or of the [`Optimizer`](tantale::core::Optimizer) [`Domains`](Domain).
/// The solution is mostly defined by an associated unique [`Domain`] and [`Codomain`].
///
/// # Attributes
/// * `sol` : [`PartialSol`]`<Dom,Info,N>` - An already created partial solution.
/// * `y` : `[`Arc`]`<Cod<Out>>` - State of the evaluation of a solution. This a [`TypeCodom`](tantale::core::objective::comdomain::Codomain::TypeCodom),
///
/// # Note
///
/// A [`ComputedSol`] can only be created from a pair of [`PartialSol`] of respectively the [`Opt`](Optimizer) and the [`Obj`](Objective)
/// [`Domain`] type.
pub struct ComputedSol<Dom, Cod, Out, Info>
where
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    Info: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub sol: Arc<PartialSol<Dom, Info>>,
    pub y: Arc<Cod::TypeCodom>,
}

impl<Dom, Cod, Out, Info> Solution<Dom, Info> for ComputedSol<Dom, Cod, Out, Info>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out: Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    fn get_id(&self) -> (u32, usize) {
        self.sol.get_id()
    }

    fn get_x(&self) -> Arc<[TypeDom<Dom>]> {
        self.sol.get_x()
    }

    fn get_info(&self) -> Arc<Info> {
        self.sol.get_info()
    }
}

impl<Dom, Info, Cod, Out> Computed<PartialSol<Dom, Info>, Dom, Info, Cod, Out>
    for ComputedSol<Dom, Cod, Out, Info>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out: Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    fn new(sol: PartialSol<Dom, Info>, y: Arc<<Cod as Codomain<Out>>::TypeCodom>) -> Self {
        let solarc = Arc::new(sol);
        ComputedSol { sol: solarc, y }
    }

    fn new_vec<I, J>(sol: I, y: J) -> Vec<Self>
    where
        I: IntoIterator<Item = PartialSol<Dom, Info>>,
        J: IntoIterator<Item = Arc<<Cod as Codomain<Out>>::TypeCodom>>,
    {
        sol.into_iter()
            .zip(y)
            .map(|(s, cod)| Self::new(s, cod))
            .collect()
    }

    fn get_sol(&self) -> Arc<PartialSol<Dom, Info>> {
        self.sol.clone()
    }

    fn get_y(&self) -> Arc<<Cod as Codomain<Out>>::TypeCodom> {
        self.y.clone()
    }
}
