use crate::domain::{Domain, TypeDom};
use crate::solution::{Partial, PartialSol, Solution};
use crate::objective::{Codomain,Outcome};
use crate::optimizer::SolInfo;

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
    P: Partial<Dom,Info>,
    Cod: Codomain<Out>,
    Out : Outcome,
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
    fn new_vec<I,J>(sol: I, y: J) -> Vec<Self>
    where
        I: IntoIterator<Item = P>,
        J: IntoIterator<Item = Arc<Cod::TypeCodom>>;

    /// Returns the [`Partial`] [`Solution`].
    fn get_sol(&self)->Arc<P>;

    /// Returns the [`TypeCodom`](Codomain::TypeCodom), i.e. result from the computation of [`Partial`].
    fn get_y(&self)->Arc<Cod::TypeCodom>;

    /// Given a [`Computed`] of type [`Self`] and a slice of type [`TypeDom`]`<B>`, 
    /// creates the twin [`Computed`] of type [`B`].
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used in [`onto_opt`](tantale::core::searchspace::onto_opt) 
    /// 
    /// # Example
    /// 
    /// ```
    /// use tantale::core::{ComputedSol,Real,Int,SingleCodomain,HashOut};
    /// use std::sync::Arc;
    /// 
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0].into_boxed_slice();
    /// let x_2 = vec![5,6,7,8].into_boxed_slice();
    /// 
    /// let real_sol = ComputedSol::<Real,SingleCodomain<HashOut>,HashOut,5>::new(std::process::id(), x_1, None);
    /// let int_sol : ComputedSol<Int,SingleCodomain<HashOut>,HashOut,5> = real_sol.twin(x_2);
    /// 
    /// println!("REAL ID : {},{}", real_sol.id.0,real_sol.id.1);
    /// println!("INT ID : {},{}", int_sol.id.0,int_sol.id.1);
    /// 
    /// for (elem1, elem2) in real_sol.x.iter().zip(int_sol.x.iter()){
    ///     println!("{},{}", elem1, elem2);
    /// }
    /// 
    /// ```
    fn twin<Twin,B,T,Pb>(&self, x: T) -> Twin
    where 
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        Pb: Partial<B,Info>,
        Twin:Computed<Pb,B,Info,Cod,Out>,
        T : AsRef<[TypeDom<B>]>
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
pub struct ComputedSol<Dom, Cod, Out, Info, const N:usize>
where
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    Info : SolInfo,
    Cod: Codomain<Out>,
    Out : Outcome,
{
    pub sol : Arc<PartialSol<Dom,Info,N>>,
    pub y: Arc<Cod::TypeCodom>,
}

impl <Dom,Cod,Out,Info,const N:usize> Solution<Dom,Info> for ComputedSol<Dom, Cod, Out, Info, N>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    fn get_id(&self)->(u32,usize) {
        self.sol.get_id()
    }

    fn get_x(&self)->Arc<[TypeDom<Dom>]> {
        self.sol.get_x()
    }

    fn get_info(&self)->Arc<Info> {
        self.sol.get_info()
    }
}

impl <Dom,Info,Cod,Out,const N:usize> Computed<PartialSol<Dom,Info,N>,Dom,Info,Cod,Out> for ComputedSol<Dom, Cod, Out, Info, N>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    fn new(sol: PartialSol<Dom,Info,N>, y: Arc<<Cod as Codomain<Out>>::TypeCodom>) -> Self {
        let solarc = Arc::new(sol);
        ComputedSol{ sol: solarc, y }
    }

    fn new_vec<I,J>(sol: I, y: J) -> Vec<Self>
    where
        I: IntoIterator<Item = PartialSol<Dom,Info,N>>,
        J: IntoIterator<Item = Arc<<Cod as Codomain<Out>>::TypeCodom>>
    {
        sol.into_iter().zip(y).map(|(s,cod)| Self::new(s, cod)).collect()
    }

    fn get_sol(&self)->Arc<PartialSol<Dom,Info,N>>{
        self.sol.clone()
    }

    fn get_y(&self)->Arc<<Cod as Codomain<Out>>::TypeCodom> {
        self.y.clone()
    }
    
    
}