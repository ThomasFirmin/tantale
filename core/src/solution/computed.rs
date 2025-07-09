use crate::objective::codomain::Codomain;
use crate::objective::Outcome;

pub trait Computed<Dom, Info, Cod, Out>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
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
    fn twin<Twin,B,T>(&self, x: T) -> Twin
    where 
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        Twin:Computed<B,Info,Cod,Out>,
        T : AsRef<[TypeDom<B>]>;
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
    Cod: Codomain<Out>,
    Out : Outcome,
    Info : SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    pub sol : PartialSol<Dom,Info,N>,
    pub y: Arc<Cod::TypeCodom>,
}

impl<Dom, Cod, Out, Info, const N: usize> ComputedSol<Dom, Cod, Out, Info, N>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    fn new(sol: PartialSol<Dom,Info,N>, y: Arc<Cod::TypeCodom>) -> Self {
        ComputedSol {sol,y}
    }
}

impl <Dom,Cod,Out,Info,const N:usize> Solution<Dom,Info> for ComputedSol<Dom, Cod, Out, Info, N>
where
    Dom: Domain + Clone + Display + Debug,
    Cod: Codomain<Out>,
    Out : Outcome,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
{
    fn get_id(&self)->(usize, u32) {
        self.sol.get_id()
    }

    fn get_x(&self)->Arc<[TypeDom<Dom>]> {
        self.sol.get_x()
    }

    fn get_info(&self)->Arc<Info> {
        self.sol.get_info()
    }
}