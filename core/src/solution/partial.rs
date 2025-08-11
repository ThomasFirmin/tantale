use crate::domain::{Domain, TypeDom};
use crate::{solution::{Id, SolInfo, Solution}};

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

pub trait Partial<SolId, Dom, Info>: Solution<SolId, Dom, Info>
where
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    SolId: Id + PartialEq + Copy + Clone,
{
    /// Creates a new [`Partial`] from a slice of [`TypeDom<Dom>`].
    ///
    /// # Attributes
    ///
    /// * `id` : `SolId` - A unique [`Id`].
    /// * `x` : [`Arc`]`<[`[`TypeDom`]`<Dom>]>` - A basic solution from the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,PartialSol,Real,EmptyInfo,SId,Id};
    ///
    /// let x = std::sync::Arc::from(vec![0.0;5]);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = PartialSol::<_,Real,_>::new(SId::generate(),x,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: AsRef<[TypeDom<Dom>]>;
    /// Creates the default slice of a [`Partial`] of `n` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,PartialSol,Real,EmptyInfo,SId,Id};
    ///
    /// let x = PartialSol::<SId,Real,EmptyInfo>::default_x(5);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = PartialSol::<_,Real,_>::new(SId::generate(),x,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    fn default_x(n: usize) -> Vec<TypeDom<Dom>>;
    /// Creates a default [`Partial`] of `n` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,PartialSol,Real,EmptyInfo,SId};
    ///
    /// let x = PartialSol::<SId,Real,EmptyInfo>::default_x(5);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = PartialSol::<SId,Real,_>::new_default(5,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    fn new_default(n: usize, info: Arc<Info>) -> Self;
    /// Creates an empty slice of [`Arc`][`<Partials>`](Partial) with `size` reserved capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,PartialSol,Real,EmptyInfo,SId};
    ///
    /// let mut vec_sol = PartialSol::<SId,Real,EmptyInfo>::new_vec(10);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// for _ in 0..10{
    ///     let psol = PartialSol::new_default(5,info.clone());
    ///     vec_sol.push(std::sync::Arc::new(psol));
    /// }
    ///
    /// for sol in &vec_sol{
    ///     println!("[");
    ///     for elem in sol.get_x().iter(){
    ///         println!("{},", elem);
    ///     }
    ///     println!("], ");
    /// }
    ///
    /// ```
    fn new_vec(size: usize) -> Vec<Arc<Self>>;
    /// Creates a [`Vec`] of [`Arc`][`<Partials>`](Partial) of `n` elements with default `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,PartialSol,Real,EmptyInfo,SId};
    ///
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let vec_sol = PartialSol::<SId,Real,EmptyInfo>::new_default_vec(5,info,10);
    ///
    /// for sol in &vec_sol{
    ///     println!("[");
    ///     for elem in sol.get_x().iter(){
    ///         println!("{},", elem);
    ///     }
    ///     println!("], ");
    /// }
    ///
    /// ```
    fn new_default_vec(n: usize, info: Arc<Info>, size: usize) -> Vec<Arc<Self>>;

    /// Given a [`Partial`] of type [`Self`] and a slice of type [`TypeDom`]`<B>`,
    /// creates the twin [`Partial`] of type `B`.
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used in [`onto_opt`](tantale::core::searchspace::onto_opt).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,PartialSol,Real,Int,EmptyInfo,SId,Id};
    ///
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0];
    /// let x_2 = vec![5,6,7,8];
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = PartialSol::<_,Real,EmptyInfo>::new(SId::generate(),x_1,info);
    /// let int_sol : PartialSol<_,Int,EmptyInfo> = real_sol.twin(x_2);
    ///
    /// println!("REAL ID : {}", real_sol.get_id().id);
    /// println!("INT ID : {}", int_sol.get_id().id);
    ///
    /// for (elem1, elem2) in real_sol.x.iter().zip(int_sol.x.iter()){
    ///     println!("{},{}", elem1, elem2);
    /// }
    ///
    /// ```
    fn twin<Twin, B, T>(&self, x: T) -> Twin
    where
        B: Domain + Clone + Display + Debug,
        TypeDom<B>: Default + Copy + Clone + Display + Debug,
        Twin: Partial<SolId, B, Info>,
        T: AsRef<[TypeDom<B>]>,
    {
        Twin::new(self.get_id(), x, self.get_info())
    }
}

/// A non-evaluated [`Solution`].
///
/// # Attributes
/// * `id` : [`Id`] - The unique [`ID`] of the solution.
/// * `x` : [`Arc`]`<[Dom::`[`TypeDom`](Domain::TypeDom)`]>` - A vector of [`TypeDom`](Domain::TypeDom).
/// * `info` : `[`Arc`]`<Info>` - Information given by the [`Optimizer`] and linked to a specific [`Solution`].
pub struct PartialSol<SolId, Dom, Info>
where
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    SolId: Id + PartialEq + Clone + Copy,
{
    pub id: SolId,
    pub x: Arc<[TypeDom<Dom>]>,
    pub info: Arc<Info>,
}

impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for PartialSol<SolId, Dom, Info>
where
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    SolId: Id + PartialEq + Clone + Copy,
{
    fn get_id(&self) -> SolId {
        self.id
    }

    fn get_x(&self) -> Arc<[TypeDom<Dom>]> {
        self.x.clone()
    }

    fn get_info(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<SolId, Dom, Info> Partial<SolId, Dom, Info> for PartialSol<SolId, Dom, Info>
where
    Dom: Domain + Clone + Display + Debug,
    Info: SolInfo,
    TypeDom<Dom>: Default + Copy + Clone + Display + Debug,
    SolId: Id + PartialEq + Clone + Copy,
{
    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: AsRef<[TypeDom<Dom>]>,
    {
        let xarc: Arc<[TypeDom<Dom>]> = Arc::from(x.as_ref());
        PartialSol { id, x: xarc, info }
    }

    fn default_x(n: usize) -> Vec<TypeDom<Dom>> {
        vec![TypeDom::<Dom>::default(); n]
    }

    fn new_default(n: usize, info: Arc<Info>) -> Self {
        Self::new(SolId::generate(), Self::default_x(n), info)
    }

    fn new_vec(size: usize) -> Vec<Arc<Self>> {
        let mut v = Vec::new();
        v.reserve_exact(size);
        v
    }

    fn new_default_vec(n: usize, info: Arc<Info>, size: usize) -> Vec<Arc<Self>> {
        let mut v = Self::new_vec(size);
        for _ in 0..size {
            v.push(Arc::new(Self::new_default(n, info.clone())));
        }
        v
    }
}
