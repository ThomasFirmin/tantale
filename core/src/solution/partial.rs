use serde::{Deserialize, Serialize};

use crate::domain::{Domain, TypeDom};
use crate::solution::{Id, SolInfo, Solution};

use std::{fmt::Debug, sync::Arc};

/// A non-evaluated [`Solution`].
///
/// # Attributes
/// * `id` : [`Id`] - The unique [`ID`] of the solution.
/// * `x` : [`Arc`]`<[Dom::`[`TypeDom`](Domain::TypeDom)`]>` - A vector of [`TypeDom`](Domain::TypeDom).
/// * `info` : `[`Arc`]`<Info>` - Information given by the [`Optimizer`] and linked to a specific [`Solution`].
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Partial<SolId, Dom, Info>
where
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    pub id: SolId,
    pub x: Arc<[Dom::TypeDom]>,
    pub info: Arc<Info>,
}

impl<SolId, Dom, Info> Partial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
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
    /// use tantale::core::{Solution,Partial,Real,EmptyInfo,SId,Id};
    ///
    /// let x = std::sync::Arc::from(vec![0.0;5]);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = Partial::<_,Real,_>::new(SId::generate(),x,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    pub fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: AsRef<[TypeDom<Dom>]>,
    {
        let xarc: Arc<[TypeDom<Dom>]> = Arc::from(x.as_ref());
        Partial { id, x: xarc, info }
    }
    /// Creates the default slice of a [`Partial`] of `n` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,Real,EmptyInfo,SId,Id};
    ///
    /// let x = Partial::<SId,Real,EmptyInfo>::default_x(5);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = Partial::<_,Real,_>::new(SId::generate(),x,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    pub fn default_x(n: usize) -> Vec<TypeDom<Dom>> {
        vec![TypeDom::<Dom>::default(); n]
    }
    /// Creates a default [`Partial`] of `n` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,Real,EmptyInfo,SId};
    ///
    /// let x = Partial::<SId,Real,EmptyInfo>::default_x(5);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = Partial::<SId,Real,_>::new_default(5,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    pub fn new_default(n: usize, info: Arc<Info>) -> Self {
        Self::new(SolId::generate(), Self::default_x(n), info)
    }
    /// Creates an empty slice of [`Arc`][`<Partials>`](Partial) with `size` reserved capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,Real,EmptyInfo,SId};
    ///
    /// let mut vec_sol = Partial::<SId,Real,EmptyInfo>::new_vec(10);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// for _ in 0..10{
    ///     let psol = Partial::new_default(5,info.clone());
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
    pub fn new_vec(size: usize) -> Vec<Arc<Self>> {
        let mut v = Vec::new();
        v.reserve_exact(size);
        v
    }
    /// Creates a [`Vec`] of [`Arc`][`<Partials>`](Partial) of `n` elements with default `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,Real,EmptyInfo,SId};
    ///
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let vec_sol = Partial::<SId,Real,EmptyInfo>::new_default_vec(5,info,10);
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
    pub fn new_default_vec(n: usize, info: Arc<Info>, size: usize) -> Vec<Arc<Self>> {
        let mut v = Self::new_vec(size);
        for _ in 0..size {
            v.push(Arc::new(Self::new_default(n, info.clone())));
        }
        v
    }

    /// Given a [`Partial`] of type [`Self`] and a slice of type [`TypeDom`]`<B>`,
    /// creates the twin [`Partial`] of type `B`.
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used in [`onto_opt`](tantale::core::searchspace::onto_opt).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,Partial,Real,Int,EmptyInfo,SId,Id};
    ///
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0];
    /// let x_2 = vec![5,6,7,8];
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = Partial::<_,Real,EmptyInfo>::new(SId::generate(),x_1,info);
    /// let int_sol : Partial<_,Int,EmptyInfo> = real_sol.twin(x_2);
    ///
    /// println!("REAL ID : {}", real_sol.get_id().id);
    /// println!("INT ID : {}", int_sol.get_id().id);
    ///
    /// for (elem1, elem2) in real_sol.x.iter().zip(int_sol.x.iter()){
    ///     println!("{},{}", elem1, elem2);
    /// }
    ///
    /// ```
    pub fn twin<B, T>(&self, x: T) -> Partial<SolId, B, Info>
    where
        B: Domain,
        T: AsRef<[TypeDom<B>]>,
    {
        Partial::new(self.get_id(), x, self.get_info())
    }
}
impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for Partial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
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
