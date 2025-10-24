use serde::{Deserialize, Serialize};

use crate::domain::{Domain, TypeDom};
use crate::solution::{Id, SolInfo, Solution};

use std::{fmt::Debug, sync::Arc};

/// Describes the fidelity of a [`Solution`].
///
/// * [`New`](FidelState::New) : A newly created solution.
/// * [`Resume`](FidelState::Resume)`(f64)` : Resume the evaluation of [`Solution`].
///   A `f64`, can be used within the raw objective function to describe by 'how much' the evaluation should be evaluated.
/// * [`Discard`](FidelState::Discard) : Discard a [`Solution`] that has already been evaluated for a few steps.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Fidelity {
    New,
    Resume(f64),
    Discard,
}

/// A non-evaluated [`Solution`].
pub trait Partial<SolId, Dom, Info>: Solution<SolId, Dom, Info>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a> + Debug,
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    type Twin<B: Domain>: Partial<SolId, B, Info>;

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
    /// use tantale::core::{Solution,BasePartial,Real,EmptyInfo,SId,Id};
    ///
    /// let x = std::sync::Arc::from(vec![0.0;5]);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = BasePartial::<_,Real,_>::new(SId::generate(),x,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: AsRef<[TypeDom<Dom>]>;

    /// Creates a [`Partial`] of `n` elements.
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,BasePartial,Real,EmptyInfo,SId,Id};
    ///
    /// let x = BasePartial::<SId,Real,EmptyInfo>::default_x(5);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = BasePartial::<_,Real,_>::new(SId::generate(),x,info);
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
    /// use tantale::core::{Solution,BasePartial,Real,EmptyInfo,SId};
    ///
    /// let x = BasePartial::<SId,Real,EmptyInfo>::default_x(5);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = BasePartial::<SId,Real,_>::new_default(5,info);
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
    /// use tantale::core::{Solution,BasePartial,Real,EmptyInfo,SId};
    ///
    /// let mut vec_sol = BasePartial::<SId,Real,EmptyInfo>::new_vec(10);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// for _ in 0..10{
    ///     let psol = BasePartial::new_default(5,info.clone());
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
    /// use tantale::core::{Solution,BasePartial,Real,EmptyInfo,SId};
    ///
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let vec_sol = BasePartial::<SId,Real,EmptyInfo>::new_default_vec(5,info,10);
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

    /// Given a [`BasePartial`] of type [`Self`] and a slice of type [`TypeDom`]`<B>`,
    /// creates the twin [`BasePartial`] of type `B`.
    /// A twin, has the same `id` as [`Self`], but has a diffferent type.
    /// It is mostly used in [`onto_opt`](tantale::core::searchspace::onto_opt).
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,BasePartial,Real,Int,EmptyInfo,SId,Id};
    ///
    /// let x_1 = vec![0.0,1.0,2.0,3.0,4.0];
    /// let x_2 = vec![5,6,7,8];
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = BasePartial::<_,Real,EmptyInfo>::new(SId::generate(),x_1,info);
    /// let int_sol : BasePartial<_,Int,EmptyInfo> = real_sol.twin(x_2);
    ///
    /// println!("REAL ID : {}", real_sol.get_id().id);
    /// println!("INT ID : {}", int_sol.get_id().id);
    ///
    /// for (elem1, elem2) in real_sol.x.iter().zip(int_sol.x.iter()){
    ///     println!("{},{}", elem1, elem2);
    /// }
    ///
    /// ```
    fn twin<B, T>(&self, x: T) -> Self::Twin<B>
    where
        B: Domain,
        T: AsRef<[TypeDom<B>]>;
}

/// Describes a [`Partial`] associated to a [`Fidelity`].
pub trait FidelityPartial<SolId, Dom, Info> : Partial<SolId, Dom, Info>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a> + Debug,
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    fn get_fidelity(&self)->Fidelity;

    /// Modifies the [`Fidelity`] of a [`FidPartial`] to [`Resume`](Fidelity::Resume).
    fn resume<B:Domain>(&mut self, twin: &mut Self::Twin<B>, value: f64);

    /// Modifies the [`Fidelity`] of a [`FidPartial`] to [`Discard`](Fidelity::Discard).
    fn discard<B:Domain>(&mut self, twin: &mut Self::Twin<B>);
}

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
pub struct BasePartial<SolId, Dom, Info>
where
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    pub id: SolId,
    pub x: Arc<[Dom::TypeDom]>,
    pub info: Arc<Info>,
}

impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for BasePartial<SolId, Dom, Info>
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

impl<SolId, Dom, Info> Partial<SolId, Dom, Info> for BasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Twin<B: Domain> = BasePartial<SolId, B, Info>;

    /// Creates a new [`BasePartial`] from a slice of [`TypeDom<Dom>`].
    ///
    /// # Attributes
    ///
    /// * `id` : `SolId` - A unique [`Id`].
    /// * `x` : [`Arc`]`<[`[`TypeDom`]`<Dom>]>` - A basic solution from the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Solution,BasePartial,Real,EmptyInfo,SId,Id};
    ///
    /// let x = std::sync::Arc::from(vec![0.0;5]);
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = BasePartial::<_,Real,_>::new(SId::generate(),x,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: AsRef<[TypeDom<Dom>]>,
    {
        let xarc: Arc<[TypeDom<Dom>]> = Arc::from(x.as_ref());
        BasePartial { id, x: xarc, info }
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

    fn twin<B, T>(&self, x: T) -> Self::Twin<B>
    where
        B: Domain,
        T: AsRef<[TypeDom<B>]>,
    {
        Self::Twin::new(self.get_id(), x, self.get_info())
    }
}




//---------------------//
//----- FIDELITY -----//
//---------------------//


/// A non-evaluated [`Solution`] containing a [`Fidelity`].
///
/// # Attributes
/// * `id` : [`Id`] - The unique [`ID`] of the solution.
/// * `x` : [`Arc`]`<[Dom::`[`TypeDom`](Domain::TypeDom)`]>` - A vector of [`TypeDom`](Domain::TypeDom).
/// * `fid` : [`Fidelity`] -  The fidelity associated to `x`.
/// * `info` : `[`Arc`]`<Info>` - Information given by the [`Optimizer`] and linked to a specific [`Solution`].
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Dom::TypeDom: Serialize",
    deserialize = "Dom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct FidPartial<SolId, Dom, Info>
where
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    pub id: SolId,
    pub x: Arc<[Dom::TypeDom]>,
    pub fid: Fidelity,
    pub info: Arc<Info>,
}

impl<SolId, Dom, Info> FidPartial<SolId, Dom, Info>
where
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    /// Creates a new [`FidPartial`] from a slice of [`TypeDom<Dom>`].
    ///
    /// # Attributes
    ///
    /// * `id` : `SolId` - A unique [`Id`].
    /// * `x` : [`Arc`]`<[`[`TypeDom`]`<Dom>]>` - A basic solution from the [`Searchspace`].
    ///
    /// # Example
    ///
    /// ```
    /// use tantale::core::{Fidelity,Solution,FidPartial,Real,EmptyInfo,SId,Id};
    ///
    /// let x = std::sync::Arc::from(vec![0.0;5]);
    /// let fidelity = Fidelity::New;
    /// let info = std::sync::Arc::new(EmptyInfo{});
    ///
    /// let real_sol = FidPartial::<_,Real,_>::new(SId::generate(),x,fidelity,info);
    ///
    /// for elem in real_sol.get_x().iter(){
    ///     println!("{},", elem);
    /// }
    ///
    /// ```
    pub fn _new<T>(id: SolId, x: T, fid:Fidelity, info: Arc<Info>) -> Self
    where
        T: AsRef<[TypeDom<Dom>]>,
    {
        let xarc: Arc<[TypeDom<Dom>]> = Arc::from(x.as_ref());
        FidPartial { id, x: xarc, fid, info }
    }
}

impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for FidPartial<SolId, Dom, Info>
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

impl<SolId, Dom, Info> Partial<SolId, Dom, Info> for FidPartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Twin<B: Domain> = FidPartial<SolId, B, Info>;

    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: AsRef<[TypeDom<Dom>]>,
    {
        let xarc: Arc<[TypeDom<Dom>]> = Arc::from(x.as_ref());
        let fid = Fidelity::New;
        FidPartial { id, x: xarc, fid, info }
    }

    fn default_x(n: usize) -> Vec<TypeDom<Dom>> {
        vec![TypeDom::<Dom>::default(); n]
    }

    fn new_default(n: usize, info: Arc<Info>) -> Self {
        Self::_new(SolId::generate(), Self::default_x(n), Fidelity::New, info)
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

    fn twin<B, T>(&self, x: T) -> Self::Twin<B>
    where
        B: Domain,
        T: AsRef<[TypeDom<B>]>,
    {
        Self::Twin::_new(self.get_id(), x, self.fid, self.get_info())
    }
}


