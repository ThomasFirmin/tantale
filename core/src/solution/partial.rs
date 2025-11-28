use crate::domain::Domain;
use crate::domain::onto::OntoDom;
use crate::objective::Step;
use crate::recorder::csv::CSVWritable;
use crate::solution::{Id, SolInfo, Solution};

use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

/// Describes the fidelity of a [`Solution`].
///
/// * [`New`](Fidelity::New) : A newly created solution.
/// * [`Resume`](Fidelity::Resume)`(f64)` : Resume the evaluation of [`Solution`].
///   A `f64`, can be used within the raw objective function to describe by 'how much' the evaluation should be evaluated.
/// * [`Last`](Fidelity::Last): When the [`EvalState`](crate::core::objective::EvalStep) is at its `penultimate` step, allows a last step.
/// * [`Discard`](Fidelity::Discard) : Discard a [`Solution`] that has already been evaluated for a few steps.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Fidelity {
    Resume(f64),
    Discard,
}

impl PartialEq for Fidelity {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Resume(l0), Self::Resume(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Display for Fidelity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fidelity::Resume(v) => write!(f, "{}", v),
            Fidelity::Discard => write!(f, "Discard"),
        }
    }
}

impl CSVWritable<(), ()> for Fidelity {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("fidelity")])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.to_string()])
    }
}

/// A non-evaluated [`Solution`].
pub trait Partial<SolId, Dom, Info>: Solution<SolId, Dom, Info>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a> + Debug,
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    type Twin<B: Domain>: Partial<SolId, B, Info, Twin<Dom> = Self>;

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
        T: Into<Self::Raw>;
        
    fn twin<T, B>(&self, x: T) -> <Self as Partial<SolId, Dom, Info>>::Twin<B>
    where
        T: Into<<<Self as Partial<SolId, Dom, Info>>::Twin<B> as Solution<SolId, B, Info>>::Raw>,
        B: Domain;
}

/// Describes a [`Partial`] associated to a [`Fidelity`].
pub trait FidelityPartial<SolId, Dom, Info>: Partial<SolId, Dom, Info>
where
    Self: Sized + Serialize + for<'a> Deserialize<'a> + Debug,
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    type Twin<B: Domain>: FidelityPartial<SolId, B, Info, Twin<Dom> = Self>;

    fn get_step(&self) -> Step;
    fn set_step(&mut self, step:Step);
    fn get_fidelity(&self) -> Option<Fidelity>;
    fn set_fidelity(&mut self, fidelity:Option<Fidelity>);

    /// Modifies the [`Fidelity`] of a [`FidelityPartial`] to [`Resume`](Fidelity::Resume).
    fn resume<B: Domain>(&mut self, twin: &mut <Self as FidelityPartial<SolId, Dom, Info>>::Twin<B>, value: f64);

    /// Modifies the [`Fidelity`] of a [`FidelityPartial`] to [`Discard`](Fidelity::Discard).
    fn discard<B: Domain>(&mut self, twin: &mut <Self as FidelityPartial<SolId, Dom, Info>>::Twin<B>);
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
    type Raw = Arc<[Dom::TypeDom]>;
    type Twin<B: OntoDom<Dom>>= BasePartial<SolId, B, Info> where Dom:OntoDom<B>;

    fn get_id(&self) -> SolId {
        self.id
    }
    
    fn get_x<T:AsRef<Self::Raw> + From<Self::Raw>>(&self) -> T {
        self.x.clone().into()
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
        T: Into<Self::Raw>,
    {
        BasePartial { id, x: x.into(), info }
    }
    
    fn twin<T, B>(&self, x: T) -> <Self as Partial<SolId, Dom, Info>>::Twin<B>
    where
        T: Into<<<Self as Partial<SolId, Dom, Info>>::Twin<B> as Solution<SolId, B, Info>>::Raw>,
        B: Domain,
    {
        BasePartial { id: self.id, x: x.into(), info: self.info.clone() }
    }
    
    // fn default_x(n: usize) -> Vec<TypeDom<Dom>> {
    //     vec![TypeDom::<Dom>::default(); n]
    // }

    // fn default(n: usize, info: Arc<Info>) -> Self {
    //     Self::new(SolId::generate(), Self::default_x(n), info)
    // }
    // fn new_vec(size: usize) -> Vec<Self> {
    //     let mut v = Vec::new();
    //     v.reserve_exact(size);
    //     v
    // }
    // fn default_vec(n: usize, info: Arc<Info>, size: usize) -> Vec<Self> {
    //     let mut v = Self::new_vec(size);
    //     for _ in 0..size {
    //         v.push(Self::default(n, info.clone()));
    //     }
    //     v
    // }
    
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
pub struct FidBasePartial<SolId, Dom, Info>
where
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    pub id: SolId,
    pub x: Arc<[Dom::TypeDom]>,
    pub step: Step,
    pub fid: Option<Fidelity>,
    pub info: Arc<Info>,
}

impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Raw = Arc<[Dom::TypeDom]>;
    type Twin<B: OntoDom<Dom>> = FidBasePartial<SolId, B, Info> where Dom:OntoDom<B>;

    fn get_id(&self) -> SolId {
        self.id
    }
    
    fn get_x<T:AsRef<Self::Raw> + From<Self::Raw>>(&self) -> T {
        self.x.clone().into()
    }

    fn get_info(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<SolId, Dom, Info> Partial<SolId, Dom, Info> for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Twin<B: Domain> = FidBasePartial<SolId, B, Info>;

    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: Into<Self::Raw>
    {
        FidBasePartial {
            id,
            x: x.into(),
            step: Step::Pending,
            fid: None,
            info,
        }
    }

    fn twin<T, B>(&self, x: T) -> <Self as Partial<SolId, Dom, Info>>::Twin<B>
    where
        T: Into<<<Self as Partial<SolId, Dom, Info>>::Twin<B> as Solution<SolId, B, Info>>::Raw>,
        B: Domain
    {
        FidBasePartial {
            id:self.id,
            x: x.into(),
            step: self.step,
            fid: self.fid,
            info: self.info.clone(),
        }
    }
    
    // fn default_x(n: usize) -> Vec<TypeDom<Dom>> {
    //     vec![TypeDom::<Dom>::default(); n]
    // }

    // fn default(n: usize, info: Arc<Info>) -> Self {
    //     Self::_new(SolId::generate(), Self::default_x(n), Step::Pending,None, info)
    // }
    // fn new_vec(size: usize) -> Vec<Self> {
    //     let mut v = Vec::new();
    //     v.reserve_exact(size);
    //     v
    // }
    // fn default_vec(n: usize, info: Arc<Info>, size: usize) -> Vec<Self> {
    //     let mut v = Self::new_vec(size);
    //     for _ in 0..size {
    //         v.push(Self::default(n, info.clone()));
    //     }
    //     v
    // }
}

impl<SolId, Dom, Info> FidelityPartial<SolId, Dom, Info> for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Twin<B: Domain> = FidBasePartial<SolId, B, Info>;

    fn get_fidelity(&self) -> Option<Fidelity> {
        self.fid
    }

    fn set_fidelity(&mut self, fidelity:Option<Fidelity>) {
        self.fid=fidelity;
    }

    fn get_step(&self) -> Step {
        self.step
    }
    
    fn set_step(&mut self, step:Step) {
        self.step=step;
    }

    fn resume<B: Domain>(&mut self, twin: &mut <Self as FidelityPartial<SolId, Dom, Info>>::Twin<B>, value: f64) {
        self.fid = Some(Fidelity::Resume(value));
        twin.fid = Some(Fidelity::Resume(value));
    }

    fn discard<B: Domain>(&mut self, twin: &mut <Self as FidelityPartial<SolId, Dom, Info>>::Twin<B>) {
        self.fid = Some(Fidelity::Discard);
        twin.fid = Some(Fidelity::Discard);
    }
}
