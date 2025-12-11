use crate::EvalStep;
use crate::domain::Domain;
use crate::objective::Step;
use crate::recorder::csv::CSVWritable;
use crate::solution::{HasFidelity, HasId, HasSolInfo, HasStep, Id, SolInfo, Solution, SolutionShape};

use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    sync::Arc,
};

/// Describes the fidelity of a [`FidelityPartial`], i.e. a given budget for the evaluation of a [`FidelityPartial`].
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Fidelity(f64);

impl CSVWritable<(), ()> for Fidelity {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("fidelity")])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.0.to_string()])
    }
}

impl Fidelity{
    pub fn set_pair<Shape,SolId,SInfo>(self, pair:&mut Shape)
    where
        SolId: Id,
        SInfo: SolInfo,
        Shape: SolutionShape<SolId,SInfo>,
        Shape::SolObj: FidelityPartial<SolId,Shape::Obj,SInfo>,
        Shape::SolOpt: FidelityPartial<SolId,Shape::Opt,SInfo>
    {
        pair.get_mut_sobj().set_fidelity(self);
        pair.get_mut_sopt().set_fidelity(self);
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
    type TwinP<B: Domain>: Partial<SolId, B, Info, Twin<Dom> =  Self, TwinP<Dom> = Self>;

    /// Creates a new [`Partial`] from a slice of [`TypeDom<Dom>`].
    ///
    /// # Attributes
    ///
    /// * `id` : `SolId` - A unique [`Id`].
    /// * `x` : [`Arc`]`<`[`TypeDom`](Domain::TypeDom)`>` - A basic solution from the [`Searchspace`](crate::Searchspace).
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
        
    fn twin<B:Domain>(&self, x: <Self::Twin<B> as Solution<SolId,B,Info>>::Raw) -> Self::TwinP<B>;
}

/// Describes a [`Partial`] associated to a [`Fidelity`].
pub trait FidelityPartial<SolId, Dom, Info>: Partial<SolId, Dom, Info> + HasStep + HasFidelity
where
    Self: Sized + Serialize + for<'a> Deserialize<'a> + Debug,
    SolId: Id,
    Dom: Domain,
    Info: SolInfo,
{
    type TwinF<B: Domain>: FidelityPartial<SolId, B, Info, TwinF<Dom> = Self>;
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

impl<SolId, Dom, Info> HasId<SolId> for BasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn get_id(&self) -> SolId {
        self.id
    }
}

impl<SolId, Dom, Info> HasSolInfo<Info> for BasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn get_sinfo(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for BasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Raw = Arc<[Dom::TypeDom]>;
    type Twin<B: Domain>= BasePartial<SolId, B, Info>;

    fn get_x(&self) -> Self::Raw {
        self.x.clone()
    }
}

impl<SolId, Dom, Info> Partial<SolId, Dom, Info> for BasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type TwinP<B: Domain> = BasePartial<SolId, B, Info>;

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
    
    fn twin<B:Domain>(&self, x: <Self::Twin<B> as Solution<SolId,B,Info>>::Raw) -> Self::TwinP<B>
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

//--------------------//
//----- FIDELITY -----//
//--------------------//

/// A non-evaluated [`Solution`] containing a [`Fidelity`].
///
/// # Attributes
/// * `id` : [`Id`] - The unique [`Id`] of a [`Solution`].
/// * `x` : [`Arc`]`<[Dom::`[`TypeDom`](Domain::TypeDom)`]>` - A vector of [`TypeDom`](Domain::TypeDom).
/// * `step` : [`Step`] -  The current evaluation [`Step`] of `x`.
/// * `fid` : [`Fidelity`] -  The [`Fidelity`] associated to `x`.
/// * `info` : `Arc<`[`SolInfo`]`>` - Information given by the [`Optimizer`](crate::Optimizer) and linked to a specific [`Solution`].
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
    pub step: EvalStep, // A `isize` [`EvalStep`] for serde and mpi communication issues.
    pub fid: Fidelity,
    pub info: Arc<Info>,
}

impl<SolId, Dom, Info> HasId<SolId> for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn get_id(&self) -> SolId {
        self.id
    }
}

impl<SolId, Dom, Info> HasSolInfo<Info> for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn get_sinfo(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<SolId, Dom, Info> HasFidelity for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn fidelity(&self) -> Fidelity {
        self.fid
    }
    fn set_fidelity(&mut self, fidelity:Fidelity) {
        self.fid=fidelity;
    }
}

impl<SolId, Dom, Info> HasStep for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    fn step(&self)->Step{
        if self.step.0 == 0 {Step::Pending}
        else if self.step.0 > 0 {Step::Partially(self.step.0)}
        else if self.step.0 == -1 {Step::Penultimate}
        else if self.step.0 == -2 {Step::Evaluated}
        else if self.step.0 == -10 {Step::Error}
        else {unimplemented!("This value ({}) for Step, is not implemented",self.step)}
    }
    
    fn pending(&mut self) {
        self.step = EvalStep(0);
    }
    /// The value must be stricly positive.
    fn partially(&mut self, value:isize) {
        self.step = EvalStep(value);
    }
    
    fn penultimate(&mut self) {
        self.step = EvalStep(-1);
    }
    
    fn evaluated(&mut self) {
        self.step = EvalStep(-2);
    }
    
    fn error(&mut self) {
        self.step = EvalStep(-10);
    }
}

impl<SolId, Dom, Info> Solution<SolId, Dom, Info> for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Raw = Arc<[Dom::TypeDom]>;
    type Twin<B: Domain> = FidBasePartial<SolId, B, Info>;
    
    fn get_x(&self) -> Self::Raw {
        self.x.clone()
    }
}

impl<SolId, Dom, Info> Partial<SolId, Dom, Info> for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type TwinP<B: Domain> = FidBasePartial<SolId, B, Info>;

    fn new<T>(id: SolId, x: T, info: Arc<Info>) -> Self
    where
        T: Into<Self::Raw>
    {
        FidBasePartial {
            id,
            x: x.into(),
            step: EvalStep::pending(),
            fid: Fidelity(0.0),
            info,
        }
    }

    fn twin<B:Domain>(&self, x: <Self::Twin<B> as Solution<SolId,B,Info>>::Raw) -> Self::TwinP<B>
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
    type TwinF<B: Domain> = FidBasePartial<SolId, B, Info>;

}
