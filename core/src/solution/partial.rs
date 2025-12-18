use crate::{Codomain, Computed, EvalStep, Outcome};
use crate::domain::Domain;
use crate::objective::Step;
use crate::recorder::csv::CSVWritable;
use crate::solution::{HasFidelity, HasId, HasSolInfo, HasStep, Id, IntoComputed, SolInfo, Solution, SolutionShape, Uncomputed};

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
        Shape: SolutionShape<SolId,SInfo> + HasFidelity,
        Shape::SolObj: HasFidelity,
        Shape::SolOpt: HasFidelity,
    {
        pair.get_mut_sobj().set_fidelity(self);
        pair.get_mut_sopt().set_fidelity(self);
    }
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

    fn twin<B:Domain>(&self, x: <Self::Twin<B> as Solution<SolId,B,Info>>::Raw) -> Self::Twin<B>
    {
        BasePartial { id: self.id, x: x.into(), info: self.info.clone() }
    }
}

impl<SolId, Dom, Info> Uncomputed<SolId,Dom,Info> for BasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type TwinUC<B:Domain> = BasePartial<SolId,B,Info>;
    /// Creates a new [`BasePartial`] from a slice of [`TypeDom`](Domain::TypeDom).
    ///
    /// # Attributes
    ///
    /// * `id` : `SolId` - A unique [`Id`].
    /// * `x` : `Into``<`[`Raw`](Solution::Raw)`>` - A [`Raw`](Solution::Raw) solution. For example a simple vector of [`TypeDom`](Domain::TypeDom).
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
    fn twin<B:Domain>(&self, x: <Self::TwinUC<B> as Solution<SolId,B,Info>>::Raw) -> Self::TwinUC<B>
    {
        BasePartial { id: self.id, x: x.into(), info: self.info.clone() }
    }
}

impl<SolId, Dom, Info> IntoComputed for BasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Computed<Cod:Codomain<Out>,Out:Outcome> = Computed<Self,SolId,Dom,Cod,Out,Info>;

    fn into_computed<Cod:crate::Codomain<Out>,Out:crate::Outcome>(self, y: Arc<Cod::TypeCodom>) -> Self::Computed<Cod,Out> {
        Computed::new(self, y)
    }
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
        self.step.into()
    }
    
    fn pending(&mut self) {
        self.step = EvalStep(0);
    }
    /// The value must be stricly positive.
    fn partially(&mut self, value:isize) {
        self.step = EvalStep(value);
    }
    
    fn evaluated(&mut self) {
        self.step = EvalStep(-1);
    }
    
    fn discard(&mut self) {
        self.step = EvalStep(-9);
    }

    fn error(&mut self) {
        self.step = EvalStep(-10);
    }
    
    fn raw_step(&self) -> EvalStep {
        self.step
    }
    
    fn set_step(&mut self,value:EvalStep) {
        self.step = value;
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

    fn twin<B:Domain>(&self, x: <Self::Twin<B> as Solution<SolId,B,Info>>::Raw) -> Self::Twin<B>
    {
        FidBasePartial {
            id:self.id,
            x: x.into(),
            step: self.step,
            fid: self.fid,
            info: self.info.clone(),
        }
    }
}

impl<SolId, Dom, Info> Uncomputed<SolId,Dom,Info> for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type TwinUC<B:Domain> = FidBasePartial<SolId,B,Info>;

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

    fn twin<B:Domain>(&self, x: <Self::TwinUC<B> as Solution<SolId,B,Info>>::Raw) -> Self::TwinUC<B>
    {
        FidBasePartial {
            id:self.id,
            x: x.into(),
            step: self.step,
            fid: self.fid,
            info: self.info.clone(),
        }
    }
}

impl<SolId, Dom, Info> IntoComputed for FidBasePartial<SolId, Dom, Info>
where
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Computed<Cod:Codomain<Out>,Out:Outcome> = Computed<Self,SolId,Dom,Cod,Out,Info>;

    fn into_computed<Cod:crate::Codomain<Out>,Out:crate::Outcome>(self, y: Arc<Cod::TypeCodom>) -> Self::Computed<Cod,Out> {
        Computed::new(self, y)
    }
}