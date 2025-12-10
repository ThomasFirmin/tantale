//! A [`Solution`] defines a point sampled within a [`Searchspace`] made of a [`Vec`] of [`Var`](tantale::core::Var).
//! Thus there are two types of solutions:
//!     * A [`Solution`] from the [`Objective`](tantale::core::objective::Objective) [`Domains`](tantale::core::domain::Domain).
//!     * A [`Solution`] from the [`Optimizer`](tantale::core::optimizer::Optimizer) [`Domains`](tantale::core::domain::Domain).
//!
//! A [`Solution`] is statically typed by a [`Domain`].
//! A [`Solution`] cannot be typed by a `dyn Trait`. This is why the [`Mixed`](tantale::core::Mixed) trait is used
//! within the [`sp!`](../../../tantale/macros/macro.sp.html) macro, to create `enum` of used [`Domains`](tantale::core::Domain) and
//! and `enum` of [`TypeDom`](tantale::core::Domain::TypeDom).
//!
//! A [`Solution`] is made of a unique `id`, an [`Arc`]`<[`[`TypeDom`](tantale::core::Domain::TypeDom)`]>` and a [`SolInfo`](tantale::core::SolInfo).
//! [`Solutions`](Solution) are strongly bounded to [`Searchspace`](tantale::core::Searchspace), from which
//! most of them will be created via simplified methods.
//! The types of [`Solution`], are then used to statically constrain
//! the [`Objective`](tantale::core::Objective) and [`Optimizer`](tantale::core::Optimizer) inputs.
//!
//! The unique `id` of a solution is computed according to a static atomic variable `SOL_ID`, and the `pid` of the process creating that solution.
//!
//! # Notes
//!
//! A [`Solution`] with an unknown [`Outcome`](tantale::core::Outcome) is [`Partial`](tantale::core::Partial).
//! When the [`Outcome`](tantale::core::Outcome) is known, a [`Partial`](tantale::core::Partial)
//! becomes a [`Computed`](tantale::core::Computed) [`Solution`], made of a [`Partial`](tantale::core::Partial),
//! and a [`Codomain`](tantale::core::Codomain).
//!
use crate::{Codomain, Outcome, domain::{Domain, onto::Linked}, objective::Step, solution::partial::FidelityPartial};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData, sync::Arc};

/// Describes single-[`Solution`] information
/// obtained after each iteration of the [`Optimizer`].
pub trait SolInfo
where
    Self: Debug + Serialize + for<'a> Deserialize<'a>,
{
}

pub type SolTwin<S, SolId, A, B, SInfo> = <S as Solution<SolId,A,SInfo>>::Twin<B>;

/// An abstract [`Solution`] made of at least a [`Domain`] and a [`SolInfo`].
pub trait Solution<SolId, Dom, Info>
where
    Self: Sized + Debug + Serialize + for<'de> Deserialize<'de>,
    Dom: Domain,
    Info: SolInfo,
    SolId: Id,
{
    type Raw: Debug + Serialize;
    type Twin<B: Domain>: Solution<SolId, B, Info, Twin<Dom> = Self>;

    /// The `id` of a [`Solution`] is made of a `pid` and a unique number.
    /// This `id` can be shared with a twin [`Solution`].
    fn get_id(&self) -> SolId;

    /// Returns the actual sampled point from the set of [`Domain`].
    fn get_x(&self) -> Self::Raw;

    /// Returns the [`SolInfo`] bounded to this [`Solution`].
    fn get_info(&self) -> Arc<Info>;

    /// Checks if two [`Solution`] are twins.
    /// Twins [`Solution`] share equal ids.
    fn is_twin<Twin, B>(&self, solb: Twin) -> bool
    where
        B: Domain,
        Twin: Solution<SolId, B, Info>,
    {
        self.get_id() == solb.get_id()
    }
}

pub type SolRaw<S, SolId, Dom, Info> = <S as Solution<SolId,Dom,Info>>::Raw;

  //-------------------------//
 //--- PAIR OF SOLUTIONS ---//
//-------------------------//

pub trait SolutionShape<SolId:Id, SInfo:SolInfo>: Linked 
where
    Self: Serialize + for<'a> Deserialize<'a>
{
    type SolObj: Solution<SolId, Self::Obj, SInfo>;
    type SolOpt: Solution<SolId, Self::Opt, SInfo>;

    fn get_sobj(&self)->&Self::SolObj;
    fn get_sopt(&self)->&Self::SolOpt;
    fn get_id(&self)->SolId;
    fn get_info(&self)->Arc<SInfo>;
}

/// Creates a [`SolutionShape`] of [`Computed`] [`Solutions`](Solution) from a [`SolutionShape`] of [`Partial`]
pub trait IntoComputed<SolId, SInfo, Cod, Out>: SolutionShape<SolId, SInfo>
where
    SolId: Id,
    SInfo: SolInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
    Self::SolObj: Partial<SolId,Self::Obj,SInfo>,
    Self::SolOpt: Partial<SolId,Self::Opt,SInfo>,
{
    type CompShape: SolutionShape<SolId,SInfo,SolObj = Computed<Self::SolObj,SolId,Self::Obj,Cod,Out,SInfo>,SolOpt = Computed<Self::SolOpt,SolId,Self::Opt,Cod,Out,SInfo>,Obj=Self::Obj,Opt=Self::Opt>;
    fn into_computed(self, y: Arc<Cod::TypeCodom>) -> Self::CompShape;
}


/// A pair made of a `Obj` [`Solution`] and its `Opt` [`Twin`](Solution::Twin).
#[derive(Debug,Serialize,Deserialize)]
#[serde(bound(
    serialize = "SolObj: Serialize, SolOpt: Serialize",
    deserialize = "SolObj: for<'a> Deserialize<'a>, SolOpt: for<'a> Deserialize<'a>",
))]
pub struct Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>(SolObj,SolOpt,PhantomData<SolId>,PhantomData<Obj>,PhantomData<Opt>,PhantomData<SInfo>)
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>;

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    pub fn new(solobj:SolObj,solopt:SolOpt)->Self{Self(solobj,solopt,PhantomData,PhantomData,PhantomData,PhantomData)}
    
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> Linked for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    type Obj = Obj;
    type Opt = Opt;
}


impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo, Cod, Out> Pair<Computed<SolObj,SolId,Obj,Cod,Out,SInfo>,Computed<SolOpt,SolId,Opt,Cod,Out,SInfo>,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    Cod: Codomain<Out>,
    Out:Outcome,
    SolObj: Partial<SolId, Obj, SInfo>,
    SolOpt: Partial<SolId, Opt, SInfo>,
{
    pub fn get_y(&self)->&Cod::TypeCodom
    {
        self.0.get_y()
    }
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: FidelityPartial<SolId, Obj, SInfo>,
    SolOpt: FidelityPartial<SolId, Opt, SInfo>,
{
    pub fn get_fidelity(&self)->Option<Fidelity>{
      self.0.get_fidelity()
    }
    pub fn get_step(&self)->Step{self.0.get_step()}
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> SolutionShape<SolId,SInfo> for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    type SolObj = SolObj;
    type SolOpt = SolOpt;

    fn get_sobj(&self)->&Self::SolObj {&self.0}
    fn get_sopt(&self)->&Self::SolOpt {&self.1}
    fn get_id(&self)->SolId{self.0.get_id()}
    fn get_info(&self)->Arc<SInfo>{self.0.get_info()}
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo,Cod,Out> IntoComputed<SolId,SInfo,Cod,Out> for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
    SolObj: Partial<SolId, Obj, SInfo>,
    SolOpt: Partial<SolId, Opt, SInfo>,
{
    type CompShape = Pair<Computed<Self::SolObj,SolId,Self::Obj,Cod,Out,SInfo>,Computed<Self::SolOpt,SolId,Self::Opt,Cod,Out,SInfo>,SolId,Self::Obj,Self::Opt,SInfo>;
    fn into_computed(self, y: Arc<Cod::TypeCodom>) -> Self::CompShape {
        (Computed::new(self.0, y.clone()),Computed::new(self.1, y)).into()
    }
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> From<(SolObj,SolOpt)> for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo>,
    SolOpt: Solution<SolId, Opt, SInfo>,
{
    fn from(value: (SolObj,SolOpt)) -> Self {
        Self(value.0,value.1,PhantomData,PhantomData,PhantomData,PhantomData)
    }
}


  //---------------------//
 //--- LONE SOLUTION ---//
//---------------------//

/// A single [`Solution`] with no link to a [`Twin`](Solution::Twin).
#[derive(Debug,Serialize,Deserialize)]
#[serde(bound(serialize = "SolObj: Serialize", deserialize = "SolObj: for<'a> Deserialize<'a>"))]
pub struct Lone<SolObj,SolId,Obj,SInfo>(SolObj,PhantomData<SolId>,PhantomData<Obj>,PhantomData<SInfo>)
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId,Obj,SInfo>;

pub type CompLone<SolObj,SolId,Obj, SInfo, Cod, Out> = Lone<Computed<SolObj,SolId,Obj,Cod,Out,SInfo>,SolId,Obj,SInfo>;

impl<SolObj,SolId,Obj,SInfo> Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId,Obj,SInfo>,
{
    pub fn new(sol:SolObj)->Self{
        Self(sol,PhantomData,PhantomData,PhantomData)
    }
}

impl<SolObj,SolId,Obj,SInfo,Cod,Out> CompLone<SolObj,SolId,Obj,SInfo,Cod,Out>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Partial<SolId,Obj,SInfo>,
    Cod: Codomain<Out>,
    Out:Outcome,
{
    pub fn get_y(&self)->&Cod::TypeCodom
    {
        self.0.get_y()
    }
}

impl<SolObj,SolId,Obj,SInfo> Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: FidelityPartial<SolId,Obj,SInfo>,
{
    pub fn get_fidelity(&self)->Option<Fidelity>{self.0.get_fidelity()}
    pub fn get_step(&self)->Step{self.0.get_step()}
}


impl<SolObj,SolId,Obj,SInfo> Linked for Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId,Obj,SInfo>,
{
    type Obj = Obj;
    type Opt = Obj;
}

impl<SolId, SInfo, Obj, SolObj> SolutionShape<SolId, SInfo> for Lone<SolObj, SolId, Obj, SInfo>
where
    SolId: Id,
    SInfo: SolInfo,
    Obj: Domain,
    SolObj: Solution<SolId, Obj, SInfo>,
{
    type SolObj = SolObj;
    type SolOpt = SolObj;

    fn get_sobj(&self)->&Self::SolObj {&self.0}
    fn get_sopt(&self)->&Self::SolOpt {&self.0}
    fn get_id(&self)->SolId{self.0.get_id()}
    fn get_info(&self)->Arc<SInfo>{self.0.get_info()}
}

impl<SolObj, SolId, Obj, SInfo,Cod,Out> IntoComputed<SolId,SInfo,Cod,Out> for  Lone<SolObj, SolId, Obj, SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
    SolObj: Partial<SolId, Obj, SInfo>,
{
    type CompShape = Lone<Computed<Self::SolObj,SolId,Self::Obj,Cod,Out,SInfo>,SolId,Self::Obj,SInfo>;
    fn into_computed(self, y: Arc<Cod::TypeCodom>) -> Self::CompShape {
        (Computed::new(self.0, y.clone())).into()
    }
}


impl<SolObj,SolId,Obj,SInfo> From<SolObj> for Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId,Obj,SInfo>,
{
    fn from(value: SolObj) -> Self {
        Self(value, PhantomData, PhantomData, PhantomData)
    }
}

pub mod id;
pub use id::{Id, ParSId, SId};

pub mod partial;
pub use partial::{BasePartial, FidBasePartial, Fidelity, Partial};

pub mod computed;
pub use computed::Computed;

pub mod outsol;
pub use outsol::OutSol;

pub mod batchtype;
pub use batchtype::{Batch, OutBatch};
