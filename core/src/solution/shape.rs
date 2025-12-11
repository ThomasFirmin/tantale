use crate::{
    Codomain, Computed, Fidelity, Id, Outcome, Partial, SolInfo, Solution, 
    domain::{Domain, onto::Linked}, 
    objective::Step, 
    solution::{HasFidelity, HasId, HasSolInfo, HasStep, HasY},
};

use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData, sync::Arc};

pub trait SolutionShape<SolId:Id, SInfo:SolInfo>: Linked 
where
    Self: Serialize + for<'a> Deserialize<'a>
{
    type SolObj: Solution<SolId, Self::Obj, SInfo>;
    type SolOpt: Solution<SolId, Self::Opt, SInfo>;

    fn get_sobj(&self)->&Self::SolObj;
    fn get_sopt(&self)->&Self::SolOpt;
    fn get_mut_sobj(&mut self)->&mut Self::SolObj;
    fn get_mut_sopt(&mut self)->&mut Self::SolOpt;
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

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> HasId<SolId> for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasId<SolId>,
    SolOpt: Solution<SolId, Opt, SInfo> + HasId<SolId>,
{
    fn get_id(&self) -> SolId {
        self.0.get_id()
    }
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> HasSolInfo<SInfo> for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasId<SolId>,
    SolOpt: Solution<SolId, Opt, SInfo> + HasId<SolId>,
{
    fn get_sinfo(&self) -> Arc<SInfo> {
        self.0.get_sinfo()
    }
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo,Cod,Out> HasY<Cod,Out> for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    Cod:Codomain<Out>,
    Out: Outcome,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasY<Cod,Out>,
    SolOpt: Solution<SolId, Opt, SInfo> + HasY<Cod,Out>,
{
    fn get_y(&self)->Arc<Cod::TypeCodom>
    {
        self.0.get_y()
    }
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> HasStep for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasStep,
    SolOpt: Solution<SolId, Opt, SInfo> + HasStep,
{
    fn step(&self) -> Step {
        self.0.step()
    }

    fn pending(&mut self) {
        self.0.pending();
        self.1.pending();
    }

    fn partially(&mut self, value:isize) {
        self.0.partially(value);
        self.1.partially(value);
    }

    fn penultimate(&mut self) {
        self.0.penultimate();
        self.1.penultimate();
    }

    fn evaluated(&mut self) {
        self.0.evaluated();
        self.1.evaluated();
    }

    fn error(&mut self) {
        self.0.error();
        self.1.error();
    }
}

impl<SolObj,SolOpt,SolId,Obj,Opt,SInfo> HasFidelity for Pair<SolObj,SolOpt,SolId,Obj,Opt,SInfo>
where
    SolId:Id,
    Obj: Domain,
    Opt: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId, Obj, SInfo> + HasFidelity,
    SolOpt: Solution<SolId, Opt, SInfo> + HasFidelity,
{
    fn fidelity(&self) ->  Fidelity {
        self.0.fidelity()
    }

    fn set_fidelity(&mut self, fidelity: Fidelity) {
        self.0.set_fidelity(fidelity);
        self.1.set_fidelity(fidelity);
    }
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
    fn get_mut_sobj(&mut self)->&mut Self::SolObj {&mut self.0}
    fn get_mut_sopt(&mut self)->&mut Self::SolOpt {&mut self.1}
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

impl<SolObj,SolId,Obj,SInfo> HasId<SolId> for Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId,Obj,SInfo> + HasId<SolId>,
{
    fn get_id(&self) -> SolId {
        self.0.get_id()
    }
}

impl<SolObj,SolId,Obj,SInfo> HasSolInfo<SInfo> for Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId,Obj,SInfo> + HasSolInfo<SInfo>,
{
    fn get_sinfo(&self) -> Arc<SInfo> {
        self.0.get_sinfo()
    }
}

impl<SolObj,SolId,Obj,SInfo, Cod, Out> HasY<Cod,Out> for Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
    SolObj: Solution<SolId,Obj,SInfo> + HasY<Cod,Out>,
{
    fn get_y(&self) -> Arc<Cod::TypeCodom> {
        self.0.get_y()
    }
}

impl<SolObj,SolId,Obj,SInfo> HasStep for Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId,Obj,SInfo> + HasStep,
{
    fn step(&self) -> Step {
        self.0.step()
    }

    fn pending(&mut self) {
        self.0.pending();
    }

    fn partially(&mut self, value:isize) {
        self.0.partially(value);
    }

    fn penultimate(&mut self) {
        self.0.penultimate();
    }

    fn evaluated(&mut self) {
        self.0.evaluated();
    }

    fn error(&mut self) {
        self.0.error();
    }
}

impl<SolObj,SolId,Obj,SInfo> HasFidelity for Lone<SolObj,SolId,Obj,SInfo>
where
    SolId:Id,
    Obj: Domain,
    SInfo:SolInfo,
    SolObj: Solution<SolId,Obj,SInfo> + HasFidelity,
{
    fn fidelity(&self) ->  Fidelity {
        self.0.fidelity()
    }

    fn set_fidelity(&mut self, fidelity: Fidelity) {
        self.0.set_fidelity(fidelity);
    }
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
    fn get_mut_sobj(&mut self)->&mut Self::SolObj {&mut self.0}
    fn get_mut_sopt(&mut self)->&mut Self::SolOpt {&mut self.0}
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