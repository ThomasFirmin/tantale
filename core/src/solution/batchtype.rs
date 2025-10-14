use crate::{
    OptInfo, VecArc, domain::Domain, objective::{Codomain, Outcome}, solution::{Computed, Id, Partial, RawSol, SolInfo}
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug,sync::Arc};


pub type BatchElem<SolId,Obj,Opt,SInfo> = (Arc<Partial<SolId,Obj,SInfo>>,Arc<Partial<SolId,Opt,SInfo>>);
pub type RawBatchElem<SolId,Obj,Opt,Out,SInfo> = (Arc<RawSol<SolId,Obj,Out,SInfo>>,Arc<RawSol<SolId,Opt,Out,SInfo>>);
pub type CompBatchElem<SolId,Obj,Opt,Cod,Out,SInfo> = (Arc<Computed<SolId,Obj,Cod,Out,SInfo>>,Arc<Computed<SolId,Opt,Cod,Out,SInfo>>);

/// A [`BatchType`] describes the output of an [`Optimizer`], made of [`Partial`].
/// It is associated with:
///  * a [`CompBatchType`] describing the input of that optimizer made of [`Computed`].
///  * a [`OutBatchType`] describing the raw output of the [`Objective`] before getting a [`Computed`].
pub trait BatchType<SolId,Obj,Opt,SInfo,Info>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome>: CompBatchType<SolId,Obj,Opt,SInfo,Info,Cod,Out>;
    type Outc<Out:Outcome>: RawBatchType<SolId,Obj,Opt,SInfo,Info,Out>;
    fn get_info(&self)->Arc<Info>;
}

/// A [`CompBatchType`]  describes the input of that optimizer made of [`Computed`].
/// It is associated with:
///  * a [`BatchType`] describing the output of an [`Optimizer`], made of [`Partial`].
///  * a [`CompBatchType`] describing the input of an [`Optimizer`], made of [`Computed`].
pub trait CompBatchType<SolId,Obj,Opt,SInfo,Info,Cod,Out>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part: BatchType<SolId,Obj,Opt,SInfo,Info>;
    type Outc: RawBatchType<SolId,Obj,Opt,SInfo,Info,Out>;
}

/// A [`OutBatchType`]  describes the raw [`Outcome`] linked to its [`Partial`], before getting a [`Computed`].
/// It is associated with:
///  * a [`BatchType`] describing the output of an [`Optimizer`], made of [`Partial`].
///  * a [`CompBatchType`] describing the input of an [`Optimizer`], made of [`Computed`].
pub trait RawBatchType<SolId,Obj,Opt,SInfo,Info,Out>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    type Part: BatchType<SolId,Obj,Opt,SInfo,Info>;
    type Comp<Cod:Codomain<Out>>: CompBatchType<SolId,Obj,Opt,SInfo,Info,Cod,Out>;
}


/// A [`Batch`] describes a collection of pairs of `Obj` and `Opt` [`Partial`] stored within 2 vectors.
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Batch<SolId,Obj,Opt,SInfo,Info>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo
{
    pub sobj : VecArc<Partial<SolId,Obj,SInfo>>,
    pub sopt : VecArc<Partial<SolId,Opt,SInfo>>,
    pub info : Arc<Info>
}


/// A [`RawBatch`] describes a collection of pairs of `Obj` and `Opt` [`RawSol`] stored within 2 vectors.
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize, Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>, Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct RawBatch<SolId,Obj,Opt,SInfo,Info,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    pub robj : VecArc<RawSol<SolId,Obj,Out,SInfo>>,
    pub ropt : VecArc<RawSol<SolId,Opt,Out,SInfo>>,
    pub info : Arc<Info>,
}


/// A [`CompBatch`] describes a collection of pairs of `Obj` and `Opt` [`Computed`] stored within 2 vectors.
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub cobj : VecArc<Computed<SolId,Obj,Cod,Out,SInfo>>,
    pub copt : VecArc<Computed<SolId,Opt,Cod,Out,SInfo>>,
    pub info : Arc<Info>
}

impl <SolId,Obj,Opt,SInfo,Info> Batch<SolId,Obj,Opt,SInfo,Info>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo
{   
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(sobj: VecArc<Partial<SolId, Obj, SInfo>>,sopt: VecArc<Partial<SolId, Opt, SInfo>>,info: Arc<Info>)->Self{
        Batch { sobj, sopt, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        Batch { sobj: Vec::new(), sopt:Vec::new(), info }
    }
    
    /// Add a new `Obj` and `Opt` pair of [`Partial`] to the batch.
    pub fn add(&mut self, sobj: Arc<Partial<SolId, Obj, SInfo>>, sopt:Arc<Partial<SolId, Opt, SInfo>>){
        self.sobj.push(sobj);
        self.sopt.push(sopt);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Partial`] to the batch.
    pub fn add_vec(&mut self, sobj: VecArc<Partial<SolId, Obj, SInfo>>, sopt:VecArc<Partial<SolId, Opt, SInfo>>){
        self.sobj.extend(sobj);
        self.sopt.extend(sopt);
    }

    /// Return the size of the [`Batch`].
    pub fn size(&self)->usize{
        self.sobj.len()
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn index(&self, index:usize)->BatchElem<SolId,Obj,Opt,SInfo>{
        (self.sobj[index].clone(),self.sopt[index].clone())
    }
}

impl <SolId,Obj,Opt,SInfo,Info,Out> RawBatch<SolId,Obj,Opt,SInfo,Info,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    /// Creates a new [`OutBatch`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(robj: VecArc<RawSol<SolId, Obj, Out,SInfo>>,ropt: VecArc<RawSol<SolId, Opt, Out,SInfo>>, info :Arc<Info> )->Self{
        RawBatch { robj, ropt, info }
    }

    /// Creates a new empty [`OutBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        RawBatch { robj: Vec::new(), ropt:Vec::new(),info }
    }

    /// Add a new `Obj` and `Opt` pair of [`RawSol`] to the batch.
    pub fn add(&mut self, robj: Arc<RawSol<SolId, Obj, Out, SInfo>>, ropt:Arc<RawSol<SolId, Opt, Out, SInfo>>){
        self.robj.push(robj);
        self.ropt.push(ropt);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`RawSol`] to the batch.
    pub fn add_vec(&mut self, robj: VecArc<RawSol<SolId, Obj, Out, SInfo>>, ropt:VecArc<RawSol<SolId, Opt, Out, SInfo>>){
        self.robj.extend(robj);
        self.ropt.extend(ropt);
    }

    /// Return the size of the [`RawBatch`]
    pub fn size(&self)->usize{
        self.robj.len()
    }

    /// Return the `Obj` and `Opt` [`RawSol`] at position `index` within the batch.
    pub fn index(&self, index:usize)->RawBatchElem<SolId,Obj,Opt,Out,SInfo>{
        (self.robj[index].clone(),self.ropt[index].clone())
    }
}


impl <SolId,Obj,Opt,SInfo,Info,Cod,Out> CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    /// Creates a new [`CompBatch`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(cobj: VecArc<Computed<SolId, Obj, Cod, Out, SInfo>> , copt: VecArc<Computed<SolId, Opt, Cod, Out, SInfo>> , info :Arc<Info> )->Self{
        CompBatch { cobj, copt, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        CompBatch { cobj: Vec::new(), copt:Vec::new(), info }
    }

    /// Add a new `Obj` and `Opt` pair of [`Computed`] to the batch.
    pub fn add(&mut self, cobj: Arc<Computed<SolId, Obj, Cod, Out, SInfo>>, copt:Arc<Computed<SolId, Opt, Cod, Out, SInfo>>){
        self.cobj.push(cobj);
        self.copt.push(copt);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Computed`] to the batch.
    pub fn add_vec(&mut self, cobj: VecArc<Computed<SolId, Obj, Cod, Out, SInfo>>, copt:VecArc<Computed<SolId, Opt, Cod, Out, SInfo>>){
        self.cobj.extend(cobj);
        self.copt.extend(copt);
    }

    /// Return the size of the [`CompBatch`].
    pub fn size(&self)->usize{
        self.cobj.len()
    }

    /// Return the `Obj` and `Opt` [`Computed`] at position `index` within the batch.
    pub fn index(&self, index:usize)->CompBatchElem<SolId,Obj,Opt,Cod,Out,SInfo>{
        (self.cobj[index].clone(),self.copt[index].clone())
    }
}

impl <SolId,Obj,Opt,SInfo,Info> BatchType<SolId,Obj,Opt,SInfo,Info> for Batch<SolId,Obj,Opt,SInfo,Info>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome> = CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>;
    type Outc<Out:Outcome> = RawBatch<SolId,Obj,Opt,SInfo,Info,Out>;
    
    fn get_info(&self)->Arc<Info> {
        self.info.clone()
    }
    
}

impl <SolId,Obj,Opt,SInfo,Info,Out> RawBatchType<SolId,Obj,Opt,SInfo,Info,Out> for RawBatch<SolId,Obj,Opt,SInfo,Info,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    type Part = Batch<SolId,Obj,Opt,SInfo,Info>;
    type Comp<Cod:Codomain<Out>> = CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>;
}

impl <SolId,Obj,Opt,SInfo,Info,Cod,Out> CompBatchType<SolId,Obj,Opt,SInfo,Info,Cod,Out> for CompBatch<SolId,Obj,Opt,SInfo,Info,Cod,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part = Batch<SolId,Obj,Opt,SInfo,Info>;
    type Outc = RawBatch<SolId,Obj,Opt,SInfo,Info,Out>;
}


//-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-//
// SINGLE BATCH MADE OF A SINGLE SOLUTION //
//-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-//


/// A [`Batch`] describes a collection of pairs of `Obj` and `Opt` [`Partial`].
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Single<SolId,Obj,Opt,SInfo,Info>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub sobj : Arc<Partial<SolId,Obj,SInfo>>,
    pub sopt : Arc<Partial<SolId,Opt,SInfo>>,
    pub info : Arc<Info>
}

/// A [`RawBatch`] describes a single pair of `Obj` and `Opt` [`RawSol`].
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct RawSingle<SolId,Obj,Opt,SInfo,Info,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    pub robj : Arc<RawSol<SolId,Obj,Out,SInfo>>,
    pub ropt : Arc<RawSol<SolId,Opt,Out,SInfo>>,
    pub info : Arc<Info>
}

/// A [`CompSingle`] describes a single pair of `Obj` and `Opt` [`Computed`].
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct CompSingle<SolId,Obj,Opt,SInfo,Info,Cod,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub cobj : Arc<Computed<SolId,Obj,Cod,Out,SInfo>>,
    pub copt : Arc<Computed<SolId,Opt,Cod,Out,SInfo>>,
    pub info : Arc<Info>
}

impl <SolId,Obj,Opt,SInfo,Info> Single<SolId,Obj,Opt,SInfo,Info>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo
{   
    /// Creates a new [`Single`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(sobj: Arc<Partial<SolId, Obj, SInfo>>,sopt: Arc<Partial<SolId, Opt, SInfo>>,info: Arc<Info>)->Self{
        Single { sobj, sopt, info }
    }
}

impl <SolId,Obj,Opt,SInfo,Info,Out> RawSingle<SolId,Obj,Opt,SInfo,Info,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    /// Creates a new [`RawSingle`] from paired `Obj` and `Opt` [`Raw`] and an [`OptInfo`].
    pub fn new(robj: Arc<RawSol<SolId, Obj, Out, SInfo>> , ropt: Arc<RawSol<SolId, Opt, Out, SInfo>>, info :Arc<Info> )->Self{
        RawSingle { robj, ropt, info }
    }
}

impl <SolId,Obj,Opt,SInfo,Info,Cod,Out> CompSingle<SolId,Obj,Opt,SInfo,Info,Cod,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    /// Creates a new [`CompSingle`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(cobj: Arc<Computed<SolId, Obj, Cod, Out, SInfo>> , copt: Arc<Computed<SolId, Opt, Cod, Out, SInfo>> , info :Arc<Info> )->Self{
        CompSingle { cobj, copt, info }
    }
}

impl <SolId,Obj,Opt,SInfo,Info> BatchType<SolId,Obj,Opt,SInfo,Info> for Single<SolId,Obj,Opt,SInfo,Info>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome> = CompSingle<SolId,Obj,Opt,SInfo,Info,Cod,Out>;
    type Outc<Out:Outcome> = RawSingle<SolId,Obj,Opt,SInfo,Info,Out>;
    
    fn get_info(&self)->Arc<Info> {
        self.info.clone()
    }
}

impl <SolId,Obj,Opt,SInfo,Info,Out> RawBatchType<SolId,Obj,Opt,SInfo,Info,Out> for RawSingle<SolId,Obj,Opt,SInfo,Info,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    type Part = Single<SolId,Obj,Opt,SInfo,Info>;
    type Comp<Cod:Codomain<Out>> = CompSingle<SolId,Obj,Opt,SInfo,Info,Cod,Out>;
}

impl <SolId,Obj,Opt,SInfo,Info,Cod,Out> CompBatchType<SolId,Obj,Opt,SInfo,Info,Cod,Out> for CompSingle<SolId,Obj,Opt,SInfo,Info,Cod,Out>
where
    SolId:Id,
    Obj:Domain,
    Opt:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part = Single<SolId,Obj,Opt,SInfo,Info>;
    type Outc = RawSingle<SolId,Obj,Opt,SInfo,Info,Out>;
}