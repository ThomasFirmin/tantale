use crate::{
    OptInfo, VecArc, domain::Domain, objective::{Codomain, Outcome}, solution::{Computed, Id, Partial, RawSol, SolInfo}
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug,sync::Arc};


pub type BatchElem<SolId,ADom,BDom,SInfo> = (Arc<Partial<SolId,ADom,SInfo>>,Arc<Partial<SolId,BDom,SInfo>>);
pub type RawBatchElem<SolId,ADom,BDom,Out,SInfo> = (Arc<RawSol<SolId,ADom,Out,SInfo>>,Arc<RawSol<SolId,BDom,Out,SInfo>>);
pub type CompBatchElem<SolId,ADom,BDom,Cod,Out,SInfo> = (Arc<Computed<SolId,ADom,Cod,Out,SInfo>>,Arc<Computed<SolId,BDom,Cod,Out,SInfo>>);

/// A [`BatchType`] describes the output of an [`Optimizer`], made of [`Partial`].
/// It is associated with:
///  * a [`CompBatchType`] describing the input of that optimizer made of [`Computed`].
///  * a [`OutBatchType`] describing the raw output of the [`Objective`] before getting a [`Computed`].
pub trait BatchType<SolId,ADom,BDom,SInfo,Info>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome>: CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
    type Outc<Out:Outcome>: RawBatchType<SolId,ADom,BDom,SInfo,Info,Out>;
    fn get_info(&self)->Arc<Info>;
}

/// A [`CompBatchType`]  describes the input of that optimizer made of [`Computed`].
/// It is associated with:
///  * a [`BatchType`] describing the output of an [`Optimizer`], made of [`Partial`].
///  * a [`CompBatchType`] describing the input of an [`Optimizer`], made of [`Computed`].
pub trait CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part: BatchType<SolId,ADom,BDom,SInfo,Info>;
    type Outc: RawBatchType<SolId,ADom,BDom,SInfo,Info,Out>;
}

/// A [`OutBatchType`]  describes the raw [`Outcome`] linked to its [`Partial`], before getting a [`Computed`].
/// It is associated with:
///  * a [`BatchType`] describing the output of an [`Optimizer`], made of [`Partial`].
///  * a [`CompBatchType`] describing the input of an [`Optimizer`], made of [`Computed`].
pub trait RawBatchType<SolId,ADom,BDom,SInfo,Info,Out>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    type Part: BatchType<SolId,ADom,BDom,SInfo,Info>;
    type Comp<Cod:Codomain<Out>>: CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
}


/// A [`Batch`] describes a collection of pairs of `Obj` and `Opt` [`Partial`] stored within 2 vectors.
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Batch<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo
{
    pub sobj : VecArc<Partial<SolId,ADom,SInfo>>,
    pub sopt : VecArc<Partial<SolId,BDom,SInfo>>,
    pub info : Arc<Info>
}


/// A [`RawBatch`] describes a collection of pairs of `Obj` and `Opt` [`RawSol`] stored within 2 vectors.
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize, BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>, BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct RawBatch<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    pub robj : VecArc<RawSol<SolId,ADom,Out,SInfo>>,
    pub ropt : VecArc<RawSol<SolId,BDom,Out,SInfo>>,
    pub info : Arc<Info>,
}


/// A [`CompBatch`] describes a collection of pairs of `Obj` and `Opt` [`Computed`] stored within 2 vectors.
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct CompBatch<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub cobj : VecArc<Computed<SolId,ADom,Cod,Out,SInfo>>,
    pub copt : VecArc<Computed<SolId,BDom,Cod,Out,SInfo>>,
    pub info : Arc<Info>
}

impl <SolId,ADom,BDom,SInfo,Info> Batch<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo
{   
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(sobj: VecArc<Partial<SolId, ADom, SInfo>>,sopt: VecArc<Partial<SolId, BDom, SInfo>>,info: Arc<Info>)->Self{
        Batch { sobj, sopt, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        Batch { sobj: Vec::new(), sopt:Vec::new(), info }
    }
    
    /// Add a new `Obj` and `Opt` pair of [`Partial`] to the batch.
    pub fn add(&mut self, sobj: Arc<Partial<SolId, ADom, SInfo>>, sopt:Arc<Partial<SolId, BDom, SInfo>>){
        self.sobj.push(sobj);
        self.sopt.push(sopt);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Partial`] to the batch.
    pub fn add_vec(&mut self, sobj: VecArc<Partial<SolId, ADom, SInfo>>, sopt:VecArc<Partial<SolId, BDom, SInfo>>){
        self.sobj.extend(sobj);
        self.sopt.extend(sopt);
    }

    /// Return the size of the [`Batch`].
    pub fn size(&self)->usize{
        self.sobj.len()
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn index(&self, index:usize)->BatchElem<SolId,ADom,BDom,SInfo>{
        (self.sobj[index].clone(),self.sopt[index].clone())
    }
}

impl <SolId,ADom,BDom,SInfo,Info,Out> RawBatch<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    /// Creates a new [`OutBatch`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(robj: VecArc<RawSol<SolId, ADom, Out,SInfo>>,ropt: VecArc<RawSol<SolId, BDom, Out,SInfo>>, info :Arc<Info> )->Self{
        RawBatch { robj, ropt, info }
    }

    /// Creates a new empty [`OutBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        RawBatch { robj: Vec::new(), ropt:Vec::new(),info }
    }

    /// Add a new `Obj` and `Opt` pair of [`RawSol`] to the batch.
    pub fn add(&mut self, robj: Arc<RawSol<SolId, ADom, Out, SInfo>>, ropt:Arc<RawSol<SolId, BDom, Out, SInfo>>){
        self.robj.push(robj);
        self.ropt.push(ropt);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`RawSol`] to the batch.
    pub fn add_vec(&mut self, robj: VecArc<RawSol<SolId, ADom, Out, SInfo>>, ropt:VecArc<RawSol<SolId, BDom, Out, SInfo>>){
        self.robj.extend(robj);
        self.ropt.extend(ropt);
    }

    /// Return the size of the [`RawBatch`]
    pub fn size(&self)->usize{
        self.robj.len()
    }

    /// Return the `Obj` and `Opt` [`RawSol`] at position `index` within the batch.
    pub fn index(&self, index:usize)->RawBatchElem<SolId,ADom,BDom,Out,SInfo>{
        (self.robj[index].clone(),self.ropt[index].clone())
    }
}


impl <SolId,ADom,BDom,SInfo,Info,Cod,Out> CompBatch<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    /// Creates a new [`CompBatch`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(cobj: VecArc<Computed<SolId, ADom, Cod, Out, SInfo>> , copt: VecArc<Computed<SolId, BDom, Cod, Out, SInfo>> , info :Arc<Info> )->Self{
        CompBatch { cobj, copt, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        CompBatch { cobj: Vec::new(), copt:Vec::new(), info }
    }

    /// Add a new `Obj` and `Opt` pair of [`Computed`] to the batch.
    pub fn add(&mut self, cobj: Arc<Computed<SolId, ADom, Cod, Out, SInfo>>, copt:Arc<Computed<SolId, BDom, Cod, Out, SInfo>>){
        self.cobj.push(cobj);
        self.copt.push(copt);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Computed`] to the batch.
    pub fn add_vec(&mut self, cobj: VecArc<Computed<SolId, ADom, Cod, Out, SInfo>>, copt:VecArc<Computed<SolId, BDom, Cod, Out, SInfo>>){
        self.cobj.extend(cobj);
        self.copt.extend(copt);
    }

    /// Return the size of the [`CompBatch`].
    pub fn size(&self)->usize{
        self.cobj.len()
    }

    /// Return the `Obj` and `Opt` [`Computed`] at position `index` within the batch.
    pub fn index(&self, index:usize)->CompBatchElem<SolId,ADom,BDom,Cod,Out,SInfo>{
        (self.cobj[index].clone(),self.copt[index].clone())
    }
}

impl <SolId,ADom,BDom,SInfo,Info> BatchType<SolId,ADom,BDom,SInfo,Info> for Batch<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome> = CompBatch<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
    type Outc<Out:Outcome> = RawBatch<SolId,ADom,BDom,SInfo,Info,Out>;
    
    fn get_info(&self)->Arc<Info> {
        self.info.clone()
    }
    
}

impl <SolId,ADom,BDom,SInfo,Info,Out> RawBatchType<SolId,ADom,BDom,SInfo,Info,Out> for RawBatch<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    type Part = Batch<SolId,ADom,BDom,SInfo,Info>;
    type Comp<Cod:Codomain<Out>> = CompBatch<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
}

impl <SolId,ADom,BDom,SInfo,Info,Cod,Out> CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out> for CompBatch<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part = Batch<SolId,ADom,BDom,SInfo,Info>;
    type Outc = RawBatch<SolId,ADom,BDom,SInfo,Info,Out>;
}


//-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-//
// SINGLE BATCH MADE OF A SINGLE SOLUTION //
//-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-//


/// A [`Batch`] describes a collection of pairs of `Obj` and `Opt` [`Partial`].
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Single<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub sobj : Arc<Partial<SolId,ADom,SInfo>>,
    pub sopt : Arc<Partial<SolId,BDom,SInfo>>,
    pub info : Arc<Info>
}

/// A [`RawBatch`] describes a single pair of `Obj` and `Opt` [`RawSol`].
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct RawSingle<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    pub robj : Arc<RawSol<SolId,ADom,Out,SInfo>>,
    pub ropt : Arc<RawSol<SolId,BDom,Out,SInfo>>,
    pub info : Arc<Info>
}

/// A [`CompSingle`] describes a single pair of `Obj` and `Opt` [`Computed`].
#[derive(Serialize,Deserialize,Debug)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct CompSingle<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub cobj : Arc<Computed<SolId,ADom,Cod,Out,SInfo>>,
    pub copt : Arc<Computed<SolId,BDom,Cod,Out,SInfo>>,
    pub info : Arc<Info>
}

impl <SolId,ADom,BDom,SInfo,Info> Single<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo
{   
    /// Creates a new [`Single`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(sobj: Arc<Partial<SolId, ADom, SInfo>>,sopt: Arc<Partial<SolId, BDom, SInfo>>,info: Arc<Info>)->Self{
        Single { sobj, sopt, info }
    }
}

impl <SolId,ADom,BDom,SInfo,Info,Out> RawSingle<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    /// Creates a new [`RawSingle`] from paired `Obj` and `Opt` [`Raw`] and an [`OptInfo`].
    pub fn new(robj: Arc<RawSol<SolId, ADom, Out, SInfo>> , ropt: Arc<RawSol<SolId, BDom, Out, SInfo>>, info :Arc<Info> )->Self{
        RawSingle { robj, ropt, info }
    }
}

impl <SolId,ADom,BDom,SInfo,Info,Cod,Out> CompSingle<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    /// Creates a new [`CompSingle`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(cobj: Arc<Computed<SolId, ADom, Cod, Out, SInfo>> , copt: Arc<Computed<SolId, BDom, Cod, Out, SInfo>> , info :Arc<Info> )->Self{
        CompSingle { cobj, copt, info }
    }
}

impl <SolId,ADom,BDom,SInfo,Info> BatchType<SolId,ADom,BDom,SInfo,Info> for Single<SolId,ADom,BDom,SInfo,Info>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod:Codomain<Out>,Out:Outcome> = CompSingle<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
    type Outc<Out:Outcome> = RawSingle<SolId,ADom,BDom,SInfo,Info,Out>;
    
    fn get_info(&self)->Arc<Info> {
        self.info.clone()
    }
}

impl <SolId,ADom,BDom,SInfo,Info,Out> RawBatchType<SolId,ADom,BDom,SInfo,Info,Out> for RawSingle<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    type Part = Single<SolId,ADom,BDom,SInfo,Info>;
    type Comp<Cod:Codomain<Out>> = CompSingle<SolId,ADom,BDom,SInfo,Info,Cod,Out>;
}

impl <SolId,ADom,BDom,SInfo,Info,Cod,Out> CompBatchType<SolId,ADom,BDom,SInfo,Info,Cod,Out> for CompSingle<SolId,ADom,BDom,SInfo,Info,Cod,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    type Part = Single<SolId,ADom,BDom,SInfo,Info>;
    type Outc = RawSingle<SolId,ADom,BDom,SInfo,Info,Out>;
}