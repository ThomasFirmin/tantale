use crate::{
    ArcVecArc, OptInfo, domain::Domain, objective::{Codomain, Outcome}, solution::{Computed, Id, Partial, SolInfo}
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;


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
    type Outc<Out:Outcome>: OutBatchType<SolId,ADom,BDom,SInfo,Info,Out>;
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
    type Outc: OutBatchType<SolId,ADom,BDom,SInfo,Info,Out>;
}

/// A [`OutBatchType`]  describes the raw [`Outcome`] linked to its [`Partial`], before getting a [`Computed`].
/// It is associated with:
///  * a [`BatchType`] describing the output of an [`Optimizer`], made of [`Partial`].
///  * a [`CompBatchType`] describing the input of an [`Optimizer`], made of [`Computed`].
pub trait OutBatchType<SolId,ADom,BDom,SInfo,Info,Out>
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

#[derive(Serialize,Deserialize)]
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
    pub sobj : ArcVecArc<Partial<SolId,ADom,SInfo>>,
    pub sopt : ArcVecArc<Partial<SolId,BDom,SInfo>>,
    pub info : Arc<Info>
}

#[derive(Serialize,Deserialize)]
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
    pub cobj : ArcVecArc<Computed<SolId,ADom,Cod,Out,SInfo>>,
    pub copt : ArcVecArc<Computed<SolId,BDom,Cod,Out,SInfo>>,
    pub info : Arc<Info>
}

#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct OutBatch<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    pub sobj : ArcVecArc<Partial<SolId,ADom,SInfo>>,
    pub sopt : ArcVecArc<Partial<SolId,BDom,SInfo>>,
    pub out: ArcVecArc<Out>,
    pub info : Arc<Info>,
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
    pub fn new(sobj: ArcVecArc<Partial<SolId, ADom, SInfo>>,sopt: ArcVecArc<Partial<SolId, BDom, SInfo>>,info: Arc<Info>)->Self{
        Batch { sobj, sopt, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        Batch { sobj: Arc::new(Vec::new()), sopt:Arc::new(Vec::new()), info }
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
    pub fn new(cobj: ArcVecArc<Computed<SolId, ADom, Cod, Out, SInfo>> , copt: ArcVecArc<Computed<SolId, BDom, Cod, Out, SInfo>> , info :Arc<Info> )->Self{
        CompBatch { cobj, copt, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        CompBatch { cobj: Arc::new(Vec::new()), copt:Arc::new(Vec::new()), info }
    }
}

impl <SolId,ADom,BDom,SInfo,Info,Out> OutBatch<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    /// Creates a new [`CompBatch`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(sobj: ArcVecArc<Partial<SolId, ADom, SInfo>> , sopt: ArcVecArc<Partial<SolId, BDom, SInfo>>, out: ArcVecArc<Out>, info :Arc<Info> )->Self{
        OutBatch { sobj, sopt, out, info }
    }

    /// Creates a new empty [`OutBatch`] from an [`OptInfo`].
    pub fn empty(info :Arc<Info> )->Self{
        OutBatch { sobj: Arc::new(Vec::new()), sopt:Arc::new(Vec::new()), out:Arc::new(Vec::new()),info }
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
    type Outc<Out:Outcome> = OutBatch<SolId,ADom,BDom,SInfo,Info,Out>;
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
    type Outc = OutBatch<SolId,ADom,BDom,SInfo,Info,Out>;
}


impl <SolId,ADom,BDom,SInfo,Info,Out> OutBatchType<SolId,ADom,BDom,SInfo,Info,Out> for OutBatch<SolId,ADom,BDom,SInfo,Info,Out>
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


//-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-//
// SINGLE BATCH MADE OF A SINGLE SOLUTION //
//-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-//


/// A [`BatchType`] describing a single [`Partial`] pair `Obj`--`Opt`.
/// Linked to its [`OptInfo`].
#[derive(Serialize,Deserialize)]
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

/// A [`CompBatchType`] describing a single [`Computed`] pair `Obj`--`Opt`.
/// Linked to its [`OptInfo`].
#[derive(Serialize,Deserialize)]
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

/// A [`OutBatchType`] describing a single [`Partial`] pair `Obj`--`Opt`.
/// Linked to its [`OptInfo`] and [`Outcome`].
#[derive(Serialize,Deserialize)]
#[serde(bound(
    serialize = "ADom::TypeDom: Serialize,BDom::TypeDom: Serialize",
    deserialize = "ADom::TypeDom: for<'a> Deserialize<'a>,BDom::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct OutSingle<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    pub sobj : Arc<Partial<SolId,ADom,SInfo>>,
    pub sopt : Arc<Partial<SolId,BDom,SInfo>>,
    pub out: Arc<Out>,
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

impl <SolId,ADom,BDom,SInfo,Info,Out> OutSingle<SolId,ADom,BDom,SInfo,Info,Out>
where
    SolId:Id,
    ADom:Domain,
    BDom:Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out:Outcome,
{
    /// Creates a new [`OutSingle`] from paired `Obj` and `Opt` [`Partial`], an [`Outcome`], and an [`OptInfo`].
    pub fn new(sobj: Arc<Partial<SolId, ADom, SInfo>> , sopt: Arc<Partial<SolId, BDom, SInfo>> , out:Arc<Out>, info :Arc<Info> )->Self{
        OutSingle { sobj, sopt, out, info }
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
    type Outc<Out:Outcome> = OutSingle<SolId,ADom,BDom,SInfo,Info,Out>;
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
    type Outc = OutSingle<SolId,ADom,BDom,SInfo,Info,Out>;
}

impl <SolId,ADom,BDom,SInfo,Info,Out> OutBatchType<SolId,ADom,BDom,SInfo,Info,Out> for OutSingle<SolId,ADom,BDom,SInfo,Info,Out>
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