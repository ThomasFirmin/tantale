use crate::{
    Fidelity, OptInfo, domain::Domain, objective::{Codomain, Outcome}, solution::{Computed, Id, Partial, SolInfo, partial::FidelityPartial}
};
use core::slice;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, iter::Zip, marker::PhantomData, sync::Arc, vec::IntoIter};

pub type BatchElem<'a, PSolA, PSolB> = (&'a PSolA, &'a PSolB);
pub type OutBatchElem<'a, SolId, Out> = (&'a SolId, &'a Out);
pub type CompBatchElem<'a, PSolA, PSolB, SolId, Obj, Opt, Cod, Out, SInfo> = (
    &'a Computed<PSolA, SolId, Obj, Cod, Out, SInfo>,
    &'a Computed<PSolB, SolId, Opt, Cod, Out, SInfo>,
);

pub type DerBatchElem<PSolA, PSolB> = (PSolA, PSolB);
pub type DerOutBatchElem<SolId, Out> = (SolId, Out);
pub type DerCompBatchElem<PSolA, PSolB, SolId, Obj, Opt, Cod, Out, SInfo> = (
    Computed<PSolA, SolId, Obj, Cod, Out, SInfo>,
    Computed<PSolB, SolId, Opt, Cod, Out, SInfo>,
);

/// A [`BatchType`] describes the output of an [`Optimizer`], made of [`Partial`].
/// It is associated with:
///  * a [`CompBatchType`] describing the input of that optimizer made of [`Computed`].
///  * a [`OutBatchType`] describing the raw output of the [`Objective`] before getting a [`Computed`].
pub trait BatchType<SolId, Obj, Opt, SInfo, PSol, Info>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod: Codomain<Out>, Out: Outcome>: CompBatchType<
        SolId,
        Obj,
        Opt,
        SInfo,
        PSol,
        Info,
        Cod,
        Out,
    >;
    type Outc<Out: Outcome>: OutBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Out>;
    fn get_info(&self) -> Arc<Info>;
}

/// A [`CompBatchType`]  describes the input of that optimizer made of [`Computed`].
/// It is associated with:
///  * a [`BatchType`] describing the output of an [`Optimizer`], made of [`Partial`].
///  * a [`CompBatchType`] describing the input of an [`Optimizer`], made of [`Computed`].
pub trait CompBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Cod, Out>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Part: BatchType<SolId, Obj, Opt, SInfo, PSol, Info>;
    type Outc: OutBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Out>;
}

/// A [`OutBatchType`]  describes the raw [`Outcome`] linked to its [`Partial`], before getting a [`Computed`].
/// It is associated with:
///  * a [`BatchType`] describing the output of an [`Optimizer`], made of [`Partial`].
///  * a [`CompBatchType`] describing the input of an [`Optimizer`], made of [`Computed`].
pub trait OutBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Out>
where
    Self: Debug,
    SolId: Id,
    PSol: Partial<SolId, Obj, SInfo>,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out: Outcome,
{
    type Part: BatchType<SolId, Obj, Opt, SInfo, PSol, Info>;
    type Comp<Cod: Codomain<Out>>: CompBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Cod, Out>;
}

/// A [`Batch`] describes a collection of pairs of `Obj` and `Opt` [`Partial`] stored within 2 vectors.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub sobj: Vec<PSol>,
    pub sopt: Vec<PSol::Twin<Opt>>,
    pub info: Arc<Info>,
}

/// A [`OutBatch`] describes a collection of pairs of `Obj` and `Opt` [`RawSol`] stored within 2 vectors.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    pub vid: Vec<SolId>,
    pub vout: Vec<Out>,
    pub info: Arc<Info>,
}

/// A [`CompBatch`] describes a collection of pairs of `Obj` and `Opt` [`Computed`] stored within 2 vectors.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
#[allow(clippy::type_complexity)]
pub struct CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub cobj: Vec<Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
    pub copt: Vec<Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
    pub info: Arc<Info>,
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(sobj: Vec<PSol>, sopt: Vec<PSol::Twin<Opt>>, info: Arc<Info>) -> Self {
        Batch { sobj, sopt, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info: Arc<Info>) -> Self {
        Batch {
            sobj: Vec::new(),
            sopt: Vec::new(),
            info,
        }
    }

    /// Add a new `Obj` and `Opt` pair of [`Partial`] to the batch.
    pub fn add(&mut self, sobj: PSol, sopt: PSol::Twin<Opt>) {
        self.sobj.push(sobj);
        self.sopt.push(sopt);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Partial`] to the batch.
    pub fn add_vec(&mut self, sobj: Vec<PSol>, sopt: Vec<PSol::Twin<Opt>>) {
        self.sobj.extend(sobj);
        self.sopt.extend(sopt);
    }

    /// Extend [`Self`] with a new [`OutBach`].
    pub fn extend(&mut self, batch: Self) {
        self.sobj.extend(batch.sobj);
        self.sopt.extend(batch.sopt);
    }

    /// Return the size of the [`Batch`].
    pub fn size(&self) -> usize {
        self.sobj.len()
    }

    /// Return `true` if [`Batch`] is empty.
    pub fn is_empty(&self) -> bool {
        self.sobj.len() == 0
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn index(&self, index: usize) -> BatchElem<'_, PSol, PSol::Twin<Opt>> {
        (&self.sobj[index], &self.sopt[index])
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn remove(&mut self, index: usize) -> (PSol, PSol::Twin<Opt>) {
        (self.sobj.remove(index), self.sopt.remove(index))
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn pop(&mut self) -> Option<(PSol, PSol::Twin<Opt>)> {
        let xobj = self.sobj.pop();
        match xobj {
            Some(x) => Some((x, self.sopt.pop().unwrap())),
            None => None,
        }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: FidelityPartial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    /// Split the [`Batch`] made of [`FidPartial`] into three [`Batch`] according to [`Fidelity`].
    pub fn chunk_by_fid(self)->(Self,Self,Self,Self){
        let info = self.get_info();
        self.into_iter().fold(
            (
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
            ),
            |(mut d,mut r,mut n, mut dn),(sobj,sopt)|
            {
                let fid = sobj.get_fidelity();
                match fid {
                    Fidelity::New => n.add(sobj, sopt),
                    Fidelity::Resume(_) => _ = r.add(sobj, sopt),
                    Fidelity::Discard => d.add(sobj, sopt),
                    Fidelity::Done => dn.add(sobj, sopt),
                }
                (d,r,n,dn)
            }
        )
    }
}

#[allow(clippy::type_complexity)]
impl<SolId, Info, Out> OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    /// Creates a new [`OutBatch`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(vid: Vec<SolId>, vout: Vec<Out>, info: Arc<Info>) -> Self {
        OutBatch { vid, vout, info }
    }

    /// Creates a new empty [`OutBatch`] from an [`OptInfo`].
    pub fn empty(info: Arc<Info>) -> Self {
        OutBatch {
            vid: Vec::new(),
            vout: Vec::new(),
            info,
        }
    }

    /// Add a new `Obj` and `Opt` pair of [`RawSol`] to the batch.
    pub fn add(&mut self, id: SolId, out: Out) {
        self.vid.push(id);
        self.vout.push(out);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`RawSol`] to the batch.
    pub fn add_vec(&mut self, vid: Vec<SolId>, vout: Vec<Out>) {
        self.vid.extend(vid);
        self.vout.extend(vout);
    }

    /// Extend [`Self`] with a new [`OutBach`].
    pub fn extend(&mut self, batch: Self) {
        self.vid.extend(batch.vid);
        self.vout.extend(batch.vout);
    }

    /// Return the size of the [`OutBatch`]
    pub fn size(&self) -> usize {
        self.vid.len()
    }

    /// Return `true` if [`OutBatch`] is empty.
    pub fn is_empty(&self) -> bool {
        self.vid.len() == 0
    }

    /// Return the [`Id`] and [`Outcome`] at position `index` within the batch.
    pub fn index(&self, index: usize) -> OutBatchElem<'_, SolId, Out> {
        (&self.vid[index], &self.vout[index])
    }

    /// Return the [`Id`] and [`Outcome`] at position `index` within the batch.
    pub fn remove(&mut self, index: usize) -> DerOutBatchElem<SolId, Out> {
        (self.vid.remove(index), self.vout.remove(index))
    }

    /// Pop the last [`Id`] and [`Outcome`] within the batch.
    pub fn pop(&mut self) -> Option<(SolId, Out)> {
        let id = self.vid.pop();
        match id {
            Some(i) => Some((i, self.vout.pop().unwrap())),
            None => None,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
    CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    /// Creates a new [`CompBatch`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(
        cobj: Vec<Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
        copt: Vec<Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
        info: Arc<Info>,
    ) -> Self {
        CompBatch { cobj, copt, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info: Arc<Info>) -> Self {
        CompBatch {
            cobj: Vec::new(),
            copt: Vec::new(),
            info,
        }
    }

    /// Add a new `Obj` and `Opt` pair of [`Computed`] to the batch.
    pub fn add(
        &mut self,
        compobj: Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
        compopt: Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
    ) {
        self.cobj.push(compobj);
        self.copt.push(compopt);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Computed`] to the batch.
    pub fn add_vec(
        &mut self,
        compobj: Vec<Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
        compopt: Vec<Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
    ) {
        self.cobj.extend(compobj);
        self.copt.extend(compopt);
    }

    /// Extend [`Self`] with a new [`OutBach`].
    pub fn extend(&mut self, batch: Self) {
        self.cobj.extend(batch.cobj);
        self.copt.extend(batch.copt);
    }

    /// Add a new `Obj` and `Opt` pair of [`CompSol`] to the batch from a pair of [`Partial`] and a [`TypeCodom`](Codomain::TypeCodom).
    pub fn add_res(&mut self, pobj: PSol, popt: PSol::Twin<Opt>, y: Cod::TypeCodom) {
        let o = Arc::new(y);
        let compobj = Computed::new(pobj, o.clone());
        let compopt = Computed::new(popt, o);
        self.add(compobj, compopt);
    }

    /// Return the size of the [`CompBatch`].
    pub fn size(&self) -> usize {
        self.cobj.len()
    }

    /// Return `true` if [`CompBatch`] is empty.
    pub fn is_empty(&self) -> bool {
        self.cobj.len() == 0
    }

    /// Return the `Obj` and `Opt` [`Computed`] at position `index` within the batch.
    pub fn index(
        &self,
        index: usize,
    ) -> CompBatchElem<'_, PSol, PSol::Twin<Opt>, SolId, Obj, Opt, Cod, Out, SInfo> {
        (&self.cobj[index], &self.copt[index])
    }

    /// Return the `Obj` and `Opt` [`Computed`] at position `index` within the batch.
    pub fn remove(
        &mut self,
        index: usize,
    ) -> DerCompBatchElem<PSol, PSol::Twin<Opt>, SolId, Obj, Opt, Cod, Out, SInfo> {
        (self.cobj.remove(index), self.copt.remove(index))
    }

    /// Return the `Obj` and `Opt` [`Computed`] at position `index` within the batch.
    pub fn pop(
        &mut self,
    ) -> Option<DerCompBatchElem<PSol, PSol::Twin<Opt>, SolId, Obj, Opt, Cod, Out, SInfo>> {
        let xobj = self.cobj.pop();
        match xobj {
            Some(x) => Some((x, self.copt.pop().unwrap())),
            None => None,
        }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> BatchType<SolId, Obj, Opt, SInfo, PSol, Info>
    for Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod: Codomain<Out>, Out: Outcome> =
        CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>;
    type Outc<Out: Outcome> = OutBatch<SolId, Info, Out>;

    fn get_info(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, Out> OutBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Out>
    for OutBatch<SolId, Info, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out: Outcome,
{
    type Part = Batch<PSol, SolId, Obj, Opt, SInfo, Info>;
    type Comp<Cod: Codomain<Out>> = CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>;
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
    CompBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Cod, Out>
    for CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Part = Batch<PSol, SolId, Obj, Opt, SInfo, Info>;
    type Outc = OutBatch<SolId, Info, Out>;
}

//--------------------//
//--- INTOITERATOR ---//
//--------------------//

impl<PSol, SolId, Obj, Opt, SInfo, Info> IntoIterator for Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Item = (PSol, PSol::Twin<Opt>);
    type IntoIter = Zip<IntoIter<PSol>, IntoIter<PSol::Twin<Opt>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.sobj.into_iter().zip(self.sopt)
    }
}

impl<'a, PSol, SolId, Obj, Opt, SInfo, Info> IntoIterator
    for &'a Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Item = (&'a PSol, &'a PSol::Twin<Opt>);
    type IntoIter = Zip<slice::Iter<'a, PSol>, slice::Iter<'a, PSol::Twin<Opt>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.sobj.iter().zip(self.sopt.iter())
    }
}

impl<'a, PSol, SolId, Obj, Opt, SInfo, Info> IntoIterator
    for &'a mut Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Item = (&'a mut PSol, &'a mut PSol::Twin<Opt>);
    type IntoIter = Zip<slice::IterMut<'a, PSol>, slice::IterMut<'a, PSol::Twin<Opt>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.sobj.iter_mut().zip(self.sopt.iter_mut())
    }
}

impl<SolId, Info, Out> IntoIterator for OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    type Item = (SolId, Out);
    type IntoIter = Zip<IntoIter<SolId>, IntoIter<Out>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vid.into_iter().zip(self.vout)
    }
}

impl<'a, SolId, Info, Out> IntoIterator for &'a OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    type Item = (&'a SolId, &'a Out);
    type IntoIter = Zip<slice::Iter<'a, SolId>, slice::Iter<'a, Out>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vid.iter().zip(self.vout.iter())
    }
}

impl<'a, SolId, Info, Out> IntoIterator for &'a mut OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    type Item = (&'a mut SolId, &'a mut Out);
    type IntoIter = Zip<slice::IterMut<'a, SolId>, slice::IterMut<'a, Out>>;

    fn into_iter(self) -> Self::IntoIter {
        self.vid.iter_mut().zip(self.vout.iter_mut())
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoIterator
    for CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Item = (
        Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
        Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
    );
    type IntoIter = Zip<
        IntoIter<Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
        IntoIter<Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.cobj.into_iter().zip(self.copt)
    }
}

impl<'a, PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoIterator
    for &'a CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Item = (
        &'a Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
        &'a Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
    );
    type IntoIter = Zip<
        slice::Iter<'a, Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
        slice::Iter<'a, Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.cobj.iter().zip(self.copt.iter())
    }
}

impl<'a, PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoIterator
    for &'a mut CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Item = (
        &'a mut Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
        &'a mut Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
    );
    type IntoIter = Zip<
        slice::IterMut<'a, Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
        slice::IterMut<'a, Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.cobj.iter_mut().zip(self.copt.iter_mut())
    }
}

//-----------------------//
//--- INTOPARITERATOR ---//
//-----------------------//

impl<PSol, SolId, Obj, Opt, SInfo, Info> IntoParallelIterator
    for Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo> + Send,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo> + Send,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Item = (PSol, PSol::Twin<Opt>);
    type Iter = rayon::iter::Zip<
        rayon::vec::IntoIter<PSol>,
        rayon::vec::IntoIter<<PSol as Partial<SolId, Obj, SInfo>>::Twin<Opt>>,
    >;

    fn into_par_iter(self) -> Self::Iter {
        self.sobj.into_par_iter().zip(self.sopt)
    }
}

impl<'a, PSol, SolId, Obj, Opt, SInfo, Info> IntoParallelIterator
    for &'a Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo> + Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo> + Send + Sync,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Info: OptInfo,
{
    type Item = (&'a PSol, &'a PSol::Twin<Opt>);
    type Iter = rayon::iter::Zip<
        rayon::slice::Iter<'a, PSol>,
        rayon::slice::Iter<'a, <PSol as Partial<SolId, Obj, SInfo>>::Twin<Opt>>,
    >;

    fn into_par_iter(self) -> Self::Iter {
        let a = <&[_]>::into_par_iter(&self.sobj);
        let b = <&[_]>::into_par_iter(&self.sopt);
        a.zip(b)
    }
}

impl<'a, PSol, SolId, Obj, Opt, SInfo, Info> IntoParallelIterator
    for &'a mut Batch<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo> + Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo> + Send + Sync,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Info: OptInfo,
{
    type Item = (&'a mut PSol, &'a mut PSol::Twin<Opt>);
    type Iter = rayon::iter::Zip<
        rayon::slice::IterMut<'a, PSol>,
        rayon::slice::IterMut<'a, <PSol as Partial<SolId, Obj, SInfo>>::Twin<Opt>>,
    >;

    fn into_par_iter(self) -> Self::Iter {
        let a = <&mut [_]>::into_par_iter(&mut self.sobj);
        let b = <&mut [_]>::into_par_iter(&mut self.sopt);
        a.zip(b)
    }
}

impl<SolId, Info, Out> IntoParallelIterator for OutBatch<SolId, Info, Out>
where
    SolId: Id + Send,
    Info: OptInfo,
    Out: Outcome + Send,
{
    type Item = (SolId, Out);
    type Iter = rayon::iter::Zip<rayon::vec::IntoIter<SolId>, rayon::vec::IntoIter<Out>>;

    fn into_par_iter(self) -> Self::Iter {
        self.vid.into_par_iter().zip(self.vout)
    }
}

impl<'a, SolId, Info, Out> IntoParallelIterator for &'a OutBatch<SolId, Info, Out>
where
    SolId: Id + Send + Sync,
    Info: OptInfo,
    Out: Outcome + Send + Sync,
{
    type Item = (&'a SolId, &'a Out);
    type Iter = rayon::iter::Zip<rayon::slice::Iter<'a, SolId>, rayon::slice::Iter<'a, Out>>;

    fn into_par_iter(self) -> Self::Iter {
        let a = <&[_]>::into_par_iter(&self.vid);
        let b = <&[_]>::into_par_iter(&self.vout);
        a.zip(b)
    }
}

impl<'a, SolId, Info, Out> IntoParallelIterator for &'a mut OutBatch<SolId, Info, Out>
where
    SolId: Id + Send + Sync,
    Info: OptInfo,
    Out: Outcome + Send + Sync,
{
    type Item = (&'a mut SolId, &'a mut Out);
    type Iter = rayon::iter::Zip<rayon::slice::IterMut<'a, SolId>, rayon::slice::IterMut<'a, Out>>;

    fn into_par_iter(self) -> Self::Iter {
        let a = <&mut [_]>::into_par_iter(&mut self.vid);
        let b = <&mut [_]>::into_par_iter(&mut self.vout);
        a.zip(b)
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoParallelIterator
    for CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo> + Send,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo> + Send,
    SolId: Id + Send,
    Obj: Domain + Send,
    Opt: Domain + Send,
    SInfo: SolInfo + Send,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome,
{
    type Item = (
        Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
        Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
    );
    type Iter = rayon::iter::Zip<
        rayon::vec::IntoIter<Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
        rayon::vec::IntoIter<Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
    >;

    fn into_par_iter(self) -> Self::Iter {
        self.cobj.into_par_iter().zip(self.copt)
    }
}

impl<'a, PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoParallelIterator
    for &'a CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo> + Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo> + Send + Sync,
    SolId: Id + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome,
{
    type Item = (
        &'a Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
        &'a Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
    );
    type Iter = rayon::iter::Zip<
        rayon::slice::Iter<'a, Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
        rayon::slice::Iter<'a, Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
    >;

    fn into_par_iter(self) -> Self::Iter {
        let a = <&[_]>::into_par_iter(&self.cobj);
        let b = <&[_]>::into_par_iter(&self.copt);
        a.zip(b)
    }
}

impl<'a, PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoParallelIterator
    for &'a mut CompBatch<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo> + Send + Sync,
    PSol::Twin<Opt>: Partial<SolId, Opt, SInfo> + Send + Sync,
    SolId: Id + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome,
{
    type Item = (
        &'a mut Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
        &'a mut Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
    );
    type Iter = rayon::iter::Zip<
        rayon::slice::IterMut<'a, Computed<PSol, SolId, Obj, Cod, Out, SInfo>>,
        rayon::slice::IterMut<'a, Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>>,
    >;

    fn into_par_iter(self) -> Self::Iter {
        let a = <&mut [_]>::into_par_iter(&mut self.cobj);
        let b = <&mut [_]>::into_par_iter(&mut self.copt);
        a.zip(b)
    }
}

//-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-//
// SINGLE BATCH MADE OF A SINGLE SOLUTION //
//-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-//

/// A [`Batch`] describes a collection of pairs of `Obj` and `Opt` [`Partial`].
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Single<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub sobj: PSol,
    pub sopt: PSol::Twin<Opt>,
    pub info: Arc<Info>,
    _id: PhantomData<SolId>,
    _obj: PhantomData<Obj>,
    _opt: PhantomData<Opt>,
    _sinfo: PhantomData<SInfo>,
}

/// A [`OutBatch`] describes a single pair of `Obj` and `Opt` [`RawSol`].
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct RawSingle<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    pub id: SolId,
    pub out: Out,
    pub info: Arc<Info>,
}

/// A [`CompSingle`] describes a single pair of `Obj` and `Opt` [`Computed`].
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
#[allow(clippy::type_complexity)]
pub struct CompSingle<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    pub cobj: Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
    pub copt: Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
    pub info: Arc<Info>,
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> Single<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    /// Creates a new [`Single`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(sobj: PSol, sopt: PSol::Twin<Opt>, info: Arc<Info>) -> Self {
        Single {
            sobj,
            sopt,
            info,
            _id: PhantomData,
            _obj: PhantomData,
            _opt: PhantomData,
            _sinfo: PhantomData,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<SolId, Info, Out> RawSingle<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    /// Creates a new [`RawSingle`] from paired `Obj` and `Opt` [`Raw`] and an [`OptInfo`].
    pub fn new(id: SolId, out: Out, info: Arc<Info>) -> Self {
        RawSingle { id, out, info }
    }
}

#[allow(clippy::type_complexity)]
impl<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
    CompSingle<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    /// Creates a new [`CompSingle`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(
        cobj: Computed<PSol, SolId, Obj, Cod, Out, SInfo>,
        copt: Computed<PSol::Twin<Opt>, SolId, Opt, Cod, Out, SInfo>,
        info: Arc<Info>,
    ) -> Self {
        CompSingle { cobj, copt, info }
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info> BatchType<SolId, Obj, Opt, SInfo, PSol, Info>
    for Single<PSol, SolId, Obj, Opt, SInfo, Info>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Comp<Cod: Codomain<Out>, Out: Outcome> =
        CompSingle<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>;
    type Outc<Out: Outcome> = RawSingle<SolId, Info, Out>;

    fn get_info(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, Out> OutBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Out>
    for RawSingle<SolId, Info, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Out: Outcome,
{
    type Part = Single<PSol, SolId, Obj, Opt, SInfo, Info>;
    type Comp<Cod: Codomain<Out>> = CompSingle<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>;
}

impl<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
    CompBatchType<SolId, Obj, Opt, SInfo, PSol, Info, Cod, Out>
    for CompSingle<PSol, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    PSol: Partial<SolId, Obj, SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Part = Single<PSol, SolId, Obj, Opt, SInfo, Info>;
    type Outc = RawSingle<SolId, Info, Out>;
}
