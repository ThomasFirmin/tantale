use crate::{
    Fidelity, OptInfo,
    domain::{Domain, onto::{OntoDom, Paired, TwinDom}}, 
    objective::{Codomain, Outcome, Step}, 
    solution::{CompPair, Computed, Id, Pair, Partial, SolInfo, partial::FidelityPartial}
};
use core::slice;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, iter::Zip, sync::Arc, vec::IntoIter};

#[cfg(feature ="mpi")]
use crate::experiment::mpi::utils::PriorityList;
#[cfg(feature ="mpi")]
use std::collections::HashMap;
#[cfg(feature ="mpi")]
use mpi::Rank;

/// A [`Batch`] describes a collection of pairs of `Obj` and `Opt` [`Partial`] stored within 2 vectors.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Obj::TypeDom: Serialize,Opt::TypeDom: Serialize",
    deserialize = "Obj::TypeDom: for<'a> Deserialize<'a>,Opt::TypeDom: for<'a> Deserialize<'a>",
))]
pub struct Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    pub pair: Vec<Pair<P,SolId,Obj,Opt,SInfo>>,
    pub info: Arc<Info>,
}

/// A [`OutBatch`] describes a collection of pairs of `Obj` and `Opt` [`RawSol`] stored within 2 vectors.
#[derive(Debug)]
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
pub struct CompBatch<P, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod:Codomain<Out>,
    Out:Outcome,
{
    pub pair: Vec<Pair<Computed<P,SolId,Opt,Cod,Out,SInfo>,SolId,Obj,Opt,SInfo>>,
    pub info: Arc<Info>,
}

impl<P, SolId, Obj, Opt, SInfo, Info> Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(pair: Vec<Pair<P,SolId,Obj,Opt,SInfo>>, info: Arc<Info>) -> Self {
        Batch { pair, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info: Arc<Info>) -> Self {
        Batch {
            pair: Vec::new(),
            info,
        }
    }

    /// Add a new `Obj` and `Opt` pair of [`Partial`] to the batch.
    pub fn add(&mut self, pair: Pair<P,SolId,Obj,Opt,SInfo>) {
        self.pair.push(pair);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Partial`] to the batch.
    pub fn add_vec(&mut self, pair:Vec<Pair<P,SolId,Obj,Opt,SInfo>>) {
        self.pair.extend(pair);
    }

    /// Extend [`Self`] with a new [`OutBach`].
    pub fn extend(&mut self, batch: Self) {
        self.pair.extend(batch.pair);
    }

    /// Return the size of the [`Batch`].
    pub fn size(&self) -> usize {
        self.pair.len()
    }

    /// Return `true` if [`Batch`] is empty.
    pub fn is_empty(&self) -> bool {
        self.pair.len() == 0
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn index(&self, index: usize) -> &'_ Pair<P,SolId,Obj,Opt,SInfo> {
        &self.pair[index]
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn remove(&mut self, index: usize) -> Pair<P,SolId,Obj,Opt,SInfo> {
        self.pair.remove(index)
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn pop(&mut self) -> Option<Pair<P,SolId,Obj,Opt,SInfo>> {
        self.pair.pop()
    }
}

impl<P, SolId, Obj, Opt, SInfo, Info> Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: FidelityPartial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    /// Split the [`Batch`] made of [`FidPartial`] into four [`Batch`] according to [`Step`].
    /// * 1st [`Error`](Step::Error)
    /// * 2nd [`Evaluated`](Step::Evaluated)
    /// * 3rd [`Penultimate`](Step::Penultimate)
    /// * 4th [`Partially`](Step::Partially)
    /// * 5th [`Pending`](Step::Pending)
    /// * 6th [`Other`](Step::Other)
    pub fn chunk_by_step(self)->(Self,Self,Self,Self,Self,Self){
        let info = self.get_info();
        self.into_iter().fold(
            (
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
            ),
            |(mut er,mut ev,mut pe,mut pa,mut pen,mut ot),pair|
            {
                match pair.0.get_step() {
                    Step::Pending => pen.add(pair),
                    Step::Partially(_) => pa.add(pair),
                    Step::Penultimate => pe.add(pair),
                    Step::Evaluated => ev.add(pair),
                    Step::Error => er.add(pair),
                    Step::Other(_) => ot.add(pair),
                }
                (er,ev,pe,pa,pen,ot)
            }
        )
    }

    /// Split the [`Batch`] made of [`FidPartial`] into four [`Batch`] according to [`Step`].
    /// * 1st [`Discard`](Fidelity::Discard)
    /// * 2nd [`Resume`](Fidelity::Resume)
    /// * 3rd [`None`]
    pub fn chunk_by_fidelity(self)->(Self,Self,Self){
        let info = self.get_info();
        self.into_iter().fold(
            (Self::empty(info.clone()),Self::empty(info.clone()),Self::empty(info.clone())),
            |(mut d,mut r,mut n),pair|
            {
                if let Some(fid) = pair.0.get_fidelity(){
                    match fid{
                        Fidelity::Resume(_) => r.add(pair),
                        Fidelity::Discard => d.add(pair),
                    }
                }
                else{
                    n.add(pair);
                }
                (r,d,n)
            }
        )
    }

    #[cfg(feature = "mpi")]
    /// Split the [`Batch`] made of [`FidPartial`] into 3 [`PriorityList`] of size `size`,
    /// and a [`Batch`], according to [`Fidelity`] and `wher_is_id` describing at wich rank
    /// a previously partially computed solution's state is stored.
    /// * 1st [`PriorityList`]: [`Discard`](Fidelity::New)
    /// * 2nd [`PriorityList`]: [`Resume`](Fidelity::Last)
    /// * 2nd [`PriorityList`]: [`Resume`](Fidelity::Resume)
    /// * 3rd [`Batch`]: [`New`](Fidelity::Discard)
    pub fn chunk_to_priority(
        self,
        where_is_id:&mut HashMap<SolId, Rank>,
        priority_discard: &mut PriorityList<Pair<P,SolId,Obj,Opt,SInfo>>,
        priority_last: &mut PriorityList<Pair<P,SolId,Obj,Opt,SInfo>>,
        priority_resume: &mut PriorityList<Pair<P,SolId,Obj,Opt,SInfo>>,
        new_batch: &mut Self,
    )
    {
        self.into_iter().for_each(
            |pair|
            {
                match pair.0.get_fidelity() {
                    Fidelity::New => new_batch.add(pair),
                    Fidelity::Last => {
                        let rank = where_is_id.remove(&pair.0.get_id()).unwrap();
                        priority_last.add(pair,rank);
                    },
                    Fidelity::Resume(_) => {
                        let rank = where_is_id.remove(&pair.0.get_id()).unwrap();
                        priority_resume.add(pair,rank);
                    },
                    Fidelity::Discard => {
                        let rank = where_is_id.remove(&pair.0.get_id()).unwrap();
                        priority_discard.add(pair,rank);
                    },
                    Fidelity::Done => {},
                }
            }
        )
    }
}

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
    pub fn index(&self, index: usize) -> (&'_ SolId, &'_ Out) {
        (&self.vid[index], &self.vout[index])
    }

    /// Return the [`Id`] and [`Outcome`] at position `index` within the batch.
    pub fn remove(&mut self, index: usize) -> (SolId,Out) {
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

impl<P, SolId, Obj, Opt, SInfo, Info, Cod, Out> CompBatch<P, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(pair: Vec<CompPair<P,SolId,Obj,Opt,SInfo, Cod, Out>>, info: Arc<Info>) -> Self {
        CompBatch { pair, info }
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info: Arc<Info>) -> Self {
        CompBatch {
            pair: Vec::new(),
            info,
        }
    }

    /// Add a new `Obj` and `Opt` pair of [`Partial`] to the batch.
    pub fn add(&mut self, pair: CompPair<P,SolId,Obj,Opt,SInfo, Cod, Out>) {
        self.pair.push(pair);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Partial`] to the batch.
    pub fn add_vec(&mut self, pair:Vec<CompPair<P,SolId,Obj,Opt,SInfo, Cod, Out>>) {
        self.pair.extend(pair);
    }

    /// Extend [`Self`] with a new [`OutBach`].
    pub fn extend(&mut self, batch: Self) {
        self.pair.extend(batch.pair);
    }

    /// Return the size of the [`Batch`].
    pub fn size(&self) -> usize {
        self.pair.len()
    }

    /// Return `true` if [`Batch`] is empty.
    pub fn is_empty(&self) -> bool {
        self.pair.len() == 0
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn index(&self, index: usize) -> &'_ CompPair<P,SolId,Obj,Opt,SInfo, Cod, Out> {
        &self.pair[index]
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn remove(&mut self, index: usize) -> CompPair<P,SolId,Obj,Opt,SInfo, Cod, Out> {
        self.pair.remove(index)
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn pop(&mut self) -> Option<CompPair<P,SolId,Obj,Opt,SInfo, Cod, Out>> {
        self.pair.pop()
    }
}

//--------------------//
//--- INTOITERATOR ---//
//--------------------//

impl<P, SolId, Obj, Opt, SInfo, Info> IntoIterator for Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Item = Pair<P,SolId,Obj,Opt,SInfo>;
    type IntoIter = IntoIter<Pair<P,SolId,Obj,Opt,SInfo>>;

    fn into_iter(self) -> Self::IntoIter {
        self.pair.into_iter()
    }
}

impl<'a, P, SolId, Obj, Opt, SInfo, Info> IntoIterator
    for &'a Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Item = &'a Pair<P,SolId,Obj,Opt,SInfo>;
    type IntoIter = slice::Iter<'a, Pair<P,SolId,Obj,Opt,SInfo>>;

    fn into_iter(self) -> Self::IntoIter {
        self.pair.iter()
    }
}

impl<'a, P, SolId, Obj, Opt, SInfo, Info> IntoIterator
    for &'a mut Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    type Item = &'a mut Pair<P,SolId,Obj,Opt,SInfo>;
    type IntoIter = slice::IterMut<'a, Pair<P,SolId,Obj,Opt,SInfo>>;

    fn into_iter(self) -> Self::IntoIter {
        self.pair.iter_mut()
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

impl<P, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoIterator
    for CompBatch<P, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Item = CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>;
    type IntoIter = IntoIter<CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>>;

    fn into_iter(self) -> Self::IntoIter {
        self.pair.into_iter()
    }
}

impl<'a, P, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoIterator
    for &'a CompBatch<P, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Item = &'a CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>;
    type IntoIter = slice::Iter<'a, CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>>;

    fn into_iter(self) -> Self::IntoIter {
        self.cobj.iter().zip(self.copt.iter())
    }
}

impl<'a, P, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoIterator
    for &'a mut CompBatch<P, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Out: Outcome,
{
    type Item = &'a mut CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>;
    type IntoIter = slice::IterMut<'a, CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>>;

    fn into_iter(self) -> Self::IntoIter {
        self.pair.iter_mut()
    }
}

//-----------------------//
//--- INTOPARITERATOR ---//
//-----------------------//

impl<P, SolId, Obj, Opt, SInfo, Info> IntoParallelIterator
    for Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo> + Send,
    P::Twin<Obj>: Send,
    SolId: Id + Send,
    Obj: Domain + Send,
    Opt: Domain + Send,
    SInfo: SolInfo + Send,
    Info: OptInfo,
{
    type Item = Pair<P,SolId,Obj,Opt,SInfo>;
    type Iter = rayon::vec::IntoIter<Pair<P,SolId,Obj,Opt,SInfo>>;

    fn into_par_iter(self) -> Self::Iter {
        self.pair.into_par_iter()
    }
}

impl<'a, P, SolId, Obj, Opt, SInfo, Info> IntoParallelIterator
    for &'a Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo> + Send + Sync,
    P::Twin<Obj>: Send + Sync,
    SolId: Id + Send,
    Obj: Domain + Send,
    Opt: Domain + Send,
    SInfo: SolInfo + Send,
    Info: OptInfo,
{
    type Item = &'a Pair<P,SolId,Obj,Opt,SInfo>;
    type Iter = rayon::slice::Iter<'a, Pair<P,SolId,Obj,Opt,SInfo>>;

    fn into_par_iter(self) -> Self::Iter {
        <&[_]>::into_par_iter(&self.pair);
    }
}

impl<'a, P, SolId, Obj, Opt, SInfo, Info> IntoParallelIterator
    for &'a mut Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo> + Send + Sync,
    P::Twin<Obj>: Send + Sync,
    SolId: Id + Send,
    Obj: Domain + Send,
    Opt: Domain + Send,
    SInfo: SolInfo + Send,
    Info: OptInfo,
{
    type Item = &'a mut Pair<P,SolId,Obj,Opt,SInfo>;
    type Iter = rayon::slice::IterMut<'a, Pair<P,SolId,Obj,Opt,SInfo>>;

    fn into_par_iter(self) -> Self::Iter {
        <&mut [_]>::into_par_iter(&mut self.pair);
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

impl<P, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoParallelIterator
    for CompBatch<P, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    P: Partial<SolId,Opt,SInfo> + Send,
    P::TwinP<Obj>: Send,
    SolId: Id + Send,
    Obj: Domain + Send,
    Opt: Domain + Send,
    SInfo: SolInfo + Send,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome,
{
    type Item = CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>;
    type Iter = rayon::vec::IntoIter<CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>>;

    fn into_par_iter(self) -> Self::Iter {
        self.pair.into_par_iter()
    }
}

impl<'a, P, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoParallelIterator
    for &'a CompBatch<P, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    P: Partial<SolId,Opt,SInfo> + Send + Sync,
    P::TwinP<Obj>: Send + Sync,
    SolId: Id + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome,
{
    type Item = &'a CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>;
    type Iter = rayon::slice::Iter<'a, CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>>;

    fn into_par_iter(self) -> Self::Iter {
        self.pair.par_iter()
    }
}

impl<'a, P, SolId, Obj, Opt, SInfo, Info, Cod, Out> IntoParallelIterator
    for &'a mut CompBatch<P, SolId, Obj, Opt, SInfo, Info, Cod, Out>
where
    P: Partial<SolId,Opt,SInfo> + Send + Sync,
    P::TwinP<Obj>: Send + Sync,
    SolId: Id + Send + Sync,
    Obj: Domain + Send + Sync,
    Opt: Domain + Send + Sync,
    SInfo: SolInfo + Send + Sync,
    Info: OptInfo,
    Cod: Codomain<Out>,
    Cod::TypeCodom: Send + Sync,
    Out: Outcome,
{
    type Item = &'a mut CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>;
    type Iter = rayon::slice::IterMut<'a, CompPair<P, SolId, Obj, Opt, SInfo,Cod,Out>>;

    fn into_par_iter(self) -> Self::Iter {
        self.pair.iter_mut()
    }
}


/// Convert a [`Vec`] of pair of `Obj` and `Opt` [`Partial`].
/// 
/// # Notes
/// 
/// The [`OptInfo`] is set to default.
impl<P, SolId, Obj, Opt, SInfo, Info> FromIterator<Pair<P,SolId,Obj,Opt,SInfo>> for Batch<P, SolId, Obj, Opt, SInfo, Info>
where
    P: Partial<SolId,Opt,SInfo>,
    SolId: Id,
    Obj: Domain,
    Opt: Domain,
    SInfo: SolInfo,
    Info: OptInfo,
{
    fn from_iter<T: IntoIterator<Item = Pair<P,SolId,Obj,Opt,SInfo>>>(iter: T) -> Self {
        Self::new(iter.into_iter(), Arc::new(Info::default()))
    }
}