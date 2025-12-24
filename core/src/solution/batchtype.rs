use crate::{
    OptInfo, objective::{Outcome, Step}, solution::{ HasInfo, HasStep, Id, SolInfo, SolutionShape}
};
use core::slice;
use rayon::iter::IntoParallelIterator;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData, sync::Arc, vec::IntoIter};

#[cfg(feature ="mpi")]
use crate::experiment::mpi::utils::PriorityList;
#[cfg(feature ="mpi")]
use std::collections::HashMap;
#[cfg(feature ="mpi")]
use mpi::Rank;

/// A [`Batch`] describes a collection of pairs of `Obj` and `Opt` [`Partial`] stored within 2 vectors.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Shape: Serialize",
    deserialize = "Shape: for<'a> Deserialize<'a>",
))]
pub struct Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>
{
    pub pairs: Vec<Shape>,
    pub info: Arc<Info>,
    _id: PhantomData<SolId>,
    _sinfo: PhantomData<SInfo>,
}

impl<SolId, SInfo, Info, Shape> Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>
{
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(pairs: Vec<Shape>, info: Arc<Info>) -> Self {
        Batch { pairs, info, _id: PhantomData, _sinfo: PhantomData}
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info: Arc<Info>) -> Self {
        Batch {
            pairs: Vec::new(),
            info,
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }

    /// Add a new `Obj` and `Opt` pair of [`Partial`] to the batch.
    pub fn add(&mut self, pair: Shape) {
        self.pairs.push(pair);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Partial`] to the batch.
    pub fn add_vec(&mut self, pairs:Vec<Shape>) {
        self.pairs.extend(pairs);
    }

    /// Extend [`Self`] with a new [`OutBach`].
    pub fn extend(&mut self, batch: Self) {
        self.pairs.extend(batch.pairs);
    }

    /// Return the size of the [`Batch`].
    pub fn size(&self) -> usize {
        self.pairs.len()
    }

    /// Return `true` if [`Batch`] is empty.
    pub fn is_empty(&self) -> bool {
        self.pairs.len() == 0
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn bindex(&self, index: usize) -> &'_ Shape {
        &self.pairs[index]
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn remove(&mut self, index: usize) -> Shape {
        self.pairs.remove(index)
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn pop(&mut self) -> Option<Shape> {
        self.pairs.pop()
    }
}

impl<SolId, SInfo, Info, Shape> Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo> + HasStep,
{
    /// Split the [`Batch`] made of [`FidPartial`] into four [`Batch`] according to [`Step`].
    /// * 1st [`Error`](Step::Error)
    /// * 1st [`Discard`](Step::Discard)
    /// * 2nd [`Evaluated`](Step::Evaluated)
    /// * 4th [`Partially`](Step::Partially)
    /// * 5th [`Pending`](Step::Pending)
    pub fn chunk_by_step(self)->(Self,Self,Self,Self,Self){
        let info = self.info.clone();
        self.into_iter().fold(
            (
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
            ),
            |(mut pend,mut part, mut disc, mut eval, mut erro),pair|
            {
                match pair.step() {
                    Step::Pending => pend.add(pair),
                    Step::Partially(_) => part.add(pair),
                    Step::Evaluated => eval.add(pair),
                    Step::Discard => disc.add(pair),
                    Step::Error => erro.add(pair),
                }
                (erro,disc,eval,part,pend)
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
        priority_discard: &mut PriorityList<Shape>,
        priority_resume: &mut PriorityList<Shape>,
        new_batch: &mut Self,
    )
    {   
        self.into_iter().for_each(
            |pair|
            {
                match pair.step() {
                    Step::Pending => new_batch.add(pair),
                    Step::Partially(_) => {
                        let rank = where_is_id.remove(&pair.get_id()).unwrap();
                        priority_resume.add(pair,rank);
                    },
                    Step::Evaluated => {
                        let rank = where_is_id.remove(&pair.get_id()).unwrap();
                        priority_discard.add(pair,rank);
                    },
                    _ => {},
                }
                
            }
        )
    }
}

impl<SolId, SInfo, Info, Shape> HasInfo<Info> for Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>
{
    fn get_info(&self) -> Arc<Info> {
        self.info.clone()
    }
}

//--------------------//
//--- INTOITERATOR ---//
//--------------------//

impl<SolId, SInfo, Info, Shape> IntoIterator for Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>
{
    type Item = Shape;
    type IntoIter = IntoIter<Shape>;

    fn into_iter(self) -> Self::IntoIter {
        self.pairs.into_iter()
    }
}

impl<'a,SolId, SInfo, Info, Shape> IntoIterator for &'a Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>
{
    type Item = &'a Shape;
    type IntoIter = slice::Iter<'a, Shape>;

    fn into_iter(self) -> Self::IntoIter {
        self.pairs.iter()
    }
}

impl<'a,SolId, SInfo, Info, Shape> IntoIterator for &'a mut Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>
{
    type Item = &'a mut Shape;
    type IntoIter = slice::IterMut<'a, Shape>;

    fn into_iter(self) -> Self::IntoIter {
        self.pairs.iter_mut()
    }
}

//-----------------------//
//--- INTOPARITERATOR ---//
//-----------------------//

impl<SolId, SInfo, Info, Shape> IntoParallelIterator for Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo> + Send + Sync
{
    type Item = Shape;
    type Iter = rayon::vec::IntoIter<Shape>;

    fn into_par_iter(self) -> Self::Iter {
        self.pairs.into_par_iter()
    }
}

impl<'a, SolId, SInfo, Info, Shape> IntoParallelIterator for &'a Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo> + Send + Sync
{
    type Item = &'a Shape;
    type Iter = rayon::slice::Iter<'a, Shape>;

    fn into_par_iter(self) -> Self::Iter {
        <&[_]>::into_par_iter(&self.pairs)
    }
}

impl<'a, SolId, SInfo, Info, Shape> IntoParallelIterator for &'a mut Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo> + Send + Sync
{
    type Item = &'a mut Shape;
    type Iter = rayon::slice::IterMut<'a, Shape>;

    fn into_par_iter(self) -> Self::Iter {
        <&mut [_]>::into_par_iter(&mut self.pairs)
    }
}

/// Convert a [`Vec`] of pair of `Obj` and `Opt` [`Partial`].
/// 
/// # Notes
/// 
/// The [`OptInfo`] is set to default.
impl<SolId, SInfo, Info, Shape> FromIterator<Shape> for Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>
{
    fn from_iter<T: IntoIterator<Item = Shape>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect(), Arc::new(Info::default()))
    }
}

/// A [`OutBatch`] describes a collection of pairs of `Obj` and `Opt` [`RawSol`] stored within 2 vectors.
#[derive(Debug)]
pub struct OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    pub out: Vec<(SolId,Out)>,
    pub info: Arc<Info>,
}

impl<SolId, Info, Out> HasInfo<Info> for OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    fn get_info(&self) -> Arc<Info> {
        self.info.clone()
    }
}

impl<SolId, Info, Out> OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    /// Creates a new [`OutBatch`] from paired `Obj` and `Opt` [`Computed`] and an [`OptInfo`].
    pub fn new(out: Vec<(SolId,Out)>, info: Arc<Info>) -> Self {
        OutBatch {out, info }
    }

    /// Creates a new empty [`OutBatch`] from an [`OptInfo`].
    pub fn empty(info: Arc<Info>) -> Self {
        OutBatch {out: Vec::new(),info}
    }

    /// Add a new `Obj` and `Opt` pair of [`RawSol`] to the batch.
    pub fn add(&mut self, out: (SolId,Out)) {
        self.out.push(out);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`RawSol`] to the batch.
    pub fn add_vec(&mut self, vout: Vec<(SolId,Out)>) {
        self.out.extend(vout);
    }

    /// Extend [`Self`] with a new [`OutBach`].
    pub fn extend(&mut self, batch: Self) {
        self.out.extend(batch.out);
    }

    /// Return the size of the [`OutBatch`]
    pub fn size(&self) -> usize {
        self.out.len()
    }

    /// Return `true` if [`OutBatch`] is empty.
    pub fn is_empty(&self) -> bool {
        self.out.len() == 0
    }

    /// Return the [`Id`] and [`Outcome`] at position `index` within the batch.
    pub fn bindex(&self, index: usize) -> &'_(SolId, Out) {
        &self.out[index]
    }

    /// Return the [`Id`] and [`Outcome`] at position `index` within the batch.
    pub fn remove(&mut self, index: usize) -> (SolId,Out) {
        self.out.remove(index)
    }

    /// Pop the last [`Id`] and [`Outcome`] within the batch.
    pub fn pop(&mut self) -> Option<(SolId, Out)> {
        self.out.pop()
    }
}



//--------------------//
//--- INTOITERATOR ---//
//--------------------//

impl<SolId, Info, Out> IntoIterator for OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    type Item = (SolId, Out);
    type IntoIter = IntoIter<(SolId,Out)>;

    fn into_iter(self) -> Self::IntoIter {
        self.out.into_iter()
    }
}

impl<'a, SolId, Info, Out> IntoIterator for &'a OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    type Item = &'a (SolId, Out);
    type IntoIter = slice::Iter<'a, (SolId,Out)>;

    fn into_iter(self) -> Self::IntoIter {
        self.out.iter()
    }
}

impl<'a, SolId, Info, Out> IntoIterator for &'a mut OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    type Item = &'a mut (SolId, Out);
    type IntoIter = slice::IterMut<'a, (SolId,Out)>;

    fn into_iter(self) -> Self::IntoIter {
        self.out.iter_mut()
    }
}

//-----------------------//
//--- INTOPARITERATOR ---//
//-----------------------//

impl<SolId, Info, Out> IntoParallelIterator for OutBatch<SolId, Info, Out>
where
    SolId: Id + Send,
    Info: OptInfo,
    Out: Outcome + Send,
{
    type Item = (SolId, Out);
    type Iter = rayon::vec::IntoIter<(SolId,Out)>;

    fn into_par_iter(self) -> Self::Iter {
        self.out.into_par_iter()
    }
}

impl<'a, SolId, Info, Out> IntoParallelIterator for &'a OutBatch<SolId, Info, Out>
where
    SolId: Id + Send + Sync,
    Info: OptInfo,
    Out: Outcome + Send + Sync,
{
    type Item = &'a(SolId, Out);
    type Iter = rayon::slice::Iter<'a, (SolId,Out)>;

    fn into_par_iter(self) -> Self::Iter {
        <&[_]>::into_par_iter(&self.out)
    }
}

impl<'a, SolId, Info, Out> IntoParallelIterator for &'a mut OutBatch<SolId, Info, Out>
where
    SolId: Id + Send + Sync,
    Info: OptInfo,
    Out: Outcome + Send + Sync,
{
    type Item = &'a mut(SolId,Out);
    type Iter = rayon::slice::IterMut<'a, (SolId,Out)>;

    fn into_par_iter(self) -> Self::Iter {
        <&mut [_]>::into_par_iter(&mut self.out)
    }
}