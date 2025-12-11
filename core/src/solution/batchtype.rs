use crate::{
    OptInfo, objective::{Outcome, Step}, solution::{Id, SolInfo, SolutionShape, partial::FidelityPartial}
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
    pub pair: Vec<Shape>,
    pub info: Arc<Info>,
    _id: PhantomData<SolId>,
    _sinfo: PhantomData<SInfo>,
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

impl<SolId, SInfo, Info, Shape> Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>
{
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [`Partial`] and an [`OptInfo`].
    pub fn new(pair: Vec<Shape>, info: Arc<Info>) -> Self {
        Batch { pair, info, _id: PhantomData, _sinfo: PhantomData}
    }

    /// Creates a new empty [`CompBatch`] from an [`OptInfo`].
    pub fn empty(info: Arc<Info>) -> Self {
        Batch {
            pair: Vec::new(),
            info,
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }

    /// Add a new `Obj` and `Opt` pair of [`Partial`] to the batch.
    pub fn add(&mut self, pair: Shape) {
        self.pair.push(pair);
    }

    /// Add a new vec of pairs `Obj` and `Opt` [`Partial`] to the batch.
    pub fn add_vec(&mut self, pair:Vec<Shape>) {
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
    pub fn index(&self, index: usize) -> &'_ Shape {
        &self.pair[index]
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn remove(&mut self, index: usize) -> Shape {
        self.pair.remove(index)
    }

    /// Return the `Obj` and `Opt` [`Partial`] at position `index` within the batch.
    pub fn pop(&mut self) -> Option<Shape> {
        self.pair.pop()
    }
}

impl<SolId, SInfo, Info, Shape> Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info:OptInfo,
    Shape: SolutionShape<SolId,SInfo>,
    Shape::SolObj: FidelityPartial<SolId,Shape::Obj,SInfo>,
    Shape::SolOpt: FidelityPartial<SolId,Shape::Opt,SInfo>,
{
    /// Split the [`Batch`] made of [`FidPartial`] into four [`Batch`] according to [`Step`].
    /// * 1st [`Error`](Step::Error)
    /// * 2nd [`Evaluated`](Step::Evaluated)
    /// * 3rd [`Penultimate`](Step::Penultimate)
    /// * 4th [`Partially`](Step::Partially)
    /// * 5th [`Pending`](Step::Pending)
    /// * 6th [`Other`](Step::Other)
    pub fn chunk_by_step(self)->(Self,Self,Self,Self,Self,Self){
        let info = self.info.clone();
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
                match pair.step() {
                    Step::Pending => pen.add(pair),
                    Step::Partially(_) => pa.add(pair),
                    Step::Penultimate => pe.add(pair),
                    Step::Evaluated => ev.add(pair),
                    Step::Error => er.add(pair),
                }
                (er,ev,pe,pa,pen,ot)
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
        priority_last: &mut PriorityList<Shape>,
        priority_resume: &mut PriorityList<Shape>,
        new_batch: &mut Self,
    )
    {
        self.into_iter().for_each(
            |pair|
            {
                match pair.get_sopt().get_step(){
                    Step::Pending => todo!(),
                    Step::Partially(_) => todo!(),
                    Step::Penultimate => todo!(),
                    Step::Evaluated => todo!(),
                    Step::Error => todo!(),
                    Step::Other(_) => todo!(),
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
    pub fn index(&self, index: usize) -> &'_(SolId, Out) {
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
        self.pair.into_iter()
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
        self.pair.iter()
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
        self.pair.into_par_iter()
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
        <&[_]>::into_par_iter(&self.pair)
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
        <&mut [_]>::into_par_iter(&mut self.pair)
    }
}

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