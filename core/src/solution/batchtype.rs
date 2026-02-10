//! Batch containers for [`SolutionShape`]s and [`Outcome`]s.
//!
//! A [`Batch`] groups multiple [`SolutionShape`](crate::solution::SolutionShape)s, typically
//! created by a [`BatchedOptimizer`](crate::BatchOptimizer). All solutions within a batch share the same [`OptInfo`](crate::OptInfo)
//! metadata. An [`OutBatch`] groups raw [`Outcome`](crate::Outcome) values paired with the
//! corresponding [`Id`](crate::Id).
//!
//! # Examples
//! ```
//! use tantale::core::{Batch, BasePartial, EmptyInfo, Id, Pair, Real, SId, Unit};
//! use std::sync::Arc;
//!
//! let info = Arc::new(EmptyInfo {});
//! let obj = BasePartial::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
//! let opt = BasePartial::<SId, Unit, _>::new(obj.get_id(), Arc::from(vec![0.9]), info.clone());
//! let pair = Pair::new(obj, opt);
//!
//! let mut batch = Batch::new(vec![pair], info);
//! assert_eq!(batch.size(), 1);
//! assert!(!batch.is_empty());
//! ```

use crate::{
    OptInfo,
    objective::{Outcome, Step},
    solution::{HasInfo, HasStep, Id, SolInfo, SolutionShape},
};
use core::slice;
use rayon::iter::IntoParallelIterator;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData, sync::Arc, vec::IntoIter};

#[cfg(feature = "mpi")]
use crate::experiment::mpi::utils::PriorityList;
#[cfg(feature = "mpi")]
use mpi::Rank;
#[cfg(feature = "mpi")]
use std::collections::HashMap;

/// A [`Batch`] describes a collection of `Obj` and `Opt` [`SolutionShape`].
///
/// All pairs in the batch share the same [`OptInfo`](crate::OptInfo).
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Shape: Serialize",
    deserialize = "Shape: for<'a> Deserialize<'a>",
))]
pub struct Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
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
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [`Solution`](crate::Solution) within a [`SolutionShape`] and an [`OptInfo`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, BasePartial, EmptyInfo, Id, Pair, Real, SId, Unit};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj = BasePartial::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt = BasePartial::<SId, Unit, _>::new(obj.get_id(), Arc::from(vec![0.9]), info.clone());
    /// let pair = Pair::new(obj, opt);
    ///
    /// let batch = Batch::new(vec![pair], info);
    /// assert_eq!(batch.size(), 1);
    /// ```
    pub fn new(pairs: Vec<Shape>, info: Arc<Info>) -> Self {
        Batch {
            pairs,
            info,
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }

    /// Creates a new empty [`Batch`] from an [`OptInfo`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, EmptyInfo};
    /// use std::sync::Arc;
    ///
    /// let batch: Batch<_, _, EmptyInfo, _> = Batch::empty(Arc::new(EmptyInfo {}));
    /// assert!(batch.is_empty());
    /// ```
    pub fn empty(info: Arc<Info>) -> Self {
        Batch {
            pairs: Vec::new(),
            info,
            _id: PhantomData,
            _sinfo: PhantomData,
        }
    }

    /// Add a new [`SolutionShape`] to the batch.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, BasePartial, EmptyInfo, Id, Pair, Real, SId, Unit};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj = BasePartial::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt = BasePartial::<SId, Unit, _>::new(obj.get_id(), Arc::from(vec![0.9]), info.clone());
    /// let pair = Pair::new(obj, opt);
    ///
    /// let mut batch = Batch::empty(info);
    /// batch.add(pair);
    /// assert_eq!(batch.size(), 1);
    /// ```
    pub fn add(&mut self, pair: Shape) {
        self.pairs.push(pair);
    }

    /// Add a new vec of [`SolutionShape`]s to the batch.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, BasePartial, EmptyInfo, Id, Pair, Real, SId, Unit};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj1 = BasePartial::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt1 = BasePartial::<SId, Unit, _>::new(obj1.get_id(), Arc::from(vec![0.9]), info.clone());
    /// let obj2 = BasePartial::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.2]), info.clone());
    /// let opt2 = BasePartial::<SId, Unit, _>::new(obj2.get_id(), Arc::from(vec![0.8]), info.clone());
    /// let pairs = vec![Pair::new(obj1, opt1), Pair::new(obj2, opt2)];
    ///
    /// let mut batch = Batch::empty(info);
    /// batch.add_vec(pairs);
    /// assert_eq!(batch.size(), 2);
    /// ```
    pub fn add_vec(&mut self, pairs: Vec<Shape>) {
        self.pairs.extend(pairs);
    }

    /// Extend current [`Batch`] with a new [`Batch`], [`OptInfo`] should be identical. Otherwise, the latter's [`OptInfo`] will be discarded in favor of the former's.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, BasePartial, EmptyInfo, Id, Pair, Real, SId, Unit};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj = BasePartial::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt = BasePartial::<SId, Unit, _>::new(obj.get_id(), Arc::from(vec![0.9]), info.clone());
    /// let pair = Pair::new(obj, opt);
    ///
    /// let mut a = Batch::new(vec![pair], info.clone());
    /// let b = Batch::empty(info);
    /// a.extend(b);
    /// assert_eq!(a.size(), 1);
    /// ```
    pub fn extend(&mut self, batch: Self) {
        self.pairs.extend(batch.pairs);
    }

    /// Return the size of the [`Batch`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, EmptyInfo};
    /// use std::sync::Arc;
    ///
    /// let batch: Batch<_, _, EmptyInfo, _> = Batch::empty(Arc::new(EmptyInfo {}));
    /// assert_eq!(batch.size(), 0);
    /// ```
    pub fn size(&self) -> usize {
        self.pairs.len()
    }

    /// Return `true` if [`Batch`] is empty.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, EmptyInfo};
    /// use std::sync::Arc;
    ///
    /// let batch: Batch<_, _, EmptyInfo, _> = Batch::empty(Arc::new(EmptyInfo {}));
    /// assert!(batch.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.pairs.len() == 0
    }

    /// Return [`SolutionShape`] at position `index` within the batch.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn bindex(&self, index: usize) -> &'_ Shape {
        &self.pairs[index]
    }

    /// Remove the  [`SolutionShape`] at position `index` within the batch.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> Shape {
        self.pairs.remove(index)
    }

    /// Pop the last [`SolutionShape`] within the batch.
    ///
    /// Returns `None` if the batch is empty.
    pub fn pop(&mut self) -> Option<Shape> {
        self.pairs.pop()
    }
}

impl<SolId, SInfo, Info, Shape> Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + HasStep,
{
    /// Split the [`Batch`] into buckets based on [`Step`].
    ///
    /// Returns `(error, discard, evaluated, partially, pending)` [`Batch`]es in this order.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, EmptyInfo, FidBasePartial, HasStep, Id, Pair, Real, SId, Unit};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj = FidBasePartial::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt = FidBasePartial::<SId, Unit, _>::new(obj.get_id(), Arc::from(vec![0.9]), info.clone());
    /// let mut pair = Pair::new(obj, opt);
    /// pair.evaluated(); // Only modified by internal state, not by the user.
    ///
    /// let batch = Batch::new(vec![pair], info);
    /// let (_err, _disc, eval, _part, _pend) = batch.chunk_by_step();
    /// assert_eq!(eval.size(), 1);
    /// ```
    pub fn chunk_by_step(self) -> (Self, Self, Self, Self, Self) {
        let info = self.info.clone();
        self.into_iter().fold(
            (
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
                Self::empty(info.clone()),
            ),
            |(mut pend, mut part, mut disc, mut eval, mut erro), pair| {
                match pair.step() {
                    Step::Pending => pend.add(pair),
                    Step::Partially(_) => part.add(pair),
                    Step::Evaluated => eval.add(pair),
                    Step::Discard => disc.add(pair),
                    Step::Error => erro.add(pair),
                }
                (erro, disc, eval, part, pend)
            },
        )
    }

    #[cfg(feature = "mpi")]
    /// Split the batch into MPI priority lists and a new batch based on [`Step`].
    ///
    /// * [`Pending`](Step::Pending) solutions go to `new_batch`.
    /// * [`Partially`](Step::Partially) solutions go to `priority_resume`.
    /// * [`Discard`](Step::Discard) solutions go to `priority_discard`.
    pub fn chunk_to_priority(
        self,
        where_is_id: &mut HashMap<SolId, Rank>,
        priority_discard: &mut PriorityList<Shape>,
        priority_resume: &mut PriorityList<Shape>,
        new_batch: &mut Self,
    ) {
        self.into_iter().for_each(|pair| match pair.step() {
            Step::Pending => new_batch.add(pair),
            Step::Partially(_) => {
                let rank = where_is_id.remove(&pair.get_id()).unwrap();
                priority_resume.add(pair, rank);
            }
            Step::Discard => {
                let rank = where_is_id.remove(&pair.get_id()).unwrap();
                priority_discard.add(pair, rank);
            }
            _ => {}
        })
    }
}

impl<SolId, SInfo, Info, Shape> HasInfo<Info> for Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    /// Return the optimizer metadata shared by the batch.
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
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    type Item = Shape;
    type IntoIter = IntoIter<Shape>;

    fn into_iter(self) -> Self::IntoIter {
        self.pairs.into_iter()
    }
}

impl<'a, SolId, SInfo, Info, Shape> IntoIterator for &'a Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    type Item = &'a Shape;
    type IntoIter = slice::Iter<'a, Shape>;

    fn into_iter(self) -> Self::IntoIter {
        self.pairs.iter()
    }
}

impl<'a, SolId, SInfo, Info, Shape> IntoIterator for &'a mut Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
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
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + Send + Sync,
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
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + Send + Sync,
{
    type Item = &'a Shape;
    type Iter = rayon::slice::Iter<'a, Shape>;

    fn into_par_iter(self) -> Self::Iter {
        <&[_]>::into_par_iter(&self.pairs)
    }
}

impl<'a, SolId, SInfo, Info, Shape> IntoParallelIterator
    for &'a mut Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo> + Send + Sync,
{
    type Item = &'a mut Shape;
    type Iter = rayon::slice::IterMut<'a, Shape>;

    fn into_par_iter(self) -> Self::Iter {
        <&mut [_]>::into_par_iter(&mut self.pairs)
    }
}

/// Convert a [`Vec`] of [`SolutionShape`]s into a [`Batch`].
///
/// # Notes
///
/// The [`OptInfo`] is set to default.
impl<SolId, SInfo, Info, Shape> FromIterator<Shape> for Batch<SolId, SInfo, Info, Shape>
where
    SolId: Id,
    SInfo: SolInfo,
    Info: OptInfo,
    Shape: SolutionShape<SolId, SInfo>,
{
    fn from_iter<T: IntoIterator<Item = Shape>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect(), Arc::new(Info::default()))
    }
}

/// A [`OutBatch`] describes a collection of solution [`Id`]s and raw [`Outcome`] values.
///
/// All entries share the same [`OptInfo`](crate::OptInfo).
#[derive(Debug)]
pub struct OutBatch<SolId, Info, Out>
where
    SolId: Id,
    Info: OptInfo,
    Out: Outcome,
{
    pub out: Vec<(SolId, Out)>,
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
    /// Creates a new [`OutBatch`] from a vec of pairs of solution [`Id`]s and raw [`Outcome`] values, and an [`OptInfo`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { value: f64 }
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let out = vec![(SId::generate(), Out { value: 1.0 })];
    /// let batch = OutBatch::new(out, info);
    /// assert_eq!(batch.size(), 1);
    /// ```
    pub fn new(out: Vec<(SolId, Out)>, info: Arc<Info>) -> Self {
        OutBatch { out, info }
    }

    /// Creates a new empty [`OutBatch`] from an [`OptInfo`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, OutBatch};
    /// use std::sync::Arc;
    ///
    /// let batch: OutBatch<_, _, _> = OutBatch::empty(Arc::new(EmptyInfo {}));
    /// assert!(batch.is_empty());
    /// ```
    pub fn empty(info: Arc<Info>) -> Self {
        OutBatch {
            out: Vec::new(),
            info,
        }
    }

    /// Add a new pair of solution [`Id`] and raw [`Outcome`] to the batch.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { value: f64 }
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let mut batch: OutBatch<_, _, Out> = OutBatch::empty(info);
    /// batch.add((SId::generate(), Out { value: 2.0 }));
    /// assert_eq!(batch.size(), 1);
    /// ```
    pub fn add(&mut self, out: (SolId, Out)) {
        self.out.push(out);
    }

    /// Add a new vec of paired solution [`Id`]s and raw [`Outcome`]s to the batch.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { value: f64 }
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let mut batch: OutBatch<_, _, Out> = OutBatch::empty(info);
    /// batch.add_vec(vec![(SId::generate(), Out { value: 2.0 }), (SId::generate(), Out { value: 3.0 })]);
    /// assert_eq!(batch.size(), 2);
    /// ```
    pub fn add_vec(&mut self, vout: Vec<(SolId, Out)>) {
        self.out.extend(vout);
    }

    /// Extend current [`OutBatch`] with a new [`OutBatch`], [`OptInfo`] should be identical. Otherwise, the latter's [`OptInfo`] will be discarded in favor of the former's.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { value: f64 }
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let mut a: OutBatch<_, _, Out> = OutBatch::empty(info.clone());
    /// let b = OutBatch::new(vec![(SId::generate(), Out { value: 1.0 })], info);
    /// a.extend(b);
    /// assert_eq!(a.size(), 1);
    /// ```
    pub fn extend(&mut self, batch: Self) {
        self.out.extend(batch.out);
    }

    /// Return the size of the [`OutBatch`]
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, OutBatch};
    /// use std::sync::Arc;
    ///
    /// let batch: OutBatch<_, _, _> = OutBatch::empty(Arc::new(EmptyInfo {}));
    /// assert_eq!(batch.size(), 0);
    /// ```
    pub fn size(&self) -> usize {
        self.out.len()
    }

    /// Return `true` if [`OutBatch`] is empty.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, OutBatch};
    /// use std::sync::Arc;
    ///
    /// let batch: OutBatch<_, _, _> = OutBatch::empty(Arc::new(EmptyInfo {}));
    /// assert!(batch.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.out.len() == 0
    }

    /// Return the [`Id`] and [`Outcome`] at position `index` within the batch.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn bindex(&self, index: usize) -> &'_ (SolId, Out) {
        &self.out[index]
    }

    /// Remove the paired [`Id`] and [`Outcome`] at position `index` within the batch.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> (SolId, Out) {
        self.out.remove(index)
    }

    /// Pop the last paired solution [`Id`] and raw [`Outcome`] within the batch.
    ///
    /// Returns `None` if the batch is empty.
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
    type IntoIter = IntoIter<(SolId, Out)>;

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
    type IntoIter = slice::Iter<'a, (SolId, Out)>;

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
    type IntoIter = slice::IterMut<'a, (SolId, Out)>;

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
    type Iter = rayon::vec::IntoIter<(SolId, Out)>;

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
    type Item = &'a (SolId, Out);
    type Iter = rayon::slice::Iter<'a, (SolId, Out)>;

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
    type Item = &'a mut (SolId, Out);
    type Iter = rayon::slice::IterMut<'a, (SolId, Out)>;

    fn into_par_iter(self) -> Self::Iter {
        <&mut [_]>::into_par_iter(&mut self.out)
    }
}
