//! Batch containers for [`SolutionShape`]s and [`Outcome`]s.
//!
//! A [`Batch`] groups multiple [`SolutionShape`]s, typically
//! created by a [`BatchedOptimizer`](crate::BatchOptimizer). All solutions within a batch share the same [`OptInfo`]
//! metadata. An [`OutBatch`] groups raw [`Outcome`] values paired with the
//! corresponding [`Id`].
//!
//! # Examples
//! ```
//! use tantale::core::{Batch, BaseSol, Id, HasId, Pair, Real, SId, Unit, Uncomputed};
//! use std::sync::Arc;
//!
//! let obj = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]));
//! let opt = BaseSol::<SId, Unit, _>::new(obj.id(), Arc::from(vec![0.9]));
//! let pair = Pair::new(obj, opt);
//!
//! let mut batch = Batch::new(vec![pair]);
//! assert_eq!(batch.size(), 1);
//! assert!(!batch.is_empty());
//! ```

use crate::{
    HasStep, objective::{Outcome, Step}, solution::{SolId, SolutionShape},
};
use core::slice;
use rayon::iter::IntoParallelIterator;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, vec::IntoIter};

use std::collections::HashMap;

/// A [`Batch`] describes a collection of [`SolutionShape`].
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(
    serialize = "Shape: Serialize",
    deserialize = "Shape: for<'a> Deserialize<'a>",
))]
pub struct Batch<Shape: SolutionShape>
{
    pub pairs: Vec<Shape>,
}

impl<Shape> Batch<Shape>
where
    Shape: SolutionShape,
{
    /// Creates a new [`Batch`] from paired `Obj` and `Opt` [solutions](crate::Solution) within a [`SolutionShape`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, BaseSol, EmptyInfo, Id, HasId, Pair, Real, SId, Unit, Uncomputed};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt = BaseSol::<SId, Unit, _>::new(obj.id(), Arc::from(vec![0.9]), info.clone());
    /// let pair = Pair::new(obj, opt);
    ///
    /// let batch = Batch::new(vec![pair], info);
    /// assert_eq!(batch.size(), 1);
    /// ```
    pub fn new(pairs: Vec<Shape>) -> Self {
        Batch { pairs }
    }

    /// Creates a new empty [`Batch`] from an [`OptInfo`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{BaseSol, EmptyInfo, Batch, Lone, Real, SId, Uncomputed};
    /// use std::sync::Arc;
    ///
    /// let batch: Batch<Lone<BaseSol<SId, Real, EmptyInfo>>> = Batch::empty();
    /// assert!(batch.is_empty());
    /// ```
    pub fn empty() -> Self {
        Batch { pairs: Vec::new() }
    }

    /// Add a new [`SolutionShape`] to the batch.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, BaseSol, EmptyInfo, Id, HasId, Pair, Real, SId, Unit, Uncomputed};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt = BaseSol::<SId, Unit, _>::new(obj.id(), Arc::from(vec![0.9]), info.clone());
    /// let pair = Pair::new(obj, opt);
    ///
    /// let mut batch = Batch::empty();
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
    /// use tantale::core::{Batch, BaseSol, EmptyInfo, Id, HasId, Pair, Real, SId, Unit, Uncomputed};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj1 = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt1 = BaseSol::<SId, Unit, _>::new(obj1.id(), Arc::from(vec![0.9]), info.clone());
    /// let obj2 = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.2]), info.clone());
    /// let opt2 = BaseSol::<SId, Unit, _>::new(obj2.id(), Arc::from(vec![0.8]), info.clone());
    /// let pairs = vec![Pair::new(obj1, opt1), Pair::new(obj2, opt2)];
    ///
    /// let mut batch = Batch::empty();
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
    /// use tantale::core::{Batch, BaseSol, EmptyInfo, Id, HasId, Pair, Real, SId, Unit, Uncomputed};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj = BaseSol::<SId, Real, _>::new(SId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt = BaseSol::<SId, Unit, _>::new(obj.id(), Arc::from(vec![0.9]), info.clone());
    /// let pair = Pair::new(obj, opt);
    ///
    /// let mut a = Batch::new(vec![pair], info.clone());
    /// let b = Batch::empty();
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
    /// use tantale::core::{BaseSol, Batch, EmptyInfo, Lone, Real, SId, Uncomputed};
    /// use std::sync::Arc;
    ///
    /// let batch: Batch<SId, EmptyInfo, EmptyInfo, Lone<BaseSol<SId, Real, EmptyInfo>, SId, Real, EmptyInfo>> = Batch::empty(Arc::new(EmptyInfo {}));
    /// assert_eq!(batch.size(), 0);
    /// ```
    pub fn size(&self) -> usize {
        self.pairs.len()
    }

    /// Return `true` if [`Batch`] is empty.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{BaseSol, Batch, EmptyInfo, Lone, Real, SId, Uncomputed};
    /// use std::sync::Arc;
    ///
    /// let batch: Batch<SId, EmptyInfo, EmptyInfo, Lone<BaseSol<SId, Real, EmptyInfo>, SId, Real, EmptyInfo>> = Batch::empty(Arc::new(EmptyInfo {}));
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

impl<Shape> Batch<Shape>
where
    Shape: SolutionShape + HasStep,
{
    /// Split the [`Batch`] into buckets based on [`Step`].
    ///
    /// Returns `(error, discard, evaluated, partially, pending)` [`Batch`]es in this order.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Batch, EmptyInfo, FidelitySol, HasStep, Id, HasId, Pair, Real, StepSId, Unit, Uncomputed};
    /// use std::sync::Arc;
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let obj = FidelitySol::<StepSId, Real, _>::new(StepSId::generate(), Arc::from(vec![0.1]), info.clone());
    /// let opt = FidelitySol::<StepSId, Unit, _>::new(obj.id(), Arc::from(vec![0.9]), info.clone());
    /// let mut pair = Pair::new(obj, opt);
    /// pair.evaluated(); // Only modified by internal state, not by the user.
    ///
    /// let batch = Batch::new(vec![pair], info);
    /// let (_err, _disc, eval, _part, _pend) = batch.chunk_by_step();
    /// assert_eq!(eval.size(), 1);
    /// ```
    pub fn chunk_by_step(self) -> (Self, Self, Self, Self, Self) {
        self.into_iter().fold(
            (
                Self::empty(),
                Self::empty(),
                Self::empty(),
                Self::empty(),
                Self::empty(),
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
    ///
    /// The previous [`id_step`](StepId::id_step) of the solution is used to determine its rank in the priority lists,
    /// so that it can be sent to the same worker as before.
    pub fn chunk_to_priority(
        self,
        where_is_id: &mut HashMap<Shape::Id, Rank>,
        priority_discard: &mut PriorityList<Shape>,
        priority_resume: &mut PriorityList<Shape>,
        new_batch: &mut Self,
    ) {
        self.into_iter().for_each(|pair| match pair.step() {
            Step::Pending => new_batch.add(pair),
            Step::Partially(_) => {
                let rank = *where_is_id.get(&pair.id().previous_id()).unwrap();
                priority_resume.add(pair, rank);
            }
            Step::Discard => {
                let rank = *where_is_id.get(&pair.id().previous_id()).unwrap();
                priority_discard.add(pair, rank);
            }
            _ => {}
        })
    }
}

//--------------------//
//--- INTOITERATOR ---//
//--------------------//

impl<Shape> IntoIterator for Batch<Shape>
where
    Shape: SolutionShape + HasStep,
{
    type Item = Shape;
    type IntoIter = IntoIter<Shape>;

    fn into_iter(self) -> Self::IntoIter {
        self.pairs.into_iter()
    }
}

impl<'a, Shape> IntoIterator for &'a Batch<Shape>
where
    Shape: SolutionShape + HasStep,
{
    type Item = &'a Shape;
    type IntoIter = slice::Iter<'a, Shape>;

    fn into_iter(self) -> Self::IntoIter {
        self.pairs.iter()
    }
}

impl<'a, Shape> IntoIterator for &'a mut Batch<Shape>
where
    Shape: SolutionShape + HasStep,
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

impl<Shape> IntoParallelIterator for Batch<Shape>
where
    Shape: SolutionShape + Send + Sync,
{
    type Item = Shape;
    type Iter = rayon::vec::IntoIter<Shape>;

    fn into_par_iter(self) -> Self::Iter {
        self.pairs.into_par_iter()
    }
}

impl<'a, Shape> IntoParallelIterator for &'a Batch<Shape>
where
    Shape: SolutionShape + Send + Sync,
{
    type Item = &'a Shape;
    type Iter = rayon::slice::Iter<'a, Shape>;

    fn into_par_iter(self) -> Self::Iter {
        <&[_]>::into_par_iter(&self.pairs)
    }
}

impl<'a, Shape> IntoParallelIterator for &'a mut Batch<Shape>
where
    Shape: SolutionShape + Send + Sync,
{
    type Item = &'a mut Shape;
    type Iter = rayon::slice::IterMut<'a, Shape>;

    fn into_par_iter(self) -> Self::Iter {
        <&mut [_]>::into_par_iter(&mut self.pairs)
    }
}

/// Convert a [`Vec`] of [`SolutionShape`]s into a [`Batch`].
impl<Shape> FromIterator<Shape> for Batch<Shape>
where
    Shape: SolutionShape,
{
    fn from_iter<T: IntoIterator<Item = Shape>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

/// A [`OutBatch`] describes a collection of solution [`Id`]s and raw [`Outcome`] values.
#[derive(Debug)]
pub struct OutBatch<Id, Out>
where
    Id: SolId,
    Out: Outcome,
{
    pub out: Vec<(Id, Out)>,
}

impl<Id, Out> OutBatch<Id, Out>
where
    Id: SolId,
    Out: Outcome,
{
    /// Creates a new [`OutBatch`] from a vec of pairs of solution [`Id`]s and raw [`Outcome`] values, and an [`OptInfo`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{Id, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { #[maximize] value: f64 }
    ///
    /// let out = vec![(SId::generate(), Out { value: 1.0 })];
    /// let batch = OutBatch::new(out);
    /// assert_eq!(batch.size(), 1);
    /// ```
    pub fn new(out: Vec<(Id, Out)>) -> Self {
        OutBatch { out }
    }

    /// Creates a new empty [`OutBatch`] from an [`OptInfo`].
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, Id, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { #[maximize] value: f64 }
    ///
    /// let batch = OutBatch::<SId, Out>::empty();
    /// assert!(batch.is_empty());
    /// ```
    pub fn empty() -> Self {
        OutBatch {
            out: Vec::new(),
        }
    }

    /// Add a new pair of solution [`Id`] and raw [`Outcome`] to the batch.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, Id, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { #[maximize] value: f64 }
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let mut batch: OutBatch<_, _, Out> = OutBatch::empty();
    /// batch.add((SId::generate(), Out { value: 2.0 }));
    /// assert_eq!(batch.size(), 1);
    /// ```
    pub fn add(&mut self, out: (Id, Out)) {
        self.out.push(out);
    }

    /// Add a new vec of paired solution [`Id`]s and raw [`Outcome`]s to the batch.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, Id, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { #[maximize] value: f64 }
    ///
    /// let info = Arc::new(EmptyInfo {});
    /// let mut batch: OutBatch<_, _, Out> = OutBatch::empty(info);
    /// batch.add_vec(vec![(SId::generate(), Out { value: 2.0 }), (SId::generate(), Out { value: 3.0 })]);
    /// assert_eq!(batch.size(), 2);
    /// ```
    pub fn add_vec(&mut self, vout: Vec<(Id, Out)>) {
        self.out.extend(vout);
    }

    /// Extend current [`OutBatch`] with a new [`OutBatch`], [`OptInfo`] should be identical. Otherwise, the latter's [`OptInfo`] will be discarded in favor of the former's.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, Id, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { #[maximize] value: f64 }
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
    /// use tantale::core::{EmptyInfo, Id, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { #[maximize] value: f64 }
    ///
    /// let batch = OutBatch::<SId, EmptyInfo, Out>::empty(Arc::new(EmptyInfo {}));
    /// assert_eq!(batch.size(), 0);
    /// ```
    pub fn size(&self) -> usize {
        self.out.len()
    }

    /// Return `true` if [`OutBatch`] is empty.
    ///
    /// # Example
    /// ```
    /// use tantale::core::{EmptyInfo, Id, OutBatch, SId};
    /// use std::sync::Arc;
    /// use serde::{Deserialize, Serialize};
    /// use tantale::macros::Outcome;
    ///
    /// #[derive(Outcome, Serialize, Deserialize, Debug, Clone)]
    /// struct Out { #[maximize] value: f64 }
    ///
    /// let batch = OutBatch::<SId, EmptyInfo, Out>::empty(Arc::new(EmptyInfo {}));
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
    pub fn bindex(&self, index: usize) -> &'_ (Id, Out) {
        &self.out[index]
    }

    /// Remove the paired [`Id`] and [`Outcome`] at position `index` within the batch.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> (Id, Out) {
        self.out.remove(index)
    }

    /// Pop the last paired solution [`Id`] and raw [`Outcome`] within the batch.
    ///
    /// Returns `None` if the batch is empty.
    pub fn pop(&mut self) -> Option<(Id, Out)> {
        self.out.pop()
    }
}

//--------------------//
//--- INTOITERATOR ---//
//--------------------//

impl<Id, Out> IntoIterator for OutBatch<Id, Out>
where
    Id: SolId,
    Out: Outcome,
{
    type Item = (Id, Out);
    type IntoIter = IntoIter<(Id, Out)>;

    fn into_iter(self) -> Self::IntoIter {
        self.out.into_iter()
    }
}

impl<'a, Id, Out> IntoIterator for &'a OutBatch<Id, Out>
where
    Id: SolId,
    Out: Outcome,
{
    type Item = &'a (Id, Out);
    type IntoIter = slice::Iter<'a, (Id, Out)>;

    fn into_iter(self) -> Self::IntoIter {
        self.out.iter()
    }
}

impl<'a, Id, Out> IntoIterator for &'a mut OutBatch<Id, Out>
where
    Id: SolId,
    Out: Outcome,
{
    type Item = &'a mut (Id, Out);
    type IntoIter = slice::IterMut<'a, (Id, Out)>;

    fn into_iter(self) -> Self::IntoIter {
        self.out.iter_mut()
    }
}

//-----------------------//
//--- INTOPARITERATOR ---//
//-----------------------//

impl<Id, Out> IntoParallelIterator for OutBatch<Id, Out>
where
    Id: SolId + Send,
    Out: Outcome + Send,
{
    type Item = (Id, Out);
    type Iter = rayon::vec::IntoIter<(Id, Out)>;

    fn into_par_iter(self) -> Self::Iter {
        self.out.into_par_iter()
    }
}

impl<'a, Id, Out> IntoParallelIterator for &'a OutBatch<Id, Out>
where
    Id: SolId + Send + Sync,
    Out: Outcome + Send + Sync,
{
    type Item = &'a (Id, Out);
    type Iter = rayon::slice::Iter<'a, (Id, Out)>;

    fn into_par_iter(self) -> Self::Iter {
        <&[_]>::into_par_iter(&self.out)
    }
}

impl<'a, Id, Out> IntoParallelIterator for &'a mut OutBatch<Id, Out>
where
    Id: SolId + Send + Sync,
    Out: Outcome + Send + Sync,
{
    type Item = &'a mut (Id, Out);
    type Iter = rayon::slice::IterMut<'a, (Id, Out)>;

    fn into_par_iter(self) -> Self::Iter {
        <&mut [_]>::into_par_iter(&mut self.out)
    }
}
