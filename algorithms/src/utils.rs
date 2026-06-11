//! Utilities for basic algorithms implemented in Tantale algorithms

use serde::{Deserialize, Serialize};
use tantale_core::{
    BaseSol, Batch, CompAcc, CompShape, FidelitySol, HasFidelity, LinkOpt, Objective, RawObj, SId, StepSId, Stepped, searchspace::SShape
};

/// A type alias for [`SolutionShape`](tantale_core::SolutionShape) made of [`Computed`](tantale_core::Computed) [`BaseSol`], identified with [`SId`].
pub type BCompShape<Scp, Out, SInfo> =
    CompShape<SShape<Scp, BaseSol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>, SId, SInfo, Out>;
/// A type alias for a [`Batch`] of [`BCompShape`]s, identified with [`SId`].
pub type BatchBCompShape<Scp, Out, Info, SInfo> =
    Batch<SId, SInfo, Info, BCompShape<Scp, Out, SInfo>>;
/// A type alias for an [`Accumulator`](tantale_core::Accumulator) of [`BCompShape`]s.
pub type BCompAcc<Scp, Out, SInfo> =
    CompAcc<SShape<Scp, BaseSol<SId, LinkOpt<Scp>, SInfo>, SId, SInfo>, SId, SInfo, Out>;

/// A type alias for a [`CompShape`] made of [`Computed`](tantale_core::Computed) [`FidelitySol`], identified with [`StepSId`].
pub type FCompShape<Scp, Out, SInfo> = CompShape<
    SShape<Scp, FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, StepSId, SInfo>,
    StepSId,
    SInfo,
    Out,
>;
/// A type alias for a [`Batch`] of [`FCompShape`]s, identified with [`StepSId`].
pub type BatchFCompShape<Scp, Out, Info, SInfo> =
    Batch<StepSId, SInfo, Info, FCompShape<Scp, Out, SInfo>>;
/// A type alias for an [`Accumulator`](tantale_core::Accumulator) of [`FCompShape`]s.
pub type FCompAcc<Scp, Out, SInfo> = CompAcc<
    SShape<Scp, FidelitySol<StepSId, LinkOpt<Scp>, SInfo>, StepSId, SInfo>,
    StepSId,
    SInfo,
    Out,
>;

/// A type alias for a simple [`Objective`] with solution identified with [`SId`].
pub type SimpleObjective<Shape, SInfo, Out> = Objective<RawObj<Shape, SId, SInfo>, Out>;
/// A type alias for a simple [`Stepped`] with solution identified with [`StepSId`].
pub type SimpleStepped<Shape, SInfo, Out, State> =
    Stepped<RawObj<Shape, StepSId, SInfo>, Out, State>;

/// Archive of points which holds the observed points sorted in ascending order.
/// The points are sorted by their corresponding objective values, with the best points at the end of the vector.
#[derive(Serialize, Deserialize, Debug)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: for<'a> Deserialize<'a>",))]
pub struct OrdArchive<T>
where
    T: Ord + Serialize + for<'a> Deserialize<'a>,
{
    /// A vector of points observed in the optimization process, sorted in ascending order by their objective values.
    /// The best points are located at the end of the vector, while the worst points are at the beginning.
    pub points: Vec<T>, // sorted ascending by point
}

impl<T> OrdArchive<T>
where
    T: Ord + Serialize + for<'a> Deserialize<'a>,
{
    pub fn new(elem: T) -> Self {
        OrdArchive { points: vec![elem] }
    }
}

impl<T> Default for OrdArchive<T>
where
    T: Ord + Serialize + for<'a> Deserialize<'a>,
{
    fn default() -> Self {
        OrdArchive { points: Vec::new() }
    }
}

impl<T> OrdArchive<T>
where
    T: Ord + Serialize + for<'a> Deserialize<'a>,
{
    pub fn add(&mut self, point: T) {
        let pos = self.points.binary_search(&point).unwrap_or_else(|e| e);
        self.points.insert(pos, point);
    }
    pub fn size(&self) -> usize {
        self.points.len()
    }
}

/// A helper function to set the fidelity of a solution that implements the [`HasFidelity`] trait.
pub fn fidelity_setter<S:HasFidelity>(mut s: S, fidelity: f64) -> S{
    s.set_fidelity(fidelity);
    s
}

pub mod mo;

pub mod hypervolume;