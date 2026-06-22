//! Utilities for basic algorithms implemented in Tantale algorithms

use tantale_core::{
    BaseSol, Batch, CompAcc, CompShape, FidelitySol, HasFidelity, LinkOpt, Objective, RawObj, SId,
    StepSId, Stepped, searchspace::SShape,
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

/// A helper function to set the fidelity of a solution that implements the [`HasFidelity`] trait.
pub fn fidelity_setter<S: HasFidelity>(mut s: S, fidelity: f64) -> S {
    s.set_fidelity(fidelity);
    s
}

pub mod mo;
