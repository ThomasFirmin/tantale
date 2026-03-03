use crate::{
    domain::{Codomain, Domain, onto::LinkOpt},
    objective::{FuncWrapper, Outcome},
    recorder::csv::CSVWritable,
    searchspace::{CompShape, OptionCompShape, Searchspace},
    solution::{Batch, Id, IntoComputed, SolInfo, SolutionShape, Uncomputed, shape::RawObj},
};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Per-iteration metadata produced by an [`Optimizer`].
///
/// This typically aggregates informations about
/// the current iteration of the [`Optimizer`].
///
/// # Associated Derive Macro
///
/// The `OptInfo` derive macro automatically implements the trait for any struct
/// satisfying the required trait bounds.
pub trait OptInfo
where
    Self: Serialize + for<'de> Deserialize<'de> + Debug + Default,
{
}

/// Serializable state of an [`Optimizer`].
///
/// Implementations should capture all information required to resume an
/// optimization after checkpointing.
///
/// # Associated Derive Macro
///
/// The `OptState` derive macro automatically implements the trait for any struct
/// satisfying the required trait bounds.
pub trait OptState
where
    Self: Serialize + for<'de> Deserialize<'de>,
{
}

/// Empty implementation for [`OptInfo`] or [`SolInfo`].
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct EmptyInfo;
impl SolInfo for EmptyInfo {}
impl OptInfo for EmptyInfo {}
impl CSVWritable<(), ()> for EmptyInfo {
    fn header(_elem: &()) -> Vec<String> {
        Vec::new()
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::new()
    }
}

/// Type aliases for cleaner associated type definitions of [`Optimizer`]s [`OptInfo`].
pub type OpInfType<Op, PSol, Scp, SolId, Out> =
    <Op as Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>>::Info;
/// Type aliases for cleaner associated type definitions of [`Optimizer`]s [`SolInfo`].
pub type OpSInfType<Op, PSol, Scp, SolId, Out> =
    <Op as Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>>::SInfo;
/// Type aliases for cleaner associated type definitions of [`Optimizer`]s [`Codomain`].
pub type OpCodType<Op, PSol, Scp, SolId, Out> =
    <Op as Optimizer<PSol, SolId, LinkOpt<Scp>, Out, Scp>>::Cod;

/// Core optimizer abstraction.
///
/// An [`Optimizer`] generates candidate [`Solution`](crate::solution::Solution)s to
/// **maximize** an objective, and maintains the state required to resume or
/// checkpoint an experiment.
pub trait Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
{
    type State: OptState;
    type Cod: Codomain<Out>;
    type SInfo: SolInfo;
    type Info: OptInfo;

    /// Returns a mutable reference to the [`OptState`] of the [`Optimizer`].
    fn get_mut_state(&mut self) -> &Self::State;
    /// Returns a reference to the [`OptState`] of the [`Optimizer`].
    fn get_state(&self) -> &Self::State;
    /// Builds an [`Optimizer`] from a previously saved [`OptState`].
    fn from_state(state: Self::State) -> Self;
}

/// A [`Batch`] of [`CompShape`](crate::searchspace::CompShape) solutions for a given [`Searchspace`] and [`Codomain`].
pub type CompBatch<SolId, SInfo, Info, Scp, PSol, Cod, Out> =
    Batch<SolId, SInfo, Info, CompShape<Scp, PSol, SolId, SInfo, Cod, Out>>;

/// Batch optimizer interface.
///
/// At each iteration, the optimizer produces a whole [`Batch`] of new candidates.
pub trait BatchOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    <Scp::SolShape as IntoComputed>::Computed<Self::Cod, Out>: SolutionShape<SolId, Self::SInfo>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    /// Executed once at the beginning of the optimization. Does not require previous
    /// computed solutions.
    fn first_step(&mut self, scp: &Scp) -> Batch<SolId, Self::SInfo, Self::Info, Scp::SolShape>;
    /// Computes a single iteration of the [`Optimizer`].
    ///
    /// It consumes a [`Batch`] of computed solutions and returns a new batch of
    /// uncomputed candidates.
    fn step(
        &mut self,
        x: CompBatch<SolId, Self::SInfo, Self::Info, Scp, PSol, Self::Cod, Out>,
        scp: &Scp,
    ) -> Batch<SolId, Self::SInfo, Self::Info, Scp::SolShape>;
}

/// Sequential optimizer interface.
///
/// At each iteration, the optimizer produces a single [`Uncomputed`] candidate.
pub trait SequentialOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    /// Computes a single iteration of the [`Optimizer`].
    ///
    /// It consumes an optional computed solution and returns a new uncomputed
    /// candidate. [`Self`] is mutable to update the internal state.
    fn step(
        &mut self,
        x: OptionCompShape<Scp, PSol, SolId, Self::SInfo, Self::Cod, Out>,
        scp: &Scp,
    ) -> Scp::SolShape;
}

/// Multi-instance optimizer interface for parallel execution.
///
/// Implementations define how optimizer instances interact and synchronize.
pub trait MultiInstanceOptimizer<PSol, SolId, Opt, Out, Scp, Fn>:
    Optimizer<PSol, SolId, Opt, Out, Scp>
where
    PSol: Uncomputed<SolId, Opt, Self::SInfo>,
    SolId: Id,
    Opt: Domain,
    Out: Outcome,
    Scp: Searchspace<PSol, SolId, Self::SInfo, Opt = Opt>,
    Fn: FuncWrapper<RawObj<Scp::SolShape, SolId, Self::SInfo>>,
{
    /// Exchange information with peer instances.
    fn interact(&self);
    /// Update internal state after interaction.
    fn update(&self);
}