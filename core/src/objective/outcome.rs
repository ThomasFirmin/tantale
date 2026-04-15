//! Defines the raw outputs of the user-defined function to optimize.
//!
//! An [`Outcome`] is a user-defined struct describing the output of
//! the objective function. This output may include optimized values, constraints,
//! evaluation cost, fidelities, and additional metadata (e.g. timing, seeds, or
//! debug information).
//!
//! # Notes
//! ## Serialization and CSV recording
//! An [`Outcome`] must be serializable for checkpointing and
//! compatible with CSV recording. In practice, this means implementing
//! [`Serialize`] and [`Deserialize`], and ensuring
//! fields can be written by the [`CSVWritable`](crate::recorder::csv::CSVWritable) layer.
//! Supported CSV field types are:
//! - Integers: [`isize`], [`i32`], [`i64`], [`usize`], [`u32`], [`u64`]
//! - Floats: [`f32`], [`f64`]
//! - Other: [`String`], [`bool`]
//! - [`Vec`] when its elements implement [`Debug`]
//!
//! Other fields remain valid for checkpointing as long as they are serializable,
//! but they will not be written to CSV.
//!
//! ## Multi-fidelity outputs
//! A [`FidOutcome`] is an [`Outcome`]
//! that exposes an [`EvalStep`], describing the current evaluation
//! state. It is used in multi-[`Fidelity`](crate::solution::partial::Fidelity) optimization and
//! [`Stepped`](crate::objective::Stepped) objectives. See [`Step`](crate::objective::Step) for the meaning of each state.
//!
//! # Example
//!
//! ```
//! use tantale::macros::Outcome;
//! use tantale::core::{Codomain, CostConstMultiCodomain};
//! use std::fmt::Debug;
//! use serde::{Serialize,Deserialize};
//!
//! #[derive(Outcome,Debug,Serialize,Deserialize)]
//! pub struct OutExample {
//!     pub cost2: f64,
//!     pub con3: f64,
//!     pub con4: f64,
//!     pub con5: f64,
//!     pub mul6: f64,
//!     pub mul7: f64,
//!     pub mul8: f64,
//!     pub mul9: f64,
//! }
//!
//! // A mock output of an objective function
//! let out = OutExample {
//!              cost2: 2.0,
//!              con3: 3.0,
//!              con4: 4.0,
//!              con5: 5.0,
//!              mul6: 6.0,
//!              mul7: 7.0,
//!              mul8: 8.0,
//!              mul9: 9.0,
//!          };
//!
//!
//! // Relation between Outcome and Codomain
//! let codom = CostConstMultiCodomain::new(
//!        // Define multi-objective
//!        vec![
//!            |h: &OutExample| h.mul6,
//!            |h: &OutExample| h.mul7,
//!            |h: &OutExample| h.mul8,
//!            |h: &OutExample| h.mul9,
//!        ]
//!        .into_boxed_slice(),
//!        // Define Cost
//!        |h: &OutExample| h.cost2,
//!        // Define constraints
//!        vec![
//!            |h: &OutExample| h.con3,
//!            |h: &OutExample| h.con4,
//!            |h: &OutExample| h.con5,
//!        ]
//!        .into_boxed_slice(),
//!    );
//! let extracted = codom.get_elem(&out);
//! println!("MULTI : {:?}", extracted.value);
//! println!("CONSTRAINT : {:?}", extracted.constraints);
//! println!("Cost : {}", extracted.cost);
//! ```

use crate::objective::EvalStep;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, path::PathBuf};

/// Trait implemented by objective outputs.
///
/// An [`Outcome`] is expected to be a named-field struct carrying the values used by
/// the [`Codomain`](crate::domain::codomain::Codomain) (objectives, constraints, cost,
/// or any metadata). It must be serializable for checkpointing and compatible with
/// CSV recording.
///
/// # Associated Derive Macro
///
/// The [`Outcome`] trait is automatically implemented by the
/// `Outcome` derive macro.
/// It implements [`Outcome`] for any struct with named fields that also implements
/// [`Debug`], [`Serialize`], and [`Deserialize`].
/// It also implements [`CSVWritable`](crate::recorder::csv::CSVWritable)
/// for the struct, writing all fields that are compatible with CSV recording
/// (see module-level documentation for supported types).
pub trait Outcome
where
    Self: Sized + Debug + Serialize + for<'de> Deserialize<'de>,
{
}

/// Trait for multi-fidelity outcomes.
///
/// A [`FidOutcome`] extends [`Outcome`] by exposing the current [`EvalStep`], which
/// indicates whether the evaluation is [`Pending`](crate::objective::Step::Pending),
/// [`Partially`](crate::objective::Step::Partially), [`Evaluated`](crate::objective::Step::Evaluated),
/// [`Discard`](crate::objective::Step::Discard), or [`Error`](crate::objective::Step::Error).
/// This is required by multi-[`Fidelity`](crate::solution::partial::Fidelity) optimization
/// and [`Stepped`](crate::objective::Stepped) objectives.
pub trait FidOutcome: Outcome
where
    Self: Sized + Debug + Serialize + for<'de> Deserialize<'de>,
{
    fn get_step(&self) -> EvalStep;
}

/// Marker trait for internal function state in [`Stepped`](crate::objective::Stepped) functions.
///
/// This represents the per-[`Solution`](crate::solution::Solution) evaluation state carried between
/// [`Step`](crate::objective::Step)s of a multi-[`Fidelity`](crate::solution::partial::Fidelity)
/// objective. The state is provided back to the [`Stepped`](crate::objective::Stepped) function
/// on the next call and can hold any serializable data needed to resume evaluation.
pub trait FuncState
where
    Self: Sized,
{
    /// Saves the function state within a folder for checkpointing.
    /// # Parameters
    /// - `path`: Folder path where the checkpoint is stored.
    ///   The user can decide how to serialize the state (e.g., using rmp_serde, bincode, etc.).
    fn save(&self, path: PathBuf) -> std::io::Result<()>;
    /// Loads the function state from a folder for checkpointing.
    /// # Parameters
    /// - `path`: Folder path where the checkpoint is stored.
    ///   The user can decide how to deserialize the state (e.g., using rmp_serde, bincode, etc.).
    fn load(path: PathBuf) -> std::io::Result<Self>;
}
