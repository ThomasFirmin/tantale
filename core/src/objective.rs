//! # Objective Functions and Evaluation State
//!
//! This module provides the core abstractions for defining and managing objective functions
//! in optimization experiments, $f$. It includes evaluation state tracking, objective function
//! wrappers, and outcome definitions.
//!
//! ## Overview
//!
//! The primary components are:
//! - Evaluation state tracking via [`Step`] and [`EvalStep`]
//! - Objective function definitions via [`Objective`], [`Stepped`] and [`FuncWrapper`]
//! - Outcome representations via [`Outcome`] and [`FidOutcome`]
//!
//! ## Evaluation State
//!
//! Tantale tracks the evaluation progress of solutions using two complementary types:
//!
//! ### User-Facing: `Step` Enum
//!
//! [`Step`] is an enum that represents evaluation states in a user-friendly way:
//! - [`Step::Pending`] - Solution not yet evaluated
//! - [`Step::Partially(n)`](Step::Partially) - Partially evaluated to step `n`
//! - [`Step::Evaluated`] - Fully evaluated solution
//! - [`Step::Discard`] - Evaluation discarded by optimizer
//! - [`Step::Error`] - Evaluation failed
//!
//! ### Internal: `EvalStep` Struct
//!
//! [`EvalStep`] is the internal representation wrapping an `isize` for efficient serialization
//! and communication (especially in distributed/MPI contexts). The mapping is:
//! - `EvalStep(0)` ↔ [`Step::Pending`]
//! - `EvalStep(n > 0)` ↔ [`Step::Partially(n)`](Step::Partially)
//! - `EvalStep(-1)` ↔ [`Step::Evaluated`]
//! - `EvalStep(-9)` ↔ [`Step::Discard`]
//! - `EvalStep(-10)` ↔ [`Step::Error`]
//!
//! Conversion between these types is automatic via the [`From`] trait.
//!
//! ## Multi-Fidelity Evaluation
//!
//! Multi-fidelity optimization evaluates solutions incrementally, progressively refining
//! estimates. The [`Step::Partially(n)`](Step::Partially) state tracks partial progress, where `n` indicates
//! how many evaluation steps have completed.
//!
//! ## Objective Functions
//!
//! Objective functions implement the [`Objective`] trait and:
//! - Accept inputs from the [`Searchspace`](crate::searchspace::Searchspace)
//! - Return an [`Outcome`] containing objective values and metadata
//! - Support stepped evaluation with [`Stepped`]
//!
//! See the [`obj`] submodule for trait definitions and the [`outcome`] submodule for
//! outcome type details.
//!
//! ## See Also
//!
//! - [`objective!`](../../../tantale/macros/macro.objective.html) - Macro for defining the [`Searchspace`](crate::searchspace::Searchspace)
//!   and wrapping the raw user-defined objective function.
//! - [`Objective`] - Core trait for objective functions
//! - [`Outcome`] - Result type for evaluations
//! - [`Stepped`] - Multi-fidelity wrapper
//! - [`FuncWrapper`] - Internal function wrapper

use crate::recorder::csv::CSVWritable;
use serde::{Deserialize, Serialize};

/// User-facing enumeration representing the evaluation state of a solution.
///
/// [`Step`] provides a high-level, ergonomic representation of how far an objective function
/// evaluation has progressed. This enum is used throughout Tantale's public API for state
/// tracking and decision-making.
///
/// # Variants
///
/// - [`Pending`](Step::Pending) - The solution has not been evaluated yet
/// - [`Partially(n)`](Step::Partially) - The solution has been partially evaluated to step `n`
/// - [`Evaluated`](Step::Evaluated) - The solution has been fully evaluated
/// - [`Discard`](Step::Discard) - The evaluation was discarded (optimizer decided not to complete it)
/// - [`Error`](Step::Error) - The evaluation encountered an error
///
/// # Multi-Fidelity Context
///
/// The [`Partially(n)`](Step::Partially) variant is primarily used in multi-fidelity optimization,
/// where a solution can be evaluated incrementally. The value `n` indicates the current fidelity
/// level or number of completed steps.
///
/// # Relationship to `EvalStep`
///
/// [`Step`] is the user-facing counterpart to [`EvalStep`], which is the internal representation
/// optimized for serialization and MPI communication. The two types convert seamlessly via [`From`]:
///
/// ```ignore
/// let step = Step::Partially(5);
/// let eval_step: EvalStep = step.into();
/// let back: Step = eval_step.into();
/// assert_eq!(step, back);
/// ```
///
/// # See Also
///
/// - [`EvalStep`] - Internal integer-based representation
/// - [`Outcome`] - Contains a step field for tracking evaluation state
/// - [`Stepped`] - Multi-fidelity function wrapper that manages step progression
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Step {
    /// Solution has not been evaluated yet.
    Pending,
    /// Solution has been partially evaluated to the specified step.
    ///
    /// Used in multi-fidelity optimization where evaluations can be performed
    /// incrementally. The `isize` value indicates how many evaluation steps
    /// have been completed. Typically positive values are used for progressive
    /// steps (1, 2, 3, ...).
    Partially(isize),
    /// Solution has been fully evaluated.
    Evaluated,
    /// Evaluation was discarded by the optimizer.
    Discard,
    /// Evaluation encountered an error.
    Error,
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::Pending => write!(f, "Pending"),
            Step::Partially(v) => write!(f, "{}", v),
            Step::Evaluated => write!(f, "Evaluated"),
            Step::Discard => write!(f, "Discarded"),
            Step::Error => write!(f, "Error"),
        }
    }
}

impl PartialEq for Step {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Partially(l0), Self::Partially(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl CSVWritable<(), ()> for Step {
    fn header(_elem: &()) -> Vec<String> {
        Vec::from([String::from("step")])
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Vec::from([self.to_string()])
    }
}

/// Internal integer-based representation of evaluation state.
///
/// [`EvalStep`] is a lightweight wrapper around `isize` that represents evaluation state
/// using a compact integer encoding. This type is used internally and in serialization contexts
/// (especially MPI communication) where enum overhead is undesirable.
///
/// # Encoding Scheme
///
/// The integer value maps to [`Step`] variants as follows:
/// - `0` → [`Step::Pending`] - Unevaluated solution
/// - `n > 0` → [`Step::Partially(n)`](Step::Partially) - Partially evaluated to step `n`
/// - `-1` → [`Step::Evaluated`] - Fully evaluated
/// - `-9` → [`Step::Discard`] - Evaluation discarded
/// - `-10` → [`Step::Error`] - Evaluation error
///
/// # Why Not an Enum?
///
/// While [`Step`] uses an enum for ergonomic user-facing APIs, [`EvalStep`] uses a plain
/// integer for:
/// - **Efficient serialization**: Direct binary encoding without enum overhead
/// - **MPI communication**: Simple integer passing in distributed contexts
/// - **Compatibility**: Works seamlessly with binary protocols like MessagePack
///
/// # Conversion
///
/// [`EvalStep`] and [`Step`] convert bidirectionally via the [`From`] trait:
///
/// ```ignore
/// use tantale_core::objective::{Step, EvalStep};
///
/// // Step to EvalStep
/// let step = Step::Partially(3);
/// let eval_step: EvalStep = step.into();
/// assert_eq!(eval_step, EvalStep(3));
///
/// // EvalStep to Step
/// let eval_step = EvalStep(-1);
/// let step: Step = eval_step.into();
/// assert_eq!(step, Step::Evaluated);
/// ```
///
/// # Constructor Methods
///
/// [`EvalStep`] provides convenient constructors matching each state:
/// - [`pending()`](EvalStep::pending) - Creates `EvalStep(0)`
/// - [`partially(n)`](EvalStep::partially) - Creates `EvalStep(n)`
/// - [`evaluated()`](EvalStep::evaluated) - Creates `EvalStep(-1)`
/// - [`discard()`](EvalStep::discard) - Creates `EvalStep(-9)`
/// - [`error()`](EvalStep::error) - Creates `EvalStep(-10)`
///
/// # See Also
///
/// - [`Step`] - User-facing enum counterpart
/// - [`Outcome`] - Contains evaluation results with step information
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct EvalStep(pub isize);

impl PartialEq for EvalStep {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl From<Step> for EvalStep {
    fn from(value: Step) -> Self {
        match value {
            Step::Pending => Self(0),
            Step::Partially(v) => Self(v),
            Step::Evaluated => Self(-1),
            Step::Discard => Self(-9),
            Step::Error => Self(-10),
        }
    }
}

impl From<EvalStep> for Step {
    fn from(value: EvalStep) -> Self {
        match value {
            EvalStep(0) => Step::Pending,
            EvalStep(-1) => Step::Evaluated,
            EvalStep(-9) => Step::Discard,
            EvalStep(-10) => Step::Error,
            EvalStep(v) => Step::Partially(v),
        }
    }
}

impl EvalStep {
    /// Creates an [`EvalStep`] representing a pending (unevaluated) solution.
    ///
    /// Equivalent to `EvalStep(0)` and converts to [`Step::Pending`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let step = EvalStep::pending();
    /// assert_eq!(step, EvalStep(0));
    /// ```
    pub fn pending() -> Self {
        Self(0)
    }

    /// Creates an [`EvalStep`] representing a partially evaluated solution.
    ///
    /// The `value` parameter indicates the current evaluation step.
    /// Equivalent to `EvalStep(value as isize)` and converts to [`Step::Partially(value)`].
    ///
    /// # Parameters
    ///
    /// * `value` - The step number (positive integer) indicating evaluation progress
    ///
    /// # Example
    ///
    /// ```ignore
    /// let step = EvalStep::partially(5);
    /// assert_eq!(step, EvalStep(5));
    /// ```
    pub fn partially(value: usize) -> Self {
        Self(value as isize)
    }

    /// Creates an [`EvalStep`] representing a fully evaluated solution.
    ///
    /// Equivalent to `EvalStep(-1)` and converts to [`Step::Evaluated`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let step = EvalStep::evaluated();
    /// assert_eq!(step, EvalStep(-1));
    /// ```
    pub fn evaluated() -> Self {
        Self(-1)
    }

    /// Creates an [`EvalStep`] representing a discarded evaluation.
    ///
    /// Equivalent to `EvalStep(-9)` and converts to [`Step::Discard`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let step = EvalStep::discard();
    /// assert_eq!(step, EvalStep(-9));
    /// ```
    pub fn discard() -> Self {
        Self(-9)
    }

    /// Creates an [`EvalStep`] representing an evaluation error.
    ///
    /// Equivalent to `EvalStep(-10)` and converts to [`Step::Error`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let step = EvalStep::error();
    /// assert_eq!(step, EvalStep(-10));
    /// ```
    pub fn error() -> Self {
        Self(-10)
    }
}

impl std::fmt::Display for EvalStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            write!(f, "Pending")
        } else if self.0 == -1 {
            write!(f, "Evaluated")
        } else if self.0 == -9 {
            write!(f, "Discard")
        } else if self.0 == -10 {
            write!(f, "Error")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

pub mod obj;
pub use obj::{FuncWrapper, Objective, Stepped};

pub mod outcome;
pub use outcome::{FidOutcome, Outcome};
