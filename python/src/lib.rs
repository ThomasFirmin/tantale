//! Python integration layer for Tantale.
//!
//! This crate provides the bridge between Tantale's Rust core and Python
//! user-defined objective functions.  It relies on [pyo3](https://pyo3.rs/) to
//! expose re-usable building blocks to define the function to optimize
//! in Python, while keeping the core logic and data structures in Rust for:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`pydomain`] | Converts Tantale domain values to Python objects |
//! | [`pyoutcome`] | Wraps Python-returned outcomes with the required Rust traits |
//! | [`pyfunction`] | Wraps Python callables as [`Objective`](tantale_core::Objective) / [`Stepped`](tantale_core::Stepped) |
//! | [`pyutils`] | Low-level helpers (pickle, path conversion, outcome class registration) |
//!
//! ## Thread-local state
//!
//! Because Tantale can evaluates objectives using multi-threading,
//! this crate stores the active callable and outcome class in
//! [`OnceLock`]s.  Call the appropriate `register` methods
//! ([`PyObjective::register`](pyfunction::PyObjective::register),
//! [`PyStepped::register`](pyfunction::PyStepped::register),
//! [`register_outcome`]) before spawning worker threads so that each thread
//! inherits its own copy.

use pyo3::{Py, PyAny};
use std::sync::OnceLock;

/// The Python callable registered as the current single-step objective.
///
/// Set by [`PyObjective::register`](crate::pyfunction::PyObjective::register).
/// Consumed inside [`py_objective`](crate::pyfunction::py_objective).
pub static PY_OBJECTIVE_FUNC: OnceLock<Py<PyAny>> = const { OnceLock::new() };

/// The Python callable registered as the current multi-fidelity stepped objective.
///
/// Set by [`PyStepped::register`](crate::pyfunction::PyStepped::register).
/// Consumed inside [`py_stepped`](crate::pyfunction::py_stepped).
pub static PY_STEPPED_FUNC: OnceLock<Py<PyAny>> = const { OnceLock::new() };

/// The Python outcome *class* (not an instance) used to derive CSV headers.
///
/// Set by [`register_outcome`].  Accessed inside [`PyOutcome`] and
/// [`PyFidOutcome`].
pub static PY_OUTCOME_CLASS: OnceLock<Py<PyAny>> = const { OnceLock::new() };

pub mod pyoutcome;
pub use pyoutcome::{PyFidOutcome, PyOutcome, PyStep};

pub mod pyfunction;
pub use pyfunction::{PyFuncState, PyObjective, PyStepped};

pub mod pydomain;

pub mod pyutils;
pub use pyutils::register_outcome;

pub use pyo3;
