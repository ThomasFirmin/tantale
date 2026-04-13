//! Python integration layer for Tantale.
//!
//! This crate provides the bridge between Tantale's Rust core and Python
//! user-defined objective functions.  It relies on [pyo3](https://pyo3.rs/) to
//! expose re-usable building blocks to define the function to optimize
//! in Python, while keeping the core logic and data structures in Rust for:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`pydomain`](crate::pydomain) | Converts Tantale domain values to Python objects |
//! | [`pyoutcome`](crate::pyoutcome) | Wraps Python-returned outcomes with the required Rust traits |
//! | [`pyfunction`](crate::pyfunction) | Wraps Python callables as [`Objective`](tantale_core::Objective) / [`Stepped`](tantale_core::Stepped) |
//! | [`pyutils`](crate::pyutils) | Low-level helpers (pickle, path conversion, outcome class registration) |
//!
//! ## Thread-local state
//!
//! Because Tantale can evaluates objectives using multi-threading,
//! this crate stores the active callable and outcome class in
//! *thread-local* [`RefCell`]s.  Call the appropriate `register` methods
//! ([`PyObjective::register`](pyfunction::PyObjective::register),
//! [`PyStepped::register`](pyfunction::PyStepped::register),
//! [`register_outcome`]) before spawning worker threads so that each thread
//! inherits its own copy.

use pyo3::{Py, PyAny};
use std::cell::RefCell;

thread_local! {
    /// The Python callable registered as the current single-step objective.
    ///
    /// Set by [`PyObjective::register`](crate::pyfunction::PyObjective::register).
    /// Consumed inside [`py_objective`](crate::pyfunction::py_objective).
    pub(crate) static PY_OBJECTIVE_FUNC: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };

    /// The Python callable registered as the current multi-fidelity stepped objective.
    ///
    /// Set by [`PyStepped::register`](crate::pyfunction::PyStepped::register).
    /// Consumed inside [`py_stepped`](crate::pyfunction::py_stepped).
    pub(crate) static PY_STEPPED_FUNC: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };

    /// The Python outcome *class* (not an instance) used to derive CSV headers.
    ///
    /// Set by [`register_outcome`].  Accessed inside [`PyOutcome`](crate::pyoutcome::PyOutcome) and
    /// [`PyFidOutcome`](crate::pyoutcome::PyFidOutcome).
    pub(crate) static PY_OUTCOME_CLASS: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
}

pub mod pyoutcome;
pub use pyoutcome::{PyFidOutcome, PyOutcome, PyStep};

pub mod pyfunction;
pub use pyfunction::{PyObjective, PyStepped};

pub mod pydomain;

pub mod pyutils;
pub use pyutils::register_outcome;

pub use pyo3;
