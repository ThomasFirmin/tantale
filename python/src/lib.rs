use pyo3::{Py, PyAny};
use std::cell::RefCell;

thread_local! {
    pub(crate) static PY_OBJECTIVE_FUNC: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
    pub(crate) static PY_STEPPED_FUNC: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
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
