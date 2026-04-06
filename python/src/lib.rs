use std::cell::RefCell;
use pyo3::{Py, PyAny};


thread_local! {
    pub(crate) static PY_OBJECTIVE_FUNC: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
    pub(crate) static PY_STEPPED_FUNC: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
    pub(crate) static PY_OUTCOME_CLASS: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
}

pub mod pyoutcome;
pub use pyoutcome::{PyOutcome, PyStep};

pub mod pyfunction;
pub use pyfunction::{PyObjective, PyStepped};

pub mod pydomain;

pub mod pyutils;
pub use pyutils::{register_outcome, PyConfig};

pub use pyo3;