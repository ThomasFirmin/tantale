//! Python function wrappers for Tantale objectives.
//!
//! This module bridges Python callables to Tantale's [`Objective`] and
//! [`Stepped`] interfaces. The bridge works in three layers:
//!
//! 1. **Thread-local storage** – the active Python callable is stored in a
//!    thread-local [`RefCell`](std::cell::RefCell) (see [`crate::PY_OBJECTIVE_FUNC`] /
//!    [`crate::PY_STEPPED_FUNC`]).  Calling `register()` on a wrapper type
//!    writes the callable to the thread-local and returns a typed Rust
//!    function pointer.
//!
//! 2. **Free functions** – [`py_objective`] and [`py_stepped`] are plain Rust
//!    `fn` pointers that read from the thread-local, convert the Rust solution
//!    to a Python `list` via [`ElementIntoPyObject`], invoke the callable,
//!    and wrap the return value.
//!
//! 3. **`#[pyclass]` wrappers** – [`PyObjective`] and [`PyStepped`] are pyo3
//!    classes exposed so that the `init_python!` macro can construct them from
//!    Python callables retrieved at runtime.
//!
//! ## Typical usage
//!
//! The `init_python!` macro (defined in [`pyutils`](crate::pyutils)) calls
//! `PyObjective::register` or `PyStepped::register` and returns the
//! resulting typed Tantale objective, ready to be passed to the optimizer.

use std::{
    fmt::{self, Display},
    path::PathBuf,
    sync::Arc,
};

use pyo3::prelude::*;
use tantale_core::{Fidelity, FuncState, Objective, Stepped};

use crate::{
    PY_OBJECTIVE_FUNC, PY_STEPPED_FUNC,
    pydomain::ElementIntoPyObject,
    pyoutcome::{PyFidOutcome, PyOutcome},
    pyutils::{path_to_str, pickle_dumps, pickle_loads, py_to_io},
};

/// Rust bridge for a Python function state used in multi-fidelity objectives.
///
/// Implements [`FuncState`] by delegating to methods on the Python object:
///
/// - [`save`](FuncState::save) calls `state.save(path: str)` and also writes a
///   `_tantale_cls.pkl` file (a pickle of the Python class) alongside the user
///   data, so that [`load`](FuncState::load) can reconstruct the correct class.
/// - [`load`](FuncState::load) reads `_tantale_cls.pkl`, unpickles the class,
///   then calls `cls.load(path: str)`.
///
/// ## Python contract
///
/// ```python
/// class MyState:
///     def save(self, path: str) -> None: ...
///
///     @staticmethod           # or @classmethod
///     def load(path: str) -> "MyState": ...
/// ```
pub struct PyFuncState(pub Py<PyAny>);

impl FuncState for PyFuncState {
    fn save(&self, path: PathBuf) -> std::io::Result<()> {
        let path_str = path_to_str(&path)?;
        Python::attach(|py| -> std::io::Result<()> {
            self.0
                .call_method1(py, "save", (&path_str,))
                .map_err(py_to_io)?;
            // Save the class so load() can reconstruct it.
            let cls = self.0.bind(py).get_type().into_any().unbind();
            let cls_bytes = pickle_dumps(py, &cls).map_err(py_to_io).unwrap();
            let path_cls = path.join("_tantale_cls.pkl");
            std::fs::write(path_cls, cls_bytes)
        })
    }

    fn load(path: PathBuf) -> std::io::Result<Self> {
        let cls_bytes = std::fs::read(path.join("_tantale_cls.pkl"))?;
        let path_str = path_to_str(&path)?;
        Python::attach(|py| -> std::io::Result<Self> {
            let cls = pickle_loads(py, &cls_bytes).map_err(py_to_io).unwrap();
            let obj = cls
                .call_method1(py, "load", (&path_str,))
                .map_err(py_to_io)
                .unwrap();
            Ok(PyFuncState(obj))
        })
    }
}

impl Display for PyFuncState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Python::attach(|py| {
            let r = self
                .0
                .bind(py)
                .repr()
                .map(|r| r.to_string())
                .unwrap_or_default();
            write!(f, "PyFuncState({r})")
        })
    }
}

/// Calls the thread-local Python objective with a typed solution slice.
///
/// Converts `x` to a Python `list` via [`ElementIntoPyObject::to_pyany`],
/// calls the callable stored in [`PY_OBJECTIVE_FUNC`](crate::PY_OBJECTIVE_FUNC),
/// and wraps the return value in a [`PyOutcome`].
///
/// # Panics
/// Panics if no callable has been registered via [`PyObjective::register`].
pub fn py_objective<E: ElementIntoPyObject>(x: Arc<[E]>) -> PyOutcome {
    let binding = PY_OBJECTIVE_FUNC.get().expect("The PyObjective was not initialized");
    Python::attach(|py| {
        let list = x.to_pyany(py).expect("failed to build Python input list");
        let result = binding.call1(py, (list,)).unwrap();
        PyOutcome(result)
    })
}

/// Calls the thread-local Python stepped objective with a solution, fidelity budget, and optional state.
///
/// Converts `x` to a Python `list`, passes `fidelity.0` as a `float`, and
/// passes `state` (if any) as the Python state object stored in [`PyFuncState`].
/// Returns a `(PyFidOutcome, PyFuncState)` reconstructed from the Python
/// 2-tuple `(outcome, new_state)` returned by the callable.
///
/// # Panics
/// Panics if no callable has been registered via [`PyStepped::register`],
/// or if the Python callable does not return a 2-tuple.
pub fn py_stepped<E: ElementIntoPyObject>(
    x: Arc<[E]>,
    fidelity: Fidelity,
    state: Option<PyFuncState>,
) -> (PyFidOutcome, PyFuncState) {
    let binding = PY_STEPPED_FUNC.get().expect("The PyStepped was not initialized");
    let py_state = state.map(|s| s.0);
    Python::attach(|py| {
        let list = x.to_pyany(py).expect("failed to build Python input list");
        let result = binding
            .call1(py, (list, fidelity.0, py_state))
            .expect("Python stepped objective raised an error");
        let (out, new_state): (Py<PyAny>, Py<PyAny>) = result
            .extract(py)
            .expect("Python stepped objective must return a 2-tuple (outcome, state)");
        (PyFidOutcome(out), PyFuncState(new_state))
    })
}

/// Python wrapper for a single-shot objective function.
///
/// Accepts any Python callable with the signature:
///
/// ```python
/// def my_func(x: list) -> outcome:
///     ...
/// ```
///
/// where `x` is a Python list of `float | int | bool | str` values derived from
/// the [`MixedTypeDom`] search space, and the return value is any Python object that
/// the configured codomain can extract from.
#[pyclass]
pub struct PyObjective(pub Py<PyAny>);

#[pymethods]
impl PyObjective {
    #[new]
    pub fn new(callable: Py<PyAny>) -> Self {
        PyObjective(callable)
    }
}

impl PyObjective {
    /// Registers this callable as the active thread-local objective and returns
    /// an [`Objective`]`<Arc<[`[`MixedTypeDom`]`]>, `[`PyOutcome`]`>`.
    ///
    /// Only one `PyObjective` may be active per thread at a time.
    pub fn register<E: ElementIntoPyObject>(&self) -> Objective<Arc<[E]>, PyOutcome> {
        Python::attach(|py| {
            PY_OBJECTIVE_FUNC.set(self.0.clone_ref(py)).expect("Failed to register Python objective function: a function has already been registered");
        });
        Objective::new(py_objective)
    }
}

/// Python wrapper for a multi-fidelity stepped objective function.
///
/// Accepts any Python callable with the signature:
///
/// ```python
/// def my_stepped(x: list, fidelity: float, state) -> tuple:
///     ...
///     return outcome, new_state
/// ```
///
/// - `x` – Python list of `float | int | bool | str` inputs.
/// - `fidelity` – Current fidelity budget as a `float`.
/// - `state` – Previous function state (any object with `save`/`load` methods),
///   or `None` on the first call.
/// - Returns a 2-tuple `(outcome, new_state)`.
///
/// Call [`register`](PyStepped::register) (from Rust) to register the callable and
/// obtain a typed [`Stepped`]`<Arc<[`[`MixedTypeDom`]`]>, `[`PyFidOutcome`]`, `[`PyFuncState`]`>`.
#[pyclass]
pub struct PyStepped(pub Py<PyAny>);

#[pymethods]
impl PyStepped {
    #[new]
    pub fn new(callable: Py<PyAny>) -> Self {
        PyStepped(callable)
    }
}

impl PyStepped {
    /// Registers this callable as the active thread-local stepped objective and
    /// returns a [`Stepped`]`<Arc<[`[`MixedTypeDom`]`]>, `[`PyFidOutcome`]`, `[`PyFuncState`]`>`.
    ///
    /// Only one `PyStepped` may be active per thread at a time.
    pub fn register<E: ElementIntoPyObject>(&self) -> Stepped<Arc<[E]>, PyFidOutcome, PyFuncState> {
        Python::attach(|py| {
            PY_STEPPED_FUNC.set(self.0.clone_ref(py)).expect("Failed to register Python stepped objective function: a function has already been registered");
        });
        Stepped::new(py_stepped)
    }
}
