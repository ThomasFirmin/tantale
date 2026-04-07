use std::{fmt::{self, Display}, path::PathBuf, sync::Arc};

use pyo3::prelude::*;
use tantale_core::{Fidelity, FuncState, Objective, Stepped};

use crate::{
    PY_OBJECTIVE_FUNC, PY_STEPPED_FUNC,
    pydomain::ElementIntoPyObject,
    pyoutcome::{PyFidOutcome, PyOutcome},
    pyutils::{path_to_str, pickle_dumps, pickle_loads, py_to_io}
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
                .map_err(py_to_io).unwrap();
            Ok(PyFuncState(obj))
        })
    }
}

impl Display for PyFuncState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Python::attach(|py| {
            let r = self.0.bind(py).repr().map(|r| r.to_string()).unwrap_or_default();
            write!(f, "PyFuncState({r})")
        })
    }
}

pub fn py_objective<E: ElementIntoPyObject>(x: Arc<[E]>) -> PyOutcome {
    PY_OBJECTIVE_FUNC.with(|cell| {
        let borrow = cell.borrow();
        let callable = borrow
            .as_ref()
            .expect("no PyObjective active; call PyObjective::activate() first");
        Python::attach(|py| {
            let list = x.to_pyany(py).expect("failed to build Python input list");
            let result = callable
                .call1(py, (list,))
                .unwrap();
            PyOutcome(result)
        })
    })
}

pub fn py_stepped<E:ElementIntoPyObject>(
    x: Arc<[E]>,
    fidelity: Fidelity,
    state: Option<PyFuncState>,
) -> (PyFidOutcome, PyFuncState) {
    Python::attach(|py| {
        PY_STEPPED_FUNC.with(|cell|{
            let borrow = cell.borrow();
            let callable = borrow
                .as_ref()
                .expect("no PyStepped active; call PyStepped::activate() first");
            let list = x.to_pyany(py).expect("failed to build Python input list");
            let py_state = state.map(|s| s.0);
            let result = callable
                .call1(py, (list, fidelity.0, py_state))
                .expect("Python stepped objective raised an error");
            let (out, new_state): (Py<PyAny>, Py<PyAny>) = result
                .extract(py)
                .expect("Python stepped objective must return a 2-tuple (outcome, state)");
            (PyFidOutcome(out), PyFuncState(new_state))
        })
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
    pub fn register<E:ElementIntoPyObject>(&self) -> Objective<Arc<[E]>, PyOutcome> {
        Python::attach(|py| {
            PY_OBJECTIVE_FUNC.with(|cell| {
                *cell.borrow_mut() = Some(self.0.clone_ref(py));
            });
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
            PY_STEPPED_FUNC.with(|cell| {
                *cell.borrow_mut() = Some(self.0.clone_ref(py));
            });
        });
        Stepped::new(py_stepped)
    }
}