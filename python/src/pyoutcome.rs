use std::fmt::{self};

use crate::{
    PY_OUTCOME_CLASS,
    pyutils::{pickle_dumps, pickle_loads},
};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use tantale_core::{CSVWritable, EvalStep, FidOutcome, Outcome, Step};

/// Python-accessible wrapper for the [`Step`] evaluation state.
#[pyclass(module = "pytantale", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStep {
    pub inner: Step,
}

#[pymethods]
impl PyStep {
    /// Creates a [`Step::Pending`] value.
    #[staticmethod]
    pub fn pending() -> Self {
        Self {
            inner: Step::Pending,
        }
    }
    /// Creates a [`Step::Evaluated`] value.
    #[staticmethod]
    pub fn evaluated() -> Self {
        Self {
            inner: Step::Evaluated,
        }
    }
    /// Creates a [`Step::Partially`]`(value)` value.
    #[staticmethod]
    pub fn partially(value: isize) -> Self {
        assert!(value > 0, "partial step value must be positive");
        Self {
            inner: Step::Partially(value),
        }
    }
    /// Creates a [`Step::Error`] value.
    #[staticmethod]
    pub fn error() -> Self {
        Self { inner: Step::Error }
    }
    /// Creates a [`Step::Discard`] value.
    #[staticmethod]
    pub fn discarded() -> Self {
        Self {
            inner: Step::Discard,
        }
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<pyo3::Bound<'py, pyo3::types::PyTuple>> {
        use pyo3::types::PyTuple;
        let cls = py.get_type::<PyStep>();
        let method_name = match self.inner {
            Step::Pending => "pending",
            Step::Partially(_) => "partially",
            Step::Evaluated => "evaluated",
            Step::Discard => "discarded",
            Step::Error => "error",
        };
        let method = cls.getattr(method_name)?;
        let args: pyo3::Bound<'py, PyTuple> = match self.inner {
            Step::Partially(v) => PyTuple::new(py, [v])?,
            _ => PyTuple::empty(py),
        };
        PyTuple::new(py, [method.into_any(), args.into_any()])
    }
}

/// Rust bridge for an arbitrary Python outcome object.
///
/// Wraps any Python object returned by a Python objective function.
/// Required trait bounds ([`Serialize`], [`Deserialize`], [`Debug`]) are satisfied
/// via Python's `pickle` module for serialization and `repr()` for debug output.
///
/// Implements [`Outcome`], making it usable as the `Out` type parameter of
/// [`Objective`]`<Arc<[`[`MixedTypeDom`]`]>, PyOutcome>`.
pub struct PyOutcome(pub Py<PyAny>);

impl fmt::Debug for PyOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Python::attach(|py| {
            let r = self
                .0
                .bind(py)
                .repr()
                .map(|r| r.to_string())
                .unwrap_or_default();
            write!(f, "PyOutcome({r})")
        })
    }
}

impl Serialize for PyOutcome {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = Python::attach(|py| pickle_dumps(py, &self.0))
            .map_err(|e| serde::ser::Error::custom(e.to_string()))?;
        bytes.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyOutcome {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Python::attach(|py| pickle_loads(py, &bytes).map(PyOutcome))
            .map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

impl CSVWritable<(), ()> for PyOutcome {
    fn header(_elem: &()) -> Vec<String> {
        PY_OUTCOME_CLASS.with(|cell| {
            Python::attach(|py| {
                cell.borrow()
                    .as_ref()
                    .and_then(|cls| {
                        cls.bind(py)
                            .call_method0("csv_header")
                            .and_then(|v| v.extract::<Vec<String>>())
                            .ok()
                    })
                    .unwrap()
            })
        })
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method0("csv_write")
                .and_then(|v| v.extract::<Vec<String>>())
                .unwrap()
        })
    }
}

impl Outcome for PyOutcome {}

impl PyOutcome {
    /// Extracts a named attribute from the Python outcome as `f64`.
    ///
    /// Mirrors direct field access on typed outcomes, enabling identical
    /// codomain closure syntax:
    /// ```rust,ignore
    /// |o: &PyOutcome| o.getattr_f64("obj1")
    /// ```
    /// Returns `f64::NAN` if the attribute is missing or not numeric.
    pub fn getattr_f64(&self, attr: &str) -> f64 {
        Python::attach(|py| {
            self.0
                .bind(py)
                .getattr(attr)
                .and_then(|v| v.extract::<f64>())
                .unwrap_or(f64::NAN)
        })
    }
}

/// Rust bridge for a Python outcome in a multi-fidelity objective.
///
/// Extends [`PyOutcome`] with [`FidOutcome`] by delegating `get_step()` to the
/// wrapped Python object.  The Python class must expose a `get_step()` method
/// that returns a [`PyStep`] instance.
///
/// Implements [`Outcome`] and [`FidOutcome`], making it usable as the `Out` type
/// parameter of [`Stepped`]`<Arc<[`[`MixedTypeDom`]`]>, PyFidOutcome, PyFuncState>`.
pub struct PyFidOutcome(pub Py<PyAny>);

impl fmt::Debug for PyFidOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Python::attach(|py| {
            let r = self
                .0
                .bind(py)
                .repr()
                .map(|r| r.to_string())
                .unwrap_or_default();
            write!(f, "PyFidOutcome({r})")
        })
    }
}

impl Serialize for PyFidOutcome {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = Python::attach(|py| pickle_dumps(py, &self.0))
            .map_err(|e| serde::ser::Error::custom(e.to_string()))?;
        bytes.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyFidOutcome {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Python::attach(|py| pickle_loads(py, &bytes).map(PyFidOutcome))
            .map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

impl Outcome for PyFidOutcome {}

impl CSVWritable<(), ()> for PyFidOutcome {
    fn header(_elem: &()) -> Vec<String> {
        PY_OUTCOME_CLASS.with(|cell| {
            Python::attach(|py| {
                cell.borrow()
                    .as_ref()
                    .and_then(|cls| {
                        cls.bind(py)
                            .call_method0("csv_header")
                            .and_then(|v| v.extract::<Vec<String>>())
                            .ok()
                    })
                    .unwrap()
            })
        })
    }

    fn write(&self, _comp: &()) -> Vec<String> {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method0("csv_write")
                .and_then(|v| v.extract::<Vec<String>>())
                .unwrap()
        })
    }
}

impl PyFidOutcome {
    /// Extracts a named attribute from the Python outcome as `f64`.
    ///
    /// ```rust,ignore
    /// |o: &PyFidOutcome| o.getattr_f64("obj1")
    /// ```
    /// Returns `f64::NAN` if the attribute is missing or not numeric.
    pub fn getattr_f64(&self, attr: &str) -> f64 {
        Python::attach(|py| {
            self.0
                .bind(py)
                .getattr(attr)
                .and_then(|v| v.extract::<f64>())
                .unwrap_or(f64::NAN)
        })
    }
}

impl FidOutcome for PyFidOutcome {
    /// Calls `self.get_step()` on the Python object and converts the returned
    /// [`PyStep`] to an [`EvalStep`].  Returns [`EvalStep::error()`] if the call fails.
    fn get_step(&self) -> EvalStep {
        Python::attach(|py| {
            self.0
                .bind(py)
                .getattr("step")
                .and_then(|s| Ok(s.extract::<PyStep>()?))
                .map(|s| s.inner.into())
                .unwrap_or_else(|_| EvalStep::error())
        })
    }
}
