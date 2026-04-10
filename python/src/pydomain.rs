use std::{convert::Infallible, sync::Arc};

use pyo3::{
    prelude::*,
    types::{DerefToPyAny, PyBool as PyO3Bool, PyFloat, PyInt as PyO3Int, PyList, PyString},
};
use serde::{Deserialize, Serialize};
use tantale_core::MixedTypeDom;

/// Bridges between [`TypeDom`] and Python objects.
pub trait ElementIntoPyObject
where
    Self: Sized + std::fmt::Debug + Serialize + for<'de> Deserialize<'de>,
{
    type Converted: DerefToPyAny;
    fn to_pyany<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, Infallible>;
}

/// Converts a f64 value to a Python `float`.
impl ElementIntoPyObject for f64 {
    type Converted = PyFloat;
    fn to_pyany<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, Infallible> {
        Ok(PyFloat::new(py, *self).into_any())
    }
}

/// Converts an i64 type to a Python `int`.
impl ElementIntoPyObject for i64 {
    type Converted = PyO3Int;
    fn to_pyany<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, Infallible> {
        Ok(PyO3Int::new(py, *self).into_any())
    }
}

/// Convert a Rust u64 to a Python int.
impl ElementIntoPyObject for u64 {
    type Converted = PyO3Int;
    fn to_pyany<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, Infallible> {
        Ok(PyO3Int::new(py, *self).into_any())
    }
}

/// Converts a `bool` to a Python `bool`.
impl ElementIntoPyObject for bool {
    type Converted = PyO3Bool;
    fn to_pyany<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, Infallible> {
        Ok(<Bound<'_, PyO3Bool> as Clone>::clone(&PyO3Bool::new(py, *self)).into_any())
    }
}

/// Converts a Rust string slice to a Python `str`.
impl ElementIntoPyObject for String {
    type Converted = PyString;
    fn to_pyany<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, Infallible> {
        Ok(PyString::new(py, self).into_any())
    }
}

/// Converts a single [`MixedTypeDom`] value to its Python equivalent.
///
/// | Rust variant            | Python type |
/// |-------------------------|-------------|
/// | [`Real`](tantale_core::Real) / [`Unit`](tantale_core::Unit) / [`GridReal`](tantale_core::GridReal) | `float` |
/// | [`Nat`](tantale_core::Nat) / [`GridNat`](tantale_core::GridNat)       | `int`   |
/// | [`Int`](tantale_core::Int) / [`GridInt`](tantale_core::GridInt)       | `int`   |
/// | [`Bool`](tantale_core::Bool)                  | `bool`  |
/// | [`Cat`](tantale_core::Cat)                   | `str`   |
impl ElementIntoPyObject for MixedTypeDom {
    type Converted = PyString;
    fn to_pyany<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, Infallible> {
        match self {
            MixedTypeDom::Real(f) | MixedTypeDom::Unit(f) | MixedTypeDom::GridReal(f) => {
                f.to_pyany(py)
            }
            MixedTypeDom::Nat(i) | MixedTypeDom::GridNat(i) => i.to_pyany(py),
            MixedTypeDom::Int(i) | MixedTypeDom::GridInt(i) => i.to_pyany(py),
            MixedTypeDom::Bool(b) => b.to_pyany(py),
            MixedTypeDom::Cat(s) => s.to_pyany(py),
        }
    }
}

/// Converts a slice of [`MixedTypeDom`] values to a Python `list`.
impl<E: ElementIntoPyObject> ElementIntoPyObject for Arc<[E]> {
    type Converted = PyList;
    fn to_pyany<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyAny>, Infallible> {
        let list = PyList::empty(py);
        for e in self.iter() {
            list.append(e.to_pyany(py)?)
                .expect("failed to append to Python list");
        }
        Ok(list.into_any())
    }
}
