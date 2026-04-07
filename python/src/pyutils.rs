use std::{io::Error, path::Path};

use pyo3::{Py, PyAny, PyErr, PyResult, Python, types::{PyAnyMethods, PyBytes}};

use crate::PY_OUTCOME_CLASS;

pub(crate) fn path_to_str(path: &Path) -> std::io::Result<String> {
    path.to_str()
        .ok_or_else(|| {
            Error::other(format!("{} is not a valid path", path.display()))
        })
        .map(str::to_owned)
}

pub(crate) fn py_to_io(e: PyErr) -> Error {
    Error::other(e.to_string())
}

pub(crate) fn pickle_dumps(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<Vec<u8>> {
    let pickle = py.import("pickle")?;
    pickle
        .call_method1("dumps", (obj.bind(py),))?
        .extract::<Vec<u8>>()
}

pub(crate) fn pickle_loads(py: Python<'_>, bytes: &[u8]) -> PyResult<Py<PyAny>> {
    let pickle = py.import("pickle")?;
    Ok(pickle
        .call_method1("loads", (PyBytes::new(py, bytes),))?
        .unbind())
}

/// Registers the Python outcome class used by [`CSVWritable::header`] for [`PyOutcome`].
///
/// The class must expose a `@staticmethod csv_header() -> list[str]` and
/// an instance method `csv_write(self) -> list[str]`.
pub fn register_outcome(cls: Py<PyAny>) {
    PY_OUTCOME_CLASS.with(|cell| *cell.borrow_mut() = Some(cls));
}

/// Initializes the Python environment and registers the specified Python modules and callables.
/// This macro should be called before the first `Python::attach` to ensure the modules are available for import.
/// 
/// # Arguments
/// * `Objective` or `Stepped` - Specifies whether to register an objective function or a stepped function.
/// * `$searchspace_module` - The Rust module containing the embedded Python module to register.
/// * `$pyconfig` - A [`PyConfig`] instance containing the configuration for the Python modules and callables to register.
#[macro_export]
macro_rules! init_python {
    (Objective, $searchspace_module:path, $func_file:expr, $func_module:expr, $func_name:expr, $out_file:expr, $out_module:expr, $out_name:expr) => {
        {
            use tantale::python::pyo3::{types::PyAnyMethods, ffi::c_str};
            use $searchspace_module::{pytantale};
            
            tantale::python::pyo3::append_to_inittab!(pytantale);
            let objective = tantale::python::pyo3::Python::attach(|py| {
                let out_mod = c_str!(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $out_file)));
                let cls = tantale::python::pyo3::types::PyModule::from_code(py, out_mod, c_str!($out_file), c_str!($out_module))?
                    .getattr($out_name)?.unbind();
                tantale::python::register_outcome(cls);

                let func_mod = c_str!(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $func_file)));
                let callable = tantale::python::pyo3::types::PyModule::from_code(py, func_mod, c_str!($func_file), c_str!($func_module))?
                    .getattr($func_name)?.unbind();
                tantale::python::pyo3::PyResult::Ok(tantale::python::PyObjective::new(callable))
            }).unwrap();
            objective.register()
        }
    };
    (Stepped, $searchspace_module:path, $func_file:expr, $func_module:expr, $func_name:expr, $out_file:expr, $out_module:expr, $out_name:expr) => {
        {
            use tantale::python::pyo3::{types::PyAnyMethods, ffi::c_str};
            use std::ffi::CStr;
            use $searchspace_module::{pytantale};

            tantale::python::pyo3::append_to_inittab!(pytantale);

            let objective = tantale::python::pyo3::Python::attach(|py| {
                let out_mod = c_str!(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $out_file)));
                let cls = tantale::python::pyo3::types::PyModule::from_code(py, out_mod, c_str!($out_file), c_str!($out_module))?
                    .getattr($out_name)?.unbind();
                tantale::python::register_outcome(cls);

                let func_mod = c_str!(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $func_file)));
                let callable = tantale::python::pyo3::types::PyModule::from_code(py, func_mod, c_str!($func_file), c_str!($func_module))?
                    .getattr($func_name)?.unbind();
                tantale::python::pyo3::PyResult::Ok(tantale::python::PyStepped::new(callable))
            }).unwrap();
            objective.register()
        }   
    };
}