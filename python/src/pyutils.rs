use std::{fs, io::Error, path::Path};

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

/// Configuration for initializing Python modules and callables.
#[derive(Debug, Clone)]
pub struct PyConfig{
    /// Absolute path to the Python module containing the objective function and outcome class.
    pub function_module: String,
    /// Name of the Python callable to use as the objective function.
    pub function_name: String,
    /// Absolute path to the Python module containing the outcome class.
    /// Defaults to `function_module` if not specified.
    pub outcome_module: String,
    /// Name of the Python class to use as the outcome class.
    pub outcome_name: String,
    /// The parent directories of the function modules, used for extending `sys.path`.
    pub function_module_parent: String,
    /// The parent directories of the outcome modules, used for extending `sys.path`.
    pub outcome_module_parent: String,
}

impl PyConfig {
    /// Creates a new `PyConfig` with the specified function module, function name, and outcome name.
    /// The `outcome_module` defaults to the same path as `function_module`.
    /// All module paths are converted to absolute paths.
    /// 
    /// # Arguments
    /// * `module_path` - Path to the Python module containing the function and outcome class.
    /// * `function_name` - Name of the Python callable to use as the objective function.
    /// * `outcome_name` - Name of the Python class to use as the outcome class.
    pub fn new(module_path: &str, function_name: &str, outcome_name: &str) -> Self {
        let module_path = String::from(module_path);
        let function_module_parent = Path::new(&module_path).parent().unwrap().to_str().unwrap().to_string();
        Self {
            function_module: module_path.clone(),
            function_name: String::from(function_name),
            outcome_module: module_path,
            outcome_name: String::from(outcome_name),
            function_module_parent: function_module_parent.clone(),
            outcome_module_parent: function_module_parent,
        }
    }
    /// Sets the `outcome_module` to a different path than `function_module`.
    /// The provided path is converted to an absolute path.
    /// 
    /// # Arguments
    /// * `module_path` - Path to the Python module containing the outcome class.
    pub fn with_outcome_module(mut self, module_path: &str) -> Self {
        let module_path = String::from(module_path);
        self.outcome_module_parent = Path::new(&module_path).parent().unwrap().to_str().unwrap().to_string();
        self
    }
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
    (Objective, $searchspace_module:path, $pyconfig:expr) => {
        {
            use $searchspace_module::{pytantale};
            tantale::python::pyo3::append_to_inittab!(pytantale);
            let objective = tantale::python::pyo3::Python::attach(|py| {
                let sys = py.import("sys")?;
                let test_dir = env!("CARGO_MANIFEST_DIR").to_string() + &$pyconfig.function_module_parent;
                sys.getattr("path")?.call_method1("append", (test_dir,))?;
                let test_dir = env!("CARGO_MANIFEST_DIR").to_string() + &$pyconfig.outcome_module_parent;
                sys.getattr("path")?.call_method1("append", (test_dir,))?;
                
                let module_name = std::path::Path::new(&$pyconfig.outcome_module).file_stem().unwrap().to_str().unwrap();
                let module = py.import(module_name)?;
                let cls = module.getattr($pyconfig.outcome_name)?.unbind();
                tantale::python::register_outcome(cls);

                let module_name = std::path::Path::new(&$pyconfig.outcome_module).file_stem().unwrap().to_str().unwrap();
                let module = py.import(module_name)?;
                let callable = module.getattr($pyconfig.function_name)?.unbind();
                tantale::python::pyo3::PyResult::Ok(tantale::python::PyObjective::new(callable))
            }).unwrap();
            objective.register()
        }
    };
    (Stepped, $searchspace_module:path, $pyconfig:expr) => {
        {
            use $searchspace_module::{pytantale};
            tantale::python::pyo3::append_to_inittab!(pytantale);
            let objective = tantale::python::pyo3::Python::attach(|py| {
                let sys = py.import("sys")?;
                let test_dir = env!("CARGO_MANIFEST_DIR").to_string() + &$pyconfig.function_module_parent;
                sys.getattr("path")?.call_method1("append", (test_dir,))?;
                let test_dir = env!("CARGO_MANIFEST_DIR").to_string() + &$pyconfig.outcome_module_parent;
                sys.getattr("path")?.call_method1("append", (test_dir,))?;
                
                let module_name = std::path::Path::new(&$pyconfig.outcome_module).file_stem().unwrap().to_str().unwrap();
                let module = py.import(module_name)?;
                let cls = module.getattr($pyconfig.outcome_name)?.unbind();
                tantale::python::register_outcome(cls);

                let module_name = std::path::Path::new(&$pyconfig.outcome_module).file_stem().unwrap().to_str().unwrap();
                let module = py.import(module_name)?;
                let callable = module.getattr($pyconfig.function_name)?.unbind();
                tantale::python::pyo3::PyResult::Ok(tantale::python::PyStepped::new(callable))
            }).unwrap();
            objective.register()
        }   
    };
}