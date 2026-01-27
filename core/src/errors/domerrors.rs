use crate::errors::ErrMsg;

use std::{error::Error, fmt};

/// Error type for [`Onto`](crate::domain::onto::Onto) operations between domains.
pub struct OntoError(pub String);

impl Error for OntoError {}
impl ErrMsg for OntoError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = "Onto error";
        write!(f, "{}, {}.", err_msg, self.0)
    }
}
impl fmt::Display for OntoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
impl fmt::Debug for OntoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
