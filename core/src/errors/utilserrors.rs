use crate::errors::ErrMsg;

use std::{error::Error, fmt};

pub struct DifferentLengthError(pub String);

impl Error for DifferentLengthError {}
impl ErrMsg for DifferentLengthError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = "Checkpoint error";
        write!(f, "{}, {}.", err_msg, self.0)
    }
}
impl fmt::Display for DifferentLengthError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
impl fmt::Debug for DifferentLengthError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
