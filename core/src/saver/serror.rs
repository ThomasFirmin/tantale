use crate::errors::ErrMsg;

use std::{error::Error, fmt};

pub struct CheckpointError(pub String);

impl Error for CheckpointError {}
impl ErrMsg for CheckpointError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = "Checkpoint error";
        write!(f, "{}, {}.", err_msg, self.0)
    }
}
impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
impl fmt::Debug for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
