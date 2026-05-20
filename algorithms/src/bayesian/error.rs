use tantale_core::errors::ErrMsg;
use std::{error::Error, fmt};

/// Error type for splitting operations from a [`Splitter`](crate::bayesian::splitter::Splitter).
pub struct SplitError(pub String);

impl Error for SplitError {}
impl ErrMsg for SplitError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = "Split error";
        write!(f, "{}, {}.", err_msg, self.0)
    }
}
impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
impl fmt::Debug for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}