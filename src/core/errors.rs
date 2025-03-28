use std::{error::Error, fmt};

trait ErrMsg {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result;
}

pub struct DomainError {
    pub code: usize,
    pub msg: String,
}

impl Error for DomainError {}
impl ErrMsg for DomainError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = match self.code {
            100 => "Boundaries error",
            101 => "Mapping not implemented",
            102 => "Resulting mapping out of bounds",
            103 => "Input out of bounds",
            _ => "Went wrong",
        };

        write!(f, "{}, {}.", err_msg, self.msg)
    }
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
impl fmt::Debug for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}

#[derive(Debug)]
pub struct VariableError {
    pub code: usize,
    pub msg: String,
}

impl Error for VariableError {}

impl fmt::Display for VariableError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = match self.code {
            100 => "Initialization error",
            _ => "Went wrong.",
        };

        write!(f, "{}, {}", err_msg, self.msg)
    }
}
