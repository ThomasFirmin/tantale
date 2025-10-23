use crate::errors::ErrMsg;

use std::{error::Error, fmt};

pub struct DomainBoundariesError(pub String);

impl Error for DomainBoundariesError {}
impl ErrMsg for DomainBoundariesError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = "Boundaries error";
        write!(f, "{}, {}.", err_msg, self.0)
    }
}
impl fmt::Display for DomainBoundariesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
impl fmt::Debug for DomainBoundariesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}

pub struct DomainOoBError(pub String);

impl Error for DomainOoBError {}
impl ErrMsg for DomainOoBError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_msg = "Item out of bounds";
        write!(f, "{}, {}.", err_msg, self.0)
    }
}
impl fmt::Display for DomainOoBError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}
impl fmt::Debug for DomainOoBError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self._err_msg(f)
    }
}

pub enum DomainError {
    Bounds(DomainBoundariesError),
    OoB(DomainOoBError),
}

impl ErrMsg for DomainError {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let err_domain_msg = "A domain error occured.";
        match self {
            Self::Bounds(e) => {
                write!(f, "{}", err_domain_msg)?;
                e._err_msg(f)
            }
            Self::OoB(e) => {
                write!(f, "{}", err_domain_msg)?;
                e._err_msg(f)
            }
        }
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