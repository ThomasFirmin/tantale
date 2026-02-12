use std::fmt;

/// Generic trait for error messages in Tantale.
pub trait ErrMsg {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result;
}

pub mod checkerrors;
pub use checkerrors::*;
pub mod domerrors;
pub use domerrors::*;