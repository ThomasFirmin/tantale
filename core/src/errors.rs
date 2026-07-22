use std::fmt;

/// Generic trait for error messages in Tantale.
pub trait ErrMsg {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result;
}

#[macro_export]
macro_rules! err_msg {
    ($type: ty) => {
        impl std::error::Error for $type {}
        impl std::fmt::Display for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self._err_msg(f)
            }
        }
        impl std::fmt::Debug for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                self._err_msg(f)
            }
        }
    };
}

pub mod checkerrors;
pub use checkerrors::*;
pub mod domerrors;
pub use domerrors::*;
