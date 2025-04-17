use std::fmt;

pub trait ErrMsg {
    fn _err_msg(&self, f: &mut fmt::Formatter) -> fmt::Result;
}
