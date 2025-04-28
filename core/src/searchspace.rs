use crate::domain::Domain;
use crate::solution::Solution;
use crate::variable::Var;

use std::fmt::{Debug, Display};

pub trait Searchspace<Obj, Opt>
where
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
{
    fn get_variables(&self) -> Vec<Var<Obj, Opt>>;
    fn onto_obj(&self, item: Solution<Obj>) -> Solution<Opt>;
    fn onto_opt(&self, item: Solution<Opt>) -> Solution<Obj>;
    fn sample_obj(&self) -> Solution<Obj>;
    fn sample_opt(&self) -> Solution<Opt>;
}

pub struct SearchspaceSingle<'a, Obj>
where
    Obj: Domain + Clone + Display + Debug,
{
    pub variables: Var<'a, Obj>,
}

#[macro_export]
macro_rules! sp {
    // Defining both objective and optimizer domains
    // Defining both samplers
    ($($x:expr),+) => {{
        use $crate::core::searchspace::{SearchspaceMixed, Searchspace};
        type sptype = ($($x),+)
        SearchspaceMixed([$($x),+])
    }};
}
