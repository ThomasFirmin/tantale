pub struct SearchspaceMixed<T>(pub T);
pub trait Searchspace {
    type TypeSolObj;
    type TypeSolOpt;
    // fn sample_sol_obj(&self, rng: &mut ThreadRng) -> <Self::TypeObj as Domain>::TypeDom;
    // fn sample_sol_opt(&self, rng: &mut ThreadRng) -> <Self::TypeOpt as Domain>::TypeDom;
    // fn replicate(&self, name: &'a str) -> Self;
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
