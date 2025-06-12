use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, Unit, TypeDom};
use tantale::core::Solution;
use tantale::core::{Codomain, Outcome, SingleCodomain, HashOut,EmptyInfo};

use std::fmt::{Debug, Display};
use std::collections::HashSet;
use std::process;

fn _test_solution_assertion<Dom, Cod, Out, const S: usize>(
    sol: &Solution<Dom, Cod, Out, EmptyInfo, S>,
    id: (usize, u32),
) where
    Dom : Domain + Clone + Display + Debug,
    Cod : Codomain<Out>,
    Out : Outcome,
    TypeDom<Dom> : Sync + Send,
{
    assert_eq!(
        sol.x,
        vec![Dom::TypeDom::default(); S].into_boxed_slice(),
        "Solution `x` mismatch."
    );
    // assert_eq!(sol.id.0, id.0, "Solution `id` mismatch.");
    assert_eq!(sol.id.1, id.1, "Solution `pid` mismatch.");
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_sol {
    ($name:ident ; ($($dom:ty),+) ; $size:expr ; ($($id:expr),+),$pid:expr) => {
        #[test]
        fn $name (){
            let mut idsol = Vec::new();
            $(
                let sol = Solution::<$dom, SingleCodomain<HashOut>, HashOut, EmptyInfo, $size>::new_default($pid);
                _test_solution_assertion(&sol, ($id, $pid));
                idsol.push(sol.id.0);
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));

            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_sol!(mixed_size_3 ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; (0,1,2,3,4,5),process::id());
