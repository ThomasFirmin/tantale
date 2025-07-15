use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, Unit, TypeDom};
use tantale_core::solution::{Solution,PartialSol,Partial};

use std::fmt::{Debug, Display};
use std::collections::HashSet;
use std::process;

use super::init_sinfo::{TestSInfo,get_sinfo};

fn _test_solution_assertion<Dom, const N: usize>(
    sol: &PartialSol<Dom,TestSInfo,N>,
    id: (u32,usize),
) where
    Dom : Domain + Clone + Display + Debug,
    TypeDom<Dom> : Sync + Send,
{
    assert_eq!(
        sol.get_x(),
        std::sync::Arc::from(vec![Dom::TypeDom::default(); N]),
        "Solution `x` mismatch."
    );
    assert_eq!(sol.get_id().0, id.0, "Solution PID mismatch.");
    assert_eq!(
        sol.get_info().info,
        42.0,
        "Wrong solution info from TestSInfo."
    );
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_sol {
    ($name:ident ; ($($dom:ty),+) ; $size:expr ; ($($id:expr),+),$pid:expr) => {
        #[test]
        fn $name (){
            let sinfo = std::sync::Arc::new(get_sinfo());
            let mut idsol = Vec::new();
            $(
                let sol = PartialSol::<$dom,TestSInfo,$size>::new_default($pid,sinfo.clone());
                _test_solution_assertion(&sol, ($pid,$id));
                idsol.push(sol.get_id().1);
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));

            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_sol!(mixed_size_3 ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; (0,1,2,3,4,5),process::id());
