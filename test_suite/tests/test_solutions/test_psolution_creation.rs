use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, TypeDom, Unit};
use tantale_core::solution::{Partial, PartialSol, Solution,ParSId};

use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::process;

use super::init_sinfo::{get_sinfo, TestSInfo};

fn _test_solution_assertion<Dom>(n: usize, sol: &PartialSol<ParSId, Dom, TestSInfo>, pid: u32)
where
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>: Sync + Send,
{
    assert_eq!(
        sol.get_x(),
        std::sync::Arc::from(vec![Dom::TypeDom::default(); n]),
        "Solution `x` mismatch."
    );
    assert_eq!(sol.get_id().pid, pid, "Solution PID mismatch.");
    assert_eq!(
        sol.get_info().info,
        42.0,
        "Wrong solution info from TestSInfo."
    );
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_sol {
    ($name:ident ; ($($dom:ty),+) ; $size:expr ; $pid:expr) => {
        #[test]
        fn $name (){
            let sinfo = std::sync::Arc::new(get_sinfo());
            let mut idsol = Vec::new();
            $(
                let sol = PartialSol::<ParSId,$dom,TestSInfo>::new_default($size,sinfo.clone());
                _test_solution_assertion($size,&sol, $pid);
                idsol.push(sol.get_id().id);
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));

            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_sol!(mixed_size_3 ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());
