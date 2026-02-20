use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, TypeDom, Unit};
use tantale_core::solution::{HasId, ParSId, Uncomputed};
use tantale_core::{BaseSol, FidelitySol};

use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::process;
use std::sync::Arc;

use super::init_sinfo::{TestSInfo, get_sinfo};

fn _test_solution_assertion<Unc, Dom>(n: usize, sol: &Unc, pid: u32)
where
    Unc: Uncomputed<ParSId, Dom, TestSInfo, Raw = Arc<[Dom::TypeDom]>>,
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>:
        Default + Clone + Display + Debug + Serialize + for<'a> Deserialize<'a> + Send + Sync,
{
    assert_eq!(
        sol.get_x(),
        std::sync::Arc::from(vec![Dom::TypeDom::default(); n]),
        "Solution `x` mismatch."
    );
    assert_eq!(
        sol.get_id().pid,
        <u32 as AsPrimitive<usize>>::as_(pid),
        "Solution PID mismatch."
    );
    assert_eq!(
        sol.get_sinfo().info,
        42.0,
        "Wrong solution info from TestSInfo."
    );
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_sol {
    ($name:ident ; $sol:ident ; ($($dom:ty),+) ; $size:expr ; $pid:expr) => {
        #[test]
        fn $name (){
            let sinfo = std::sync::Arc::new(get_sinfo());
            let mut idsol = Vec::new();
            $(
                let sol = $sol::<ParSId,$dom,TestSInfo>::default(sinfo.clone(),$size);
                _test_solution_assertion($size,&sol, $pid);
                idsol.push(sol.get_id().id);
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));

            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_sol!(mixed_size_3 ; BaseSol ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());
get_default_sol!(fid_mixed_size_3 ; FidelitySol ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());
