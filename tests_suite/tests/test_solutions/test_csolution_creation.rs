use tantale::core::Solution;
use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, TypeDom, Unit};
use tantale::core::{
    BaseSol, Codomain, Computed, FidelitySol, ParSId, SingleCodomain,
    solution::{HasId, HasSolInfo, HasY, Uncomputed},
};

use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::process;
use std::sync::Arc;

use super::init_outcome::{OutExample, get_struct};
use super::init_sinfo::{TestSInfo, get_sinfo};

type TestComp<Sol, Dom> =
    Computed<Sol, ParSId, Dom, SingleCodomain<OutExample>, OutExample, TestSInfo>;

fn _test_solution_assertion<Unc, Dom>(n: usize, sol: &TestComp<Unc, Dom>, pid: u32)
where
    Unc: Uncomputed<ParSId, Dom, TestSInfo, Raw = Arc<[Dom::TypeDom]>>,
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>: Sync + Send,
    TypeDom<Dom>: Default + Clone + Display + Debug + Serialize + for<'a> Deserialize<'a>,
{
    assert_eq!(
        sol.clone_x(),
        std::sync::Arc::from(vec![Dom::TypeDom::default(); n]),
        "Solution `x` mismatch."
    );
    assert_eq!(sol.y().value, 1.0, "Wrong value from codomain.");
    assert_eq!(
        sol.sinfo().info,
        42.0,
        "Wrong solution info from TestSInfo."
    );
    assert_eq!(
        sol.id().pid,
        <u32 as AsPrimitive<usize>>::as_(pid),
        "Solution PID mismatch."
    );
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_sol {
    ($name:ident ; $sol:ident ; ($($dom:ty),+) ; $size:expr ; $pid:expr) => {
        #[test]
        fn $name (){
            let mut idsol = Vec::new();
            let sinfo = std::sync::Arc::new(get_sinfo());
            let out = get_struct();
            let codom = SingleCodomain::new(|h : &OutExample| h.obj1);
            $(
                let y = std::sync::Arc::new(codom.get_elem(&out));
                let psol = $sol::<ParSId,$dom,TestSInfo>::default(sinfo.clone(),$size);
                let sol = Computed::new(psol,y.clone());
                _test_solution_assertion($size,&sol, $pid);
                idsol.push(sol.id().id);
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));

            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_sol!(mixed_size_3 ; BaseSol ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());
get_default_sol!(fid_mixed_size_3 ; FidelitySol ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());
