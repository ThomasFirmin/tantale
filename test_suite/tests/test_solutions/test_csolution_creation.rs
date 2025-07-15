use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, Unit, TypeDom};
use tantale::core::{ComputedSol,Computed, PartialSol,Partial,Codomain,SingleCodomain};
use tantale_core::Solution;

use std::fmt::{Debug, Display};
use std::collections::HashSet;
use std::process;

use super::init_outcome::{get_struct, OutExample};
use super::init_sinfo::{get_sinfo,TestSInfo};

fn _test_solution_assertion<Dom,const N: usize>(
    sol: &ComputedSol<Dom,SingleCodomain<OutExample>, OutExample,TestSInfo,N>,
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
    assert_eq!(
        sol.get_y().value,
        1.0,
        "Wrong value from codomain."
    );
    assert_eq!(
        sol.get_info().info,
        42.0,
        "Wrong solution info from TestSInfo."
    );
    assert_eq!(sol.get_id().0, id.0, "Solution PID mismatch.");
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_sol {
    ($name:ident ; ($($dom:ty),+) ; $size:expr ; ($($id:expr),+),$pid:expr) => {
        #[test]
        fn $name (){
            let mut idsol = Vec::new();
            let sinfo = std::sync::Arc::new(get_sinfo());
            let out = get_struct();
            let codom = SingleCodomain::new(|h : &OutExample| h.obj1);
            $(
                let y = std::sync::Arc::new(codom.get_elem(&out));
                let psol = PartialSol::<$dom,TestSInfo,$size>::new_default($pid,sinfo.clone());
                let sol = ComputedSol::<_,SingleCodomain<OutExample>,_,_,$size>::new(psol,y.clone());
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
