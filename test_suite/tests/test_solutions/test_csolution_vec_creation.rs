use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, Unit};
use tantale::core::{Computed, ComputedSol, Partial, PartialSol, Solution};
use tantale_core::domain::TypeDom;
use tantale_core::{Codomain, SingleCodomain};

use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::process;

use super::init_outcome::{get_struct, OutExample};
use super::init_sinfo::{get_sinfo, TestSInfo};

fn _test_solution_assertion<Dom>(
    n: usize,
    sol: &[ComputedSol<Dom, SingleCodomain<OutExample>, OutExample, TestSInfo>],
    pid: u32,
) where
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>: Sync + Send,
{
    for s in sol {
        assert_eq!(
            s.get_sol().x,
            std::sync::Arc::from(vec![Dom::TypeDom::default(); n]),
            "Solution `x` mismatch."
        );
        assert_eq!(
            s.get_info().info,
            42.0,
            "Wrong solution info from TestSInfo."
        );
        assert_eq!(s.get_id().0, pid, "Solution PID mismatch.");
    }
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_vec {
    ($name:ident ; ($($dom:ty),+) ; $size:expr ; $pid:expr) => {
        #[test]
        fn $name (){
            let mut idsol = Vec::new();
            let sinfo = std::sync::Arc::new(get_sinfo());
            let out = get_struct();
            let codom = SingleCodomain::new(|h : &OutExample| h.obj1);
            $(
                let y = std::sync::Arc::new(codom.get_elem(&out));
                let psol = PartialSol::<$dom,TestSInfo>::new_default_vec($size,$pid,sinfo.clone(),7);
                let vec_y = vec![y;7];
                let v = ComputedSol::<_,SingleCodomain<OutExample>,_,_>::new_vec(psol,vec_y);
                _test_solution_assertion($size,&v, $pid);
                v.iter().for_each(|x| idsol.push(x.get_id().1));
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));
            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_vec!(mixed_size_3 ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());
