use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, Unit};
use tantale::core::{Solution,Partial,PartialSol};
use tantale_core::domain::TypeDom;

use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::process;

use super::init_sinfo::{TestSInfo,get_sinfo};

fn _test_solution_assertion<Dom, const N: usize>(
    sol: &[PartialSol<Dom,TestSInfo,N>],
    pid: u32,
) where
    Dom : Domain + Clone + Display + Debug,
    TypeDom<Dom> : Sync + Send,
{
    for s in sol{
        assert_eq!(
            s.x,
            std::sync::Arc::from(vec![Dom::TypeDom::default(); N]),
            "Solution `x` mismatch."
        );
        assert_eq!(s.get_id().0, pid, "Solution `pid` mismatch.");
        assert_eq!(
            s.get_info().info,
            42.0,
            "Wrong solution info from TestSInfo."
        );
    }
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_vec {
    ($name:ident ; ($($dom:ty),+) ; $size:expr ; $pid:expr) => {
        #[test]
        fn $name (){
            let sinfo = std::sync::Arc::new(get_sinfo());
            let mut idsol = Vec::new();
            $(
                let v = PartialSol::<$dom,TestSInfo,$size>::new_default_vec($pid,sinfo.clone(),7);
                _test_solution_assertion(&v, $pid);
                v.iter().for_each(|x| idsol.push(x.id.1));
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));
            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_vec!(mixed_size_3 ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());