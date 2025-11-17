use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, Unit};
use tantale::core::{ParSId, Partial, Solution};
use tantale_core::domain::TypeDom;
use tantale_core::BasePartial;

use num::cast::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::process;

use super::init_sinfo::{get_sinfo, TestSInfo};

fn _test_solution_assertion<Dom>(n: usize, sol: &[BasePartial<ParSId, Dom, TestSInfo>], pid: u32)
where
    Dom: Domain + Clone + Display + Debug,
    TypeDom<Dom>:
        Default + Clone + Display + Debug + Serialize + for<'a> Deserialize<'a> + Send + Sync,
{
    for s in sol {
        assert_eq!(
            s.x,
            std::sync::Arc::from(vec![Dom::TypeDom::default(); n]),
            "Solution `x` mismatch."
        );
        assert_eq!(
            s.get_id().pid,
            <u32 as AsPrimitive<usize>>::as_(pid),
            "Solution `pid` mismatch."
        );
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
                let v = Partial::<ParSId,$dom,TestSInfo>::new_default_vec($size,sinfo.clone(),7);
                _test_solution_assertion($size,&v, $pid);
                v.iter().for_each(|x| idsol.push(x.id.id));
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));
            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_vec!(mixed_size_3 ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());
