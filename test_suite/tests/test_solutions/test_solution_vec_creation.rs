use tantale::core::domain::{Bool, Cat, Domain, Int, Nat, Real, Unit};
use tantale::core::Solution;

use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::process;

fn _test_solution_assertion<'a, D: Domain, const S: usize>(
    sol: &Vec<Solution<D, S>>, pid : u32,
) where
    D: Domain + Clone + Display + Debug,
    D::TypeDom: Sync + Send,
{
    for s in sol{
        assert_eq!(
            s.x,
            vec![D::TypeDom::default(); S].into_boxed_slice(),
            "Solution `x` mismatch."
        );
        assert_eq!(s.id.1, pid, "Solution `pid` mismatch.");
    }
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_vec {
    ($name:ident ; ($($dom:ty),+) ; $size:expr ; $pid:expr) => {
        #[test]
        fn $name (){
            let mut idsol = Vec::new();
            $(
                let v = Solution::<$dom,$size>::new_default_vec($pid);
                _test_solution_assertion(&v, $pid);
                v.iter().for_each(|x| idsol.push(x.id.0));
            )*
            let mut unique = HashSet::new();
            idsol.iter().all(|x| unique.insert(x));
            assert_eq!(idsol.len(),unique.len(), "An `id` is not unique.");
        }
    };
}

get_default_vec!(mixed_size_3 ; (Real,Nat, Int, Cat, Bool, Unit) ; 3 ; process::id());