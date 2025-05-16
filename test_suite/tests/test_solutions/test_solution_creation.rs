use tantale::core::Solution;
use tantale::core::domain::{Domain, Real,Nat,Int,Bool,Cat,Unit};

use std::fmt::{Display,Debug};
use std::process;

fn _test_solution_assertion<'a, D:Domain, const S:usize>(sol: &'a Solution<D,S>, id : (usize,u32))
{
    let def = [D::TypeDom::default();S];
    let mstr:Vec<_> = def.iter().map(|s| s.to_string()).collect();
    let defstr = mstr.join(",");
    println!("DEFAULT {}",defstr);
    assert_eq!(sol.x, [D::TypeDom::default();S], "Solution `x` mismatch.");
    assert_eq!(sol.id.0, id.0, "Solution `id` mismatch.");
    assert_eq!(sol.id.1, id.1, "Solution `pid` mismatch.");
}

// BOTH DOMAINS ARE DEFINED
macro_rules! get_default_sol {
    ($name:ident ; ($($dom:ty),+) ; $size:expr ; ($($id:expr),+),$pid:expr) => {
        #[test]
        fn $name (){
            $(
                let sol = Solution::<$dom,$size>::default($pid);
                _test_solution_assertion(&sol, ($id, $pid));
            )*
        }
    };
}

get_default_sol!(mixed_size_3 ; (Real,Nat, Int, Cat) ; 3 ; (0,1,2,3),process::id());