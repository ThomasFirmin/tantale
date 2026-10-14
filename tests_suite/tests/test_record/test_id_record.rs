use paste::paste;
use tantale::core::utils::record::{Recordable, Schema};
use tantale::core::solution::{
    SolId,
    id::{ParSId, SId},
};

macro_rules! test_header {
    ($($name:ident, $idty:ident, $expected:expr);*) => {
        $(
            paste!{
                #[test]
                fn [<test_$name _header>](){
                    let head = $idty::columns(&(), ());
                    assert_eq!(head,$expected,"CSV Header does not match the baseline.")
                }
            }
        )*
    };
}

test_header!(
    sid, SId, Vec::from(["id"]);
    parsid, ParSId, Vec::from(["pid","id"])
);

macro_rules! test_write {
    ($($name:ident, $idty:ident, $expected:expr);*) => {
        $(
            paste!{
                #[test]
                fn [<test_$name _write>](){
                    let id = $idty::generate();
                    let expected = $expected;
                    let wrote = id.record(&(), ());
                    assert_eq!(wrote,expected(id),"Written CSV ID does not match the baseline.")
                }
            }
        )*
    };
}

test_write!(
    sid, SId, |id : SId| -> Vec<String> {Vec::from([format!("{}",id.id)])};
    parsid, ParSId, |id : ParSId| -> Vec<String> {Vec::from([format!("{}",id.pid),format!("{}",id.id)])}
);
