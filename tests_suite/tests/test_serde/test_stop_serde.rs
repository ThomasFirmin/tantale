use rmp_serde;
use tantale::core::stop::{Calls, ExpStep};
use tantale_core::{objective::Step, Stop};

#[test]
fn test_calls_json() {
    let mut calls = Calls::new(10);
    calls.update(ExpStep::Distribution(Step::Evaluated));
    let st_ser = rmp_serde::encode::to_vec(&calls).unwrap();
    let mut ncalls: Calls = rmp_serde::decode::from_slice(&st_ser).unwrap();

    assert_eq!(calls.0, ncalls.0, "Serde mismatch on loaded Calls #calls");
    assert_eq!(calls.0, 1, "Serde on Calls wrong #calls");
    assert_eq!(ncalls.0, 1, "Serde on loaded Calls wrong #calls");
    assert_eq!(
        calls.1, ncalls.1,
        "Serde mismatch on loaded Calls threshold"
    );
    ncalls.update(ExpStep::Distribution(Step::Evaluated));
    assert_eq!(
        ncalls.0, 2,
        "Serde on Calls wrong #calls after loading and updating"
    );
}
