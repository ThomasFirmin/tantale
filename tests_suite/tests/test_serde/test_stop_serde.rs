use serde_json;
use tantale::core::stop::{Calls, ExpStep};
use tantale_core::Stop;

#[test]
fn test_calls_json() {
    let mut calls = Calls::new(10);
    calls.update(ExpStep::Distribution);
    let st_ser = serde_json::to_string(&calls).unwrap();
    let mut ncalls: Calls = serde_json::from_str(&st_ser).unwrap();

    assert_eq!(calls.0, ncalls.0, "Serde mismatch on loaded Calls #calls");
    assert_eq!(calls.0, 1, "Serde on Calls wrong #calls");
    assert_eq!(ncalls.0, 1, "Serde on loaded Calls wrong #calls");
    assert_eq!(
        calls.1, ncalls.1,
        "Serde mismatch on loaded Calls threshold"
    );
    ncalls.update(ExpStep::Distribution);
    assert_eq!(
        ncalls.0, 2,
        "Serde on Calls wrong #calls after loading and updating"
    );
}
