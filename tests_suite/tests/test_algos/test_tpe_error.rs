use std::error::Error;

use tantale::algos::bayesian::error::SplitError;

#[test]
fn test_split_error_display() {
    let err = SplitError("Beta must be between 0 and 1".into());
    let msg = format!("{}", err);
    assert!(
        msg.contains("Split error"),
        "Display should contain 'Split error', got: {}",
        msg
    );
    assert!(
        msg.contains("Beta must be between 0 and 1"),
        "Display should contain the inner message, got: {}",
        msg
    );
}

#[test]
fn test_split_error_debug() {
    let err = SplitError("some message".into());
    let dbg = format!("{:?}", err);
    assert!(
        dbg.contains("Split error"),
        "Debug should contain 'Split error', got: {}",
        dbg
    );
}

#[test]
fn test_split_error_is_error() {
    let err = SplitError("test".into());
    // SplitError implements std::error::Error
    let _: &dyn Error = &err;
}
