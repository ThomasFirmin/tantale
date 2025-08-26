use serde_json;
use tantale::algos::RSState;

#[test]
fn test_rsstate_json() {
    let state = RSState {
        batch: 10,
        iteration: 3,
    };

    let st_ser = serde_json::to_string(&state).unwrap();
    let nstate: RSState = serde_json::from_str(&st_ser).unwrap();

    assert_eq!(
        state.batch, nstate.batch,
        "Serde mismatch on loaded RSState batch"
    );
    assert_eq!(
        state.iteration, nstate.iteration,
        "Serde mismatch on loaded RSState batch"
    );
}
