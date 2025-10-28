use rmp_serde;
use tantale::algos::RSState;

#[test]
fn test_rsstate_json() {
    let state = RSState {
        batch: 10,
        iteration: 3,
    };

    let st_ser = rmp_serde::encode::to_vec(&state).unwrap();
    let nstate: RSState = rmp_serde::decode::from_slice(&st_ser).unwrap();

    assert_eq!(
        state.batch, nstate.batch,
        "Serde mismatch on loaded RSState batch"
    );
    assert_eq!(
        state.iteration, nstate.iteration,
        "Serde mismatch on loaded RSState batch"
    );
}
