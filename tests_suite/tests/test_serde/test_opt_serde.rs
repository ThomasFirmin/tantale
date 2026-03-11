use rmp_serde;
use tantale::algos::BatchRSState;

#[test]
fn test_rsstate_json() {
    let state = BatchRSState::new(10, 3);

    let st_ser = rmp_serde::encode::to_vec(&state).unwrap();
    let nstate: BatchRSState = rmp_serde::decode::from_slice(&st_ser).unwrap();

    assert_eq!(
        state.batch, nstate.batch,
        "Serde mismatch on loaded RSState batch"
    );
    assert_eq!(
        state.iteration, nstate.iteration,
        "Serde mismatch on loaded RSState batch"
    );
}
