use rmp_serde;
use tantale::algos::BatchRandomSearch;
use tantale::core::{EmptyInfo, Mixed, NoDomain, SId, Sp};
use tantale::core::problem::Pb;
use super::init_sp::one_step_func::{FuncType};

#[test]
#[allow(clippy::type_complexity)]
fn test_rsstate_json() {
    let state = BatchRandomSearch::<Pb<SId, EmptyInfo, Sp<Mixed, NoDomain>, FuncType<Mixed>>>::new(10);

    let st_ser = rmp_serde::encode::to_vec(&state).unwrap();
    let nstate: BatchRandomSearch::<Pb<SId, EmptyInfo, Sp<Mixed, NoDomain>, FuncType<Mixed>>> = rmp_serde::decode::from_slice(&st_ser).unwrap();

    assert_eq!(
        state.batch_size, nstate.batch_size,
        "Serde mismatch on loaded RSState batch"
    );
    assert_eq!(
        state.calls, nstate.calls,
        "Serde mismatch on loaded RSState batch"
    );
}
