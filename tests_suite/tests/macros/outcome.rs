use serde::{Deserialize, Serialize};
use tantale::core::objective::EvalStep;
use tantale::macros::Outcome;

#[test]
fn mixed_derive() {
    use tantale::core::CostConstMultiCodomain;
    use tantale::core::Codomain;

    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct OutExample {
        pub cost2: f64,
        pub con3: f64,
        pub con4: f64,
        pub con5: f64,
        pub mul6: f64,
        pub mul7: f64,
        pub mul8: f64,
        pub mul9: f64,
        pub fid10: EvalStep,
    }

    pub fn get_struct() -> OutExample {
        OutExample {
            cost2: 2.0,
            con3: 3.0,
            con4: 4.0,
            con5: 5.0,
            mul6: 6.0,
            mul7: 7.0,
            mul8: 8.0,
            mul9: 9.0,
            fid10: EvalStep::partially(0),
        }
    }

    let codom = CostConstMultiCodomain::new(
        // Define multi-objective
        vec![
            |h: &OutExample| h.mul6,
            |h: &OutExample| h.mul7,
            |h: &OutExample| h.mul8,
            |h: &OutExample| h.mul9,
        ]
        .into_boxed_slice(),
        // Define fidelity
        |h: &OutExample| h.cost2,
        // Define constraints
        vec![
            |h: &OutExample| h.con3,
            |h: &OutExample| h.con4,
            |h: &OutExample| h.con5,
        ]
        .into_boxed_slice(),
    );
    let out = get_struct();
    let extracted = codom.get_elem(&out);

    assert_eq!(
        extracted.value[0], 6.0,
        "Wrong extraction of mul6 in derive Outcome."
    );
    assert_eq!(
        extracted.value[1], 7.0,
        "Wrong extraction of mul7 in derive Outcome."
    );
    assert_eq!(
        extracted.value[2], 8.0,
        "Wrong extraction of mul8 in derive Outcome."
    );
    assert_eq!(
        extracted.value[3], 9.0,
        "Wrong extraction of mul9 in derive Outcome."
    );

    assert_eq!(
        extracted.constraints[0], 3.0,
        "Wrong extraction of con3 in derive Outcome."
    );
    assert_eq!(
        extracted.constraints[1], 4.0,
        "Wrong extraction of con4 in derive Outcome."
    );
    assert_eq!(
        extracted.constraints[2], 5.0,
        "Wrong extraction of con5 in derive Outcome."
    );

    assert_eq!(
        extracted.cost, 2.0,
        "Wrong extraction of cost2 in derive Outcome."
    );
}
