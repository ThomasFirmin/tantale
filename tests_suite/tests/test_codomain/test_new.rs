use tantale::core::{
    ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain, ElemCostConstCodomain,
    ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain, ElemSingleCodomain,
    ElemSpikeConstCodomain, ElemSpikeConstMultiCodomain, ElemSpikeCostCodomain, ElemSpikeCostConstCodomain,
    ElemSpikeCostConstMultiCodomain, ElemSpikeCostMultiCodomain, ElemSpikeMultiCodomain, ElemSpikeCodomain,
};

#[test]
fn new_elemsinglecodomain() {
    let elem = ElemSingleCodomain::new(1.1);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemSingleCodomain::new.");
}

#[test]
fn new_elemcostcodomain() {
    let elem = ElemCostCodomain::new(1.1, 2.2);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemCostCodomain::new.");
    assert_eq!(elem.cost, 2.2, "Wrong cost for ElemCostCodomain::new.");
}

#[test]
fn new_elemconstcodomain() {
    let elem = ElemConstCodomain::new(1.1, vec![3.3, 4.4]);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemConstCodomain::new.");
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemConstCodomain::new."
    );
}

#[test]
fn new_elemcostconstcodomain() {
    let elem = ElemCostConstCodomain::new(1.1, 2.2, vec![3.3, 4.4]);

    assert_eq!(
        elem.value, 1.1,
        "Wrong value for ElemCostConstCodomain::new."
    );
    assert_eq!(elem.cost, 2.2, "Wrong cost for ElemCostConstCodomain::new.");
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemCostConstCodomain::new."
    );
}

#[test]
fn new_elemmulticodomain() {
    let elem = ElemMultiCodomain::new(vec![1.1, 2.2]);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemMultiCodomain::new."
    );
}

#[test]
fn new_elemcostmulticodomain() {
    let elem = ElemCostMultiCodomain::new(vec![1.1, 2.2], 3.3);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemCostMultiCodomain::new."
    );
    assert_eq!(elem.cost, 3.3, "Wrong cost for ElemCostMultiCodomain::new.");
}

#[test]
fn new_elemconstmulticodomain() {
    let elem = ElemConstMultiCodomain::new(vec![1.1, 2.2], vec![3.3, 4.4]);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemConstMultiCodomain::new."
    );
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemConstMultiCodomain::new."
    );
}

#[test]
fn new_elemcostconstmulticodomain() {
    let elem = ElemCostConstMultiCodomain::new(vec![1.1, 2.2], 3.3, vec![4.4, 5.5]);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemCostConstMultiCodomain::new."
    );
    assert_eq!(
        elem.cost, 3.3,
        "Wrong cost for ElemCostConstMultiCodomain::new."
    );
    assert_eq!(
        elem.constraints.as_ref(),
        [4.4, 5.5].as_ref(),
        "Wrong constraints for ElemCostConstMultiCodomain::new."
    );
}

#[test]
fn new_elemspikecodomain() {
    let elem = ElemSpikeCodomain::new(1.1, 10, 3);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemSpikeCodomain::new.");
    assert_eq!(elem.samples, 10, "Wrong sampling for ElemSpikeCodomain::new.");
    assert_eq!(elem.spiking, 3, "Wrong spiking for ElemSpikeCodomain::new.");
}

#[test]
fn new_elemspikecostcodomain() {
    let elem = ElemSpikeCostCodomain::new(1.1, 2.2, 10, 3);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemSpikeCostCodomain::new.");
    assert_eq!(elem.cost, 2.2, "Wrong cost for ElemSpikeCostCodomain::new.");
    assert_eq!(elem.samples, 10, "Wrong sampling for ElemSpikeCodomain::new.");
    assert_eq!(elem.spiking, 3, "Wrong spiking for ElemSpikeCodomain::new.");
}

#[test]
fn new_elemspikeconstcodomain() {
    let elem = ElemSpikeConstCodomain::new(1.1, vec![3.3, 4.4], 10, 3);

    assert_eq!(elem.value, 1.1, "Wrong value for ElemSpikeConstCodomain::new.");
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemSpikeConstCodomain::new."
    );
    assert_eq!(elem.samples, 10, "Wrong sampling for ElemSpikeCodomain::new.");
    assert_eq!(elem.spiking, 3, "Wrong spiking for ElemSpikeCodomain::new.");
}

#[test]
fn new_elemspikecostconstcodomain() {
    let elem = ElemSpikeCostConstCodomain::new(1.1, 2.2, vec![3.3, 4.4], 10, 3);

    assert_eq!(
        elem.value, 1.1,
        "Wrong value for ElemSpikeCostConstCodomain::new."
    );
    assert_eq!(elem.cost, 2.2, "Wrong cost for ElemSpikeCostConstCodomain::new.");
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemSpikeCostConstCodomain::new."
    );
    assert_eq!(elem.samples, 10, "Wrong sampling for ElemSpikeCodomain::new.");
    assert_eq!(elem.spiking, 3, "Wrong spiking for ElemSpikeCodomain::new.");
}

#[test]
fn new_elemspikemulticodomain() {
    let elem = ElemSpikeMultiCodomain::new(vec![1.1, 2.2], 10, 3);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemSpikeMultiCodomain::new."
    );
    assert_eq!(elem.samples, 10, "Wrong sampling for ElemSpikeCodomain::new.");
    assert_eq!(elem.spiking, 3, "Wrong spiking for ElemSpikeCodomain::new.");
}

#[test]
fn new_elemspikecostmulticodomain() {
    let elem = ElemSpikeCostMultiCodomain::new(vec![1.1, 2.2], 3.3, 10, 3);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemSpikeCostMultiCodomain::new."
    );
    assert_eq!(elem.cost, 3.3, "Wrong cost for ElemSpikeCostMultiCodomain::new.");
    assert_eq!(elem.samples, 10, "Wrong sampling for ElemSpikeCodomain::new.");
    assert_eq!(elem.spiking, 3, "Wrong spiking for ElemSpikeCodomain::new.");
}

#[test]
fn new_elemspikeconstmulticodomain() {
    let elem = ElemSpikeConstMultiCodomain::new(vec![1.1, 2.2], vec![3.3, 4.4], 10, 3);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemSpikeConstMultiCodomain::new."
    );
    assert_eq!(
        elem.constraints.as_ref(),
        [3.3, 4.4].as_ref(),
        "Wrong constraints for ElemSpikeConstMultiCodomain::new."
    );
    assert_eq!(elem.samples, 10, "Wrong sampling for ElemSpikeCodomain::new.");
    assert_eq!(elem.spiking, 3, "Wrong spiking for ElemSpikeCodomain::new.");
}

#[test]
fn new_elemspikecostconstmulticodomain() {
    let elem = ElemSpikeCostConstMultiCodomain::new(vec![1.1, 2.2], 3.3, vec![4.4, 5.5], 10, 3);

    assert_eq!(
        elem.value.as_ref(),
        [1.1, 2.2].as_ref(),
        "Wrong values for ElemSpikeCostConstMultiCodomain::new."
    );
    assert_eq!(
        elem.cost, 3.3,
        "Wrong cost for ElemSpikeCostConstMultiCodomain::new."
    );
    assert_eq!(
        elem.constraints.as_ref(),
        [4.4, 5.5].as_ref(),
        "Wrong constraints for ElemSpikeCostConstMultiCodomain::new."
    );
    assert_eq!(elem.samples, 10, "Wrong sampling for ElemSpikeCodomain::new.");
    assert_eq!(elem.spiking, 3, "Wrong spiking for ElemSpikeCodomain::new.");
}
