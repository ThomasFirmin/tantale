use tantale::core::{ElemSpikeCostCodomain, Outcome, spikes::{ElemSpikeCodomain, ElemSpikeConstCodomain, ElemSpikeConstMultiCodomain, ElemSpikeCostConstCodomain, ElemSpikeCostConstMultiCodomain, ElemSpikeCostMultiCodomain, ElemSpikeMultiCodomain}};
pub use tantale::core::{
    domain::codomain::{
        ConstCodomain, ConstMultiCodomain, CostCodomain, CostConstCodomain, CostConstMultiCodomain,
        CostMultiCodomain, ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain,
        ElemCostConstCodomain, ElemCostConstMultiCodomain, ElemCostMultiCodomain,
        ElemMultiCodomain, ElemSingleCodomain, FidCriteria, MultiCodomain, SingleCodomain,
    },
    objective::Step,
};
use tantale::macros::Outcome;

use serde::{Deserialize, Serialize};

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCodSingle {
    #[maximize]
    pub obj1: f64,
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    pub samples: usize,
    pub spiking: usize,
}

pub fn get_elemsingle() -> (<OutCodSingle as Outcome>::Cod, ElemSingleCodomain) {
    (OutCodSingle::codomain(), ElemSingleCodomain { value: 1.1 })
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCodCost {
    #[maximize]
    pub obj1: f64,
    #[cost]
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    pub samples: usize,
    pub spiking: usize,
}
pub fn get_elemcost() -> (<OutCodCost as Outcome>::Cod, ElemCostCodomain) {
    (
        OutCodCost::codomain(),
        ElemCostCodomain {
            value: 1.1,
            cost: 2.2,
        },
    )
}
#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCodConst {
    #[maximize]
    pub obj1: f64,
    pub cost2: f64,
    #[constraint]
    pub con3: f64,
    #[constraint]
    pub con4: f64,
    #[constraint]
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    pub samples: usize,
    pub spiking: usize,
}
pub fn get_elemconst() -> (<OutCodConst as Outcome>::Cod, ElemConstCodomain) {
    (
        OutCodConst::codomain(),
        ElemConstCodomain {
            value: 1.1,
            constraints: Box::from([2.2, 3.3, 4.4]),
        },
    )
}
#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCodCostConst {
    #[maximize]
    pub obj1: f64,
    #[cost]
    pub cost2: f64,
    #[constraint]
    pub con3: f64,
    #[constraint]
    pub con4: f64,
    #[constraint]
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    pub samples: usize,
    pub spiking: usize,
}
pub fn get_elemcostconst() -> (<OutCodCostConst as Outcome>::Cod, ElemCostConstCodomain) {
    (
        OutCodCostConst::codomain(),
        ElemCostConstCodomain {
            value: 1.1,
            cost: 2.2,
            constraints: Box::from([3.3, 4.4]),
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCodMulti {
    pub obj1: f64,
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    #[maximize]
    pub mul6: f64,
    #[maximize]
    pub mul7: f64,
    #[maximize]
    pub mul8: f64,
    #[maximize]
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    pub samples: usize,
    pub spiking: usize,
}
pub fn get_elemmulti() -> (<OutCodMulti as Outcome>::Cod, ElemMultiCodomain) {
    (
        OutCodMulti::codomain(),
        ElemMultiCodomain {
            value: Box::from([1.1, 2.2, 3.3, 4.4]),
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCodCostMulti {
    pub obj1: f64,
    #[cost]
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    #[maximize]
    pub mul6: f64,
    #[maximize]
    pub mul7: f64,
    #[maximize]
    pub mul8: f64,
    #[maximize]
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    pub samples: usize,
    pub spiking: usize,
}
pub fn get_elemcostmulti() -> (<OutCodCostMulti as Outcome>::Cod, ElemCostMultiCodomain) {
    (
        OutCodCostMulti::codomain(),
        ElemCostMultiCodomain {
            value: Box::from([1.1, 2.2, 3.3, 4.4]),
            cost: 3.3,
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCodConstMulti {
    pub obj1: f64,
    pub cost2: f64,
    #[constraint]
    pub con3: f64,
    #[constraint]
    pub con4: f64,
    #[constraint]
    pub con5: f64,
    #[maximize]
    pub mul6: f64,
    #[maximize]
    pub mul7: f64,
    #[maximize]
    pub mul8: f64,
    #[maximize]
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    pub samples: usize,
    pub spiking: usize,
}
pub fn get_elemconstmulti() -> (<OutCodConstMulti as Outcome>::Cod, ElemConstMultiCodomain) {
    (
        OutCodConstMulti::codomain(),
        ElemConstMultiCodomain {
            value: Box::from([1.1, 2.2, 3.3, 4.4]),
            constraints: Box::from([3.3, 4.4]),
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCodCostConstMulti {
    pub obj1: f64,
    #[cost]
    pub cost2: f64,
    #[constraint]
    pub con3: f64,
    #[constraint]
    pub con4: f64,
    #[constraint]
    pub con5: f64,
    #[maximize]
    pub mul6: f64,
    #[maximize]
    pub mul7: f64,
    #[maximize]
    pub mul8: f64,
    #[maximize]
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    pub samples: usize,
    pub spiking: usize,
}
pub fn get_elemcostconstmulti() -> (
    <OutCodCostConstMulti as Outcome>::Cod,
    ElemCostConstMultiCodomain,
) {
    (
        OutCodCostConstMulti::codomain(),
        ElemCostConstMultiCodomain {
            value: Box::from([1.1, 2.2, 3.3, 4.4]),
            cost: 3.3,
            constraints: Box::from([4.4, 5.5]),
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutSpikeCodSingle {
    #[maximize]
    pub obj1: f64,
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    #[samples]
    pub samples: usize,
    #[spiking]
    pub spiking: usize,
}

pub fn get_elemspike() -> (<OutSpikeCodSingle as Outcome>::Cod, ElemSpikeCodomain) {
    (OutSpikeCodSingle::codomain(), ElemSpikeCodomain { value: 1.1 , samples: 10, spiking: 3})
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutSpikeCodCost {
    #[maximize]
    pub obj1: f64,
    #[cost]
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    #[samples]
    pub samples: usize,
    #[spiking]
    pub spiking: usize,
}
pub fn get_elemspikecost() -> (<OutSpikeCodCost as Outcome>::Cod, ElemSpikeCostCodomain) {
    (
        OutSpikeCodCost::codomain(),
        ElemSpikeCostCodomain {
            value: 1.1,
            cost: 2.2,
            samples: 10,
            spiking: 3,
        },
    )
}
#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutSpikeCodConst {
    #[maximize]
    pub obj1: f64,
    pub cost2: f64,
    #[constraint]
    pub con3: f64,
    #[constraint]
    pub con4: f64,
    #[constraint]
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    #[samples]
    pub samples: usize,
    #[spiking]
    pub spiking: usize,
}
pub fn get_elemspikeconst() -> (<OutSpikeCodConst as Outcome>::Cod, ElemSpikeConstCodomain) {
    (
        OutSpikeCodConst::codomain(),
        ElemSpikeConstCodomain {
            value: 1.1,
            constraints: Box::from([2.2, 3.3, 4.4]),
            samples: 10,
            spiking: 3,
        },
    )
}
#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutSpikeCodCostConst {
    #[maximize]
    pub obj1: f64,
    #[cost]
    pub cost2: f64,
    #[constraint]
    pub con3: f64,
    #[constraint]
    pub con4: f64,
    #[constraint]
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    #[samples]
    pub samples: usize,
    #[spiking]
    pub spiking: usize,
}
pub fn get_elemspikecostconst() -> (<OutSpikeCodCostConst as Outcome>::Cod, ElemSpikeCostConstCodomain) {
    (
        OutSpikeCodCostConst::codomain(),
        ElemSpikeCostConstCodomain {
            value: 1.1,
            cost: 2.2,
            constraints: Box::from([3.3, 4.4]),
            samples: 10,
            spiking: 3,
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutSpikeCodMulti {
    pub obj1: f64,
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    #[maximize]
    pub mul6: f64,
    #[maximize]
    pub mul7: f64,
    #[maximize]
    pub mul8: f64,
    #[maximize]
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    #[samples]
    pub samples: usize,
    #[spiking]
    pub spiking: usize,
}
pub fn get_elemspikemulti() -> (<OutSpikeCodMulti as Outcome>::Cod, ElemSpikeMultiCodomain) {
    (
        OutSpikeCodMulti::codomain(),
        ElemSpikeMultiCodomain {
            value: Box::from([1.1, 2.2, 3.3, 4.4]),
            samples: 10,
            spiking: 3,
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutSpikeCodCostMulti {
    pub obj1: f64,
    #[cost]
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    #[maximize]
    pub mul6: f64,
    #[maximize]
    pub mul7: f64,
    #[maximize]
    pub mul8: f64,
    #[maximize]
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    #[samples]
    pub samples: usize,
    #[spiking]
    pub spiking: usize,
}
pub fn get_elemspikecostmulti() -> (<OutSpikeCodCostMulti as Outcome>::Cod, ElemSpikeCostMultiCodomain) {
    (
        OutSpikeCodCostMulti::codomain(),
        ElemSpikeCostMultiCodomain {
            value: Box::from([1.1, 2.2, 3.3, 4.4]),
            cost: 3.3,
            samples: 10,
            spiking: 3,
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutSpikeCodConstMulti {
    pub obj1: f64,
    pub cost2: f64,
    #[constraint]
    pub con3: f64,
    #[constraint]
    pub con4: f64,
    #[constraint]
    pub con5: f64,
    #[maximize]
    pub mul6: f64,
    #[maximize]
    pub mul7: f64,
    #[maximize]
    pub mul8: f64,
    #[maximize]
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    #[samples]
    pub samples: usize,
    #[spiking]
    pub spiking: usize,
}
pub fn get_elemspikeconstmulti() -> (<OutSpikeCodConstMulti as Outcome>::Cod, ElemSpikeConstMultiCodomain) {
    (
        OutSpikeCodConstMulti::codomain(),
        ElemSpikeConstMultiCodomain {
            value: Box::from([1.1, 2.2, 3.3, 4.4]),
            constraints: Box::from([3.3, 4.4]),
            samples: 10,
            spiking: 3,
        },
    )
}

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutSpikeCodCostConstMulti {
    pub obj1: f64,
    #[cost]
    pub cost2: f64,
    #[constraint]
    pub con3: f64,
    #[constraint]
    pub con4: f64,
    #[constraint]
    pub con5: f64,
    #[maximize]
    pub mul6: f64,
    #[maximize]
    pub mul7: f64,
    #[maximize]
    pub mul8: f64,
    #[maximize]
    pub mul9: f64,
    pub fid10: Step,
    pub more: f64,
    pub info: f64,
    #[samples]
    pub samples: usize,
    #[spiking]
    pub spiking: usize,
}
pub fn get_elemspikecostconstmulti() -> (
    <OutSpikeCodCostConstMulti as Outcome>::Cod,
    ElemSpikeCostConstMultiCodomain,
) {
    (
        OutSpikeCodCostConstMulti::codomain(),
        ElemSpikeCostConstMultiCodomain {
            value: Box::from([1.1, 2.2, 3.3, 4.4]),
            cost: 3.3,
            constraints: Box::from([4.4, 5.5]),
            samples: 10,
            spiking: 3,
        },
    )
}
