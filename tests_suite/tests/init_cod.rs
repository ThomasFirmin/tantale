use serde::{Deserialize, Serialize};
pub use tantale::core::objective::codomain::{
    ConstCodomain, ConstMultiCodomain, CostCodomain, CostConstCodomain, CostConstMultiCodomain,
    CostMultiCodomain, ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain,
    ElemCostConstCodomain, ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemFidCodomain,
    ElemFidConstCodomain, ElemFidConstMultiCodomain, ElemFidCostCodomain, ElemFidCostConstCodomain,
    ElemFidCostConstMultiCodomain, ElemFidCostMultiCodomain, ElemFidMultiCodomain,
    ElemMultiCodomain, ElemSingleCodomain, FidCodomain, FidConstCodomain, FidConstMultiCodomain,
    FidCostCodomain, FidCostConstCodomain, FidCostConstMultiCodomain, FidCostMultiCodomain,
    FidCriteria, FidMultiCodomain, MultiCodomain, SingleCodomain,
};
use tantale_core::objective::codomain::EvalState;
use tantale_macros::Outcome;

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCod {
    pub obj1: f64,
    pub cost2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub fid10: EvalState,
    pub more: f64,
    pub info: f64,
}

pub fn get_elemsingle() -> (SingleCodomain<OutCod>, ElemSingleCodomain) {
    (
        SingleCodomain::new(|a| a.obj1),
        ElemSingleCodomain { value: 1.1 },
    )
}
pub fn get_elemcost() -> (CostCodomain<OutCod>, ElemCostCodomain) {
    (
        CostCodomain::new(|a| a.obj1, |a| a.cost2),
        ElemCostCodomain {
            value: 1.1,
            cost: 2.2,
        },
    )
}
pub fn get_elemconst() -> (ConstCodomain<OutCod>, ElemConstCodomain) {
    (
        ConstCodomain::new(
            |a| a.obj1,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemConstCodomain {
            value: 1.1,
            constraints: Box::from([2.2, 3.3]),
        },
    )
}
pub fn get_elemcostconst() -> (CostConstCodomain<OutCod>, ElemCostConstCodomain) {
    (
        CostConstCodomain::new(
            |a| a.obj1,
            |a| a.cost2,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemCostConstCodomain {
            value: 1.1,
            cost: 2.2,
            constraints: Box::from([3.3, 4.4]),
        },
    )
}
pub fn get_elemmulti() -> (MultiCodomain<OutCod>, ElemMultiCodomain) {
    (
        MultiCodomain::new(vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice()),
        ElemMultiCodomain {
            value: Box::from([1.1, 2.2]),
        },
    )
}
pub fn get_elemcostmulti() -> (CostMultiCodomain<OutCod>, ElemCostMultiCodomain) {
    (
        CostMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            |a| a.cost2,
        ),
        ElemCostMultiCodomain {
            value: Box::from([1.1, 2.2]),
            cost: 3.3,
        },
    )
}
pub fn get_elemconstmulti() -> (ConstMultiCodomain<OutCod>, ElemConstMultiCodomain) {
    (
        ConstMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemConstMultiCodomain {
            value: Box::from([1.1, 2.2]),
            constraints: Box::from([3.3, 4.4]),
        },
    )
}
pub fn get_elemcostconstmulti() -> (CostConstMultiCodomain<OutCod>, ElemCostConstMultiCodomain) {
    (
        CostConstMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            |a| a.cost2,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemCostConstMultiCodomain {
            value: Box::from([1.1, 2.2]),
            cost: 3.3,
            constraints: Box::from([4.4, 5.5]),
        },
    )
}

pub fn get_elemfid() -> (FidCodomain<OutCod>, ElemFidCodomain) {
    (
        FidCodomain::new(|a| a.obj1, |a| a.fid10),
        ElemFidCodomain {
            value: 1.1,
            fidelity: EvalState::Completed,
        },
    )
}
pub fn get_elemfidcost() -> (FidCostCodomain<OutCod>, ElemFidCostCodomain) {
    (
        FidCostCodomain::new(|a| a.obj1, |a| a.fid10, |a| a.cost2),
        ElemFidCostCodomain {
            value: 1.1,
            cost: 2.2,
            fidelity: EvalState::Completed,
        },
    )
}
pub fn get_elemfidconst() -> (FidConstCodomain<OutCod>, ElemFidConstCodomain) {
    (
        FidConstCodomain::new(
            |a| a.obj1,
            |a| a.fid10,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemFidConstCodomain {
            value: 1.1,
            constraints: Box::from([2.2, 3.3]),
            fidelity: EvalState::Completed,
        },
    )
}
pub fn get_elemfidcostconst() -> (FidCostConstCodomain<OutCod>, ElemFidCostConstCodomain) {
    (
        FidCostConstCodomain::new(
            |a| a.obj1,
            |a| a.fid10,
            |a| a.cost2,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemFidCostConstCodomain {
            value: 1.1,
            cost: 2.2,
            constraints: Box::from([3.3, 4.4]),
            fidelity: EvalState::Completed,
        },
    )
}
pub fn get_elemfidmulti() -> (FidMultiCodomain<OutCod>, ElemFidMultiCodomain) {
    (
        FidMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            |a| a.fid10,
        ),
        ElemFidMultiCodomain {
            value: Box::from([1.1, 2.2]),
            fidelity: EvalState::Completed,
        },
    )
}
pub fn get_elemfidcostmulti() -> (FidCostMultiCodomain<OutCod>, ElemFidCostMultiCodomain) {
    (
        FidCostMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            |a| a.fid10,
            |a| a.cost2,
        ),
        ElemFidCostMultiCodomain {
            value: Box::from([1.1, 2.2]),
            fidelity: EvalState::Completed,
            cost: 3.3,
        },
    )
}
pub fn get_elemfidconstmulti() -> (FidConstMultiCodomain<OutCod>, ElemFidConstMultiCodomain) {
    (
        FidConstMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            |a| a.fid10,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemFidConstMultiCodomain {
            value: Box::from([1.1, 2.2]),
            fidelity: EvalState::Completed,
            constraints: Box::from([3.3, 4.4]),
        },
    )
}
pub fn get_elemfidcostconstmulti() -> (
    FidCostConstMultiCodomain<OutCod>,
    ElemFidCostConstMultiCodomain,
) {
    (
        FidCostConstMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            |a| a.fid10,
            |a| a.cost2,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemFidCostConstMultiCodomain {
            value: Box::from([1.1, 2.2]),
            fidelity: EvalState::Completed,
            cost: 3.3,
            constraints: Box::from([4.4, 5.5]),
        },
    )
}
