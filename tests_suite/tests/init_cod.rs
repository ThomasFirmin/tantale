pub use tantale::core::objective::{EvalStep, codomain::{
    ConstCodomain, ConstMultiCodomain, CostCodomain, CostConstCodomain, CostConstMultiCodomain,
    CostMultiCodomain, ElemConstCodomain, ElemConstMultiCodomain, ElemCostCodomain,
    ElemCostConstCodomain, ElemCostConstMultiCodomain, ElemCostMultiCodomain, ElemMultiCodomain,
    ElemSingleCodomain, FidCriteria, MultiCodomain, SingleCodomain,
}};
use tantale_macros::Outcome;

use serde::{Deserialize, Serialize};

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
    pub fid10: EvalStep,
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