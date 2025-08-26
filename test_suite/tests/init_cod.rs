use serde::{Deserialize, Serialize};
pub use tantale::core::objective::codomain::{
    ConstCodomain, ConstMultiCodomain, ElemConstCodomain, ElemConstMultiCodomain,
    ElemFidelCodomain, ElemFidelConstCodomain, ElemFidelConstMultiCodomain, ElemFidelMultiCodomain,
    ElemMultiCodomain, ElemSingleCodomain, FidelCodomain, FidelConstCodomain,
    FidelConstMultiCodomain, FidelMultiCodomain, MultiCodomain, SingleCodomain,
};
use tantale_macros::Outcome;

#[derive(Outcome, Debug, Serialize, Deserialize)]
pub struct OutCod {
    pub obj1: f64,
    pub fid2: f64,
    pub con3: f64,
    pub con4: f64,
    pub con5: f64,
    pub mul6: f64,
    pub mul7: f64,
    pub mul8: f64,
    pub mul9: f64,
    pub more: f64,
    pub info: f64,
}

pub fn get_elemsingle() -> (SingleCodomain<OutCod>, ElemSingleCodomain) {
    (
        SingleCodomain::new(|a| a.obj1),
        ElemSingleCodomain { value: 1.1 },
    )
}
pub fn get_elemfidel() -> (FidelCodomain<OutCod>, ElemFidelCodomain) {
    (
        FidelCodomain::new(|a| a.obj1, |a| a.fid2),
        ElemFidelCodomain {
            value: 1.1,
            fidelity: 2.2,
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
pub fn get_elemfidelconst() -> (FidelConstCodomain<OutCod>, ElemFidelConstCodomain) {
    (
        FidelConstCodomain::new(
            |a| a.obj1,
            |a| a.fid2,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemFidelConstCodomain {
            value: 1.1,
            fidelity: 2.2,
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
pub fn get_elemfidelmulti() -> (FidelMultiCodomain<OutCod>, ElemFidelMultiCodomain) {
    (
        FidelMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            |a| a.fid2,
        ),
        ElemFidelMultiCodomain {
            value: Box::from([1.1, 2.2]),
            fidelity: 3.3,
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
pub fn get_elemfidelconstmulti() -> (FidelConstMultiCodomain<OutCod>, ElemFidelConstMultiCodomain) {
    (
        FidelConstMultiCodomain::new(
            vec![|a: &OutCod| a.mul6, |a: &OutCod| a.mul7].into_boxed_slice(),
            |a| a.fid2,
            vec![|a: &OutCod| a.con3, |a: &OutCod| a.con4].into_boxed_slice(),
        ),
        ElemFidelConstMultiCodomain {
            value: Box::from([1.1, 2.2]),
            fidelity: 3.3,
            constraints: Box::from([4.4, 5.5]),
        },
    )
}
