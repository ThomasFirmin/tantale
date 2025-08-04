pub use tantale::core::objective::codomain::{
    ElemConstCodomain, ElemConstMultiCodomain, ElemFidelCodomain, ElemFidelConstCodomain,
    ElemFidelConstMultiCodomain, ElemFidelMultiCodomain, ElemMultiCodomain, ElemSingleCodomain,
};

pub fn get_elemsingle() -> ElemSingleCodomain {
    ElemSingleCodomain { value: 1.1 }
}
pub fn get_elemfidel() -> ElemFidelCodomain {
    ElemFidelCodomain {
        value: 1.1,
        fidelity: 2.2,
    }
}
pub fn get_elemconst() -> ElemConstCodomain {
    ElemConstCodomain {
        value: 1.1,
        constraints: Box::from([2.2, 3.3]),
    }
}
pub fn get_elemfidelconst() -> ElemFidelConstCodomain {
    ElemFidelConstCodomain {
        value: 1.1,
        fidelity: 2.2,
        constraints: Box::from([3.3, 4.4]),
    }
}
pub fn get_elemmulti() -> ElemMultiCodomain {
    ElemMultiCodomain {
        value: Box::from([1.1, 2.2]),
    }
}
pub fn get_elemfidelmulti() -> ElemFidelMultiCodomain {
    ElemFidelMultiCodomain {
        value: Box::from([1.1, 2.2]),
        fidelity: 3.3,
    }
}
pub fn get_elemconstmulti() -> ElemConstMultiCodomain {
    ElemConstMultiCodomain {
        value: Box::from([1.1, 2.2]),
        constraints: Box::from([3.3, 4.4]),
    }
}
pub fn get_elemfidelconstmulti() -> ElemFidelConstMultiCodomain {
    ElemFidelConstMultiCodomain {
        value: Box::from([1.1, 2.2]),
        fidelity: 3.3,
        constraints: Box::from([4.4, 5.5]),
    }
}
