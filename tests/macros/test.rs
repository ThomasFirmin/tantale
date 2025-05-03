#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
pub mod searchspace {
    use tantale_macros::sp;
    use tantale_core::domain::{Real, Int, Nat, Bool};
    use tantale_core::domain::sampler::{uniform_real, uniform_nat};
    pub enum _TantaleMixedObj {
        Bool(Bool),
        Real(Real),
        Nat(Nat),
    }
    impl std::fmt::Display for _TantaleMixedObj {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                Self::Real(d) => std::fmt::Display::fmt(&d, f),
                Self::Nat(d) => std::fmt::Display::fmt(&d, f),
            }
        }
    }
    impl std::fmt::Debug for _TantaleMixedObj {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                Self::Real(d) => std::fmt::Display::fmt(&d, f),
                Self::Nat(d) => std::fmt::Display::fmt(&d, f),
            }
        }
    }
    pub enum _TantaleMixedObjTypeDom {
        Bool(<Bool as tantale_core::Domain>::TypeDom),
        Real(<Real as tantale_core::Domain>::TypeDom),
        Nat(<Nat as tantale_core::Domain>::TypeDom),
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for _TantaleMixedObjTypeDom {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                _TantaleMixedObjTypeDom::Bool(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Bool",
                        &__self_0,
                    )
                }
                _TantaleMixedObjTypeDom::Real(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Real",
                        &__self_0,
                    )
                }
                _TantaleMixedObjTypeDom::Nat(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Nat",
                        &__self_0,
                    )
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for _TantaleMixedObjTypeDom {}
    #[automatically_derived]
    impl ::core::clone::Clone for _TantaleMixedObjTypeDom {
        #[inline]
        fn clone(&self) -> _TantaleMixedObjTypeDom {
            let _: ::core::clone::AssertParamIsClone<
                <Bool as tantale_core::Domain>::TypeDom,
            >;
            let _: ::core::clone::AssertParamIsClone<
                <Real as tantale_core::Domain>::TypeDom,
            >;
            let _: ::core::clone::AssertParamIsClone<
                <Nat as tantale_core::Domain>::TypeDom,
            >;
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for _TantaleMixedObjTypeDom {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for _TantaleMixedObjTypeDom {
        #[inline]
        fn eq(&self, other: &_TantaleMixedObjTypeDom) -> bool {
            let __self_discr = ::core::intrinsics::discriminant_value(self);
            let __arg1_discr = ::core::intrinsics::discriminant_value(other);
            __self_discr == __arg1_discr
                && match (self, other) {
                    (
                        _TantaleMixedObjTypeDom::Bool(__self_0),
                        _TantaleMixedObjTypeDom::Bool(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    (
                        _TantaleMixedObjTypeDom::Real(__self_0),
                        _TantaleMixedObjTypeDom::Real(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    (
                        _TantaleMixedObjTypeDom::Nat(__self_0),
                        _TantaleMixedObjTypeDom::Nat(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    _ => unsafe { ::core::intrinsics::unreachable() }
                }
        }
    }
    impl std::fmt::Display for _TantaleMixedObjTypeDom {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                Self::Real(d) => std::fmt::Display::fmt(&d, f),
                Self::Nat(d) => std::fmt::Display::fmt(&d, f),
            }
        }
    }
    impl tantale_core::Domain for _TantaleMixedObj {
        type TypeDom = _TantaleMixedObjTypeDom;
        fn sample(&self, rng: &mut rand::prelude::ThreadRng) -> Self::TypeDom {
            match self {
                Self::Bool(d) => _TantaleMixedObjTypeDom::Bool(d.sample(rng)),
                Self::Real(d) => _TantaleMixedObjTypeDom::Real(d.sample(rng)),
                Self::Nat(d) => _TantaleMixedObjTypeDom::Nat(d.sample(rng)),
            }
        }
        fn is_in(&self, item: &Self::TypeDom) -> bool {
            match self {
                Self::Bool(d) => {
                    match item {
                        Self::TypeDom::Bool(i) => d.is_in(i),
                        _ => false,
                    }
                }
                Self::Real(d) => {
                    match item {
                        Self::TypeDom::Real(i) => d.is_in(i),
                        _ => false,
                    }
                }
                Self::Nat(d) => {
                    match item {
                        Self::TypeDom::Nat(i) => d.is_in(i),
                        _ => false,
                    }
                }
            }
        }
    }
    impl tantale_core::domain::Mixed for _TantaleMixedObj {}
    #[automatically_derived]
    impl ::core::clone::Clone for _TantaleMixedObj {
        #[inline]
        fn clone(&self) -> _TantaleMixedObj {
            match self {
                _TantaleMixedObj::Bool(__self_0) => {
                    _TantaleMixedObj::Bool(::core::clone::Clone::clone(__self_0))
                }
                _TantaleMixedObj::Real(__self_0) => {
                    _TantaleMixedObj::Real(::core::clone::Clone::clone(__self_0))
                }
                _TantaleMixedObj::Nat(__self_0) => {
                    _TantaleMixedObj::Nat(::core::clone::Clone::clone(__self_0))
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for _TantaleMixedObj {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for _TantaleMixedObj {
        #[inline]
        fn eq(&self, other: &_TantaleMixedObj) -> bool {
            let __self_discr = ::core::intrinsics::discriminant_value(self);
            let __arg1_discr = ::core::intrinsics::discriminant_value(other);
            __self_discr == __arg1_discr
                && match (self, other) {
                    (
                        _TantaleMixedObj::Bool(__self_0),
                        _TantaleMixedObj::Bool(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    (
                        _TantaleMixedObj::Real(__self_0),
                        _TantaleMixedObj::Real(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    (
                        _TantaleMixedObj::Nat(__self_0),
                        _TantaleMixedObj::Nat(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    _ => unsafe { ::core::intrinsics::unreachable() }
                }
        }
    }
    pub enum _TantaleMixedOpt {
        Bool(Bool),
        Nat(Nat),
        Int(Int),
    }
    impl std::fmt::Display for _TantaleMixedOpt {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                Self::Nat(d) => std::fmt::Display::fmt(&d, f),
                Self::Int(d) => std::fmt::Display::fmt(&d, f),
            }
        }
    }
    impl std::fmt::Debug for _TantaleMixedOpt {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                Self::Nat(d) => std::fmt::Display::fmt(&d, f),
                Self::Int(d) => std::fmt::Display::fmt(&d, f),
            }
        }
    }
    pub enum _TantaleMixedOptTypeDom {
        Bool(<Bool as tantale_core::Domain>::TypeDom),
        Nat(<Nat as tantale_core::Domain>::TypeDom),
        Int(<Int as tantale_core::Domain>::TypeDom),
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for _TantaleMixedOptTypeDom {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                _TantaleMixedOptTypeDom::Bool(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Bool",
                        &__self_0,
                    )
                }
                _TantaleMixedOptTypeDom::Nat(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Nat",
                        &__self_0,
                    )
                }
                _TantaleMixedOptTypeDom::Int(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Int",
                        &__self_0,
                    )
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for _TantaleMixedOptTypeDom {}
    #[automatically_derived]
    impl ::core::clone::Clone for _TantaleMixedOptTypeDom {
        #[inline]
        fn clone(&self) -> _TantaleMixedOptTypeDom {
            let _: ::core::clone::AssertParamIsClone<
                <Bool as tantale_core::Domain>::TypeDom,
            >;
            let _: ::core::clone::AssertParamIsClone<
                <Nat as tantale_core::Domain>::TypeDom,
            >;
            let _: ::core::clone::AssertParamIsClone<
                <Int as tantale_core::Domain>::TypeDom,
            >;
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for _TantaleMixedOptTypeDom {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for _TantaleMixedOptTypeDom {
        #[inline]
        fn eq(&self, other: &_TantaleMixedOptTypeDom) -> bool {
            let __self_discr = ::core::intrinsics::discriminant_value(self);
            let __arg1_discr = ::core::intrinsics::discriminant_value(other);
            __self_discr == __arg1_discr
                && match (self, other) {
                    (
                        _TantaleMixedOptTypeDom::Bool(__self_0),
                        _TantaleMixedOptTypeDom::Bool(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    (
                        _TantaleMixedOptTypeDom::Nat(__self_0),
                        _TantaleMixedOptTypeDom::Nat(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    (
                        _TantaleMixedOptTypeDom::Int(__self_0),
                        _TantaleMixedOptTypeDom::Int(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    _ => unsafe { ::core::intrinsics::unreachable() }
                }
        }
    }
    impl std::fmt::Display for _TantaleMixedOptTypeDom {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                Self::Nat(d) => std::fmt::Display::fmt(&d, f),
                Self::Int(d) => std::fmt::Display::fmt(&d, f),
            }
        }
    }
    impl tantale_core::Domain for _TantaleMixedOpt {
        type TypeDom = _TantaleMixedOptTypeDom;
        fn sample(&self, rng: &mut rand::prelude::ThreadRng) -> Self::TypeDom {
            match self {
                Self::Bool(d) => _TantaleMixedOptTypeDom::Bool(d.sample(rng)),
                Self::Nat(d) => _TantaleMixedOptTypeDom::Nat(d.sample(rng)),
                Self::Int(d) => _TantaleMixedOptTypeDom::Int(d.sample(rng)),
            }
        }
        fn is_in(&self, item: &Self::TypeDom) -> bool {
            match self {
                Self::Bool(d) => {
                    match item {
                        Self::TypeDom::Bool(i) => d.is_in(i),
                        _ => false,
                    }
                }
                Self::Nat(d) => {
                    match item {
                        Self::TypeDom::Nat(i) => d.is_in(i),
                        _ => false,
                    }
                }
                Self::Int(d) => {
                    match item {
                        Self::TypeDom::Int(i) => d.is_in(i),
                        _ => false,
                    }
                }
            }
        }
    }
    impl tantale_core::domain::Mixed for _TantaleMixedOpt {}
    #[automatically_derived]
    impl ::core::clone::Clone for _TantaleMixedOpt {
        #[inline]
        fn clone(&self) -> _TantaleMixedOpt {
            match self {
                _TantaleMixedOpt::Bool(__self_0) => {
                    _TantaleMixedOpt::Bool(::core::clone::Clone::clone(__self_0))
                }
                _TantaleMixedOpt::Nat(__self_0) => {
                    _TantaleMixedOpt::Nat(::core::clone::Clone::clone(__self_0))
                }
                _TantaleMixedOpt::Int(__self_0) => {
                    _TantaleMixedOpt::Int(::core::clone::Clone::clone(__self_0))
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for _TantaleMixedOpt {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for _TantaleMixedOpt {
        #[inline]
        fn eq(&self, other: &_TantaleMixedOpt) -> bool {
            let __self_discr = ::core::intrinsics::discriminant_value(self);
            let __arg1_discr = ::core::intrinsics::discriminant_value(other);
            __self_discr == __arg1_discr
                && match (self, other) {
                    (
                        _TantaleMixedOpt::Bool(__self_0),
                        _TantaleMixedOpt::Bool(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    (
                        _TantaleMixedOpt::Nat(__self_0),
                        _TantaleMixedOpt::Nat(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    (
                        _TantaleMixedOpt::Int(__self_0),
                        _TantaleMixedOpt::Int(__arg1_0),
                    ) => __self_0 == __arg1_0,
                    _ => unsafe { ::core::intrinsics::unreachable() }
                }
        }
    }
    pub fn get_searchpace<'a>() -> Vec<
        tantale_core::variable::var::Var<'a, _TantaleMixedObj, _TantaleMixedOpt>,
    > {
        use tantale_core::domain::{Onto, Domain};
        let mut variables: Vec<
            tantale_core::variable::var::Var<'a, _TantaleMixedObj, _TantaleMixedOpt>,
        > = Vec::new();
        variables
            .push(tantale_core::variable::var::Var {
                name: "a",
                domain_obj: std::rc::Rc::new(
                    _TantaleMixedObj::Real(Real::new(0.0, 1.0)),
                ),
                domain_opt: std::rc::Rc::new(_TantaleMixedOpt::Int(Int::new(0, 1))),
                sampler_obj: |dom, rng| match dom {
                    _TantaleMixedObj::Real(d) => {
                        _TantaleMixedObjTypeDom::Real(uniform_real(d, rng))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "An error occured while sampling from a mixed domain. The mixed variant is of wrong type.",
                                ),
                            ),
                        );
                    }
                },
                sampler_opt: |dom, rng| match dom {
                    _TantaleMixedOpt::Int(d) => {
                        _TantaleMixedOptTypeDom::Int(Int::sample(d, rng))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "An error occured while sampling from a mixed domain. The mixed variant is of wrong type.",
                                ),
                            ),
                        );
                    }
                },
                _onto_obj_fn: |indom, sample, outdom| match indom {
                    _TantaleMixedOpt::Int(i) => {
                        match outdom {
                            _TantaleMixedObj::Real(o) => {
                                match sample {
                                    _TantaleMixedOptTypeDom::Int(s) => {
                                        let mapped = Int::onto(i, s, o);
                                        match mapped {
                                            Ok(m) => Ok(_TantaleMixedObjTypeDom::Real(m)),
                                            Err(e) => Err(e),
                                        }
                                    }
                                    _ => {
                                        ::core::panicking::panic_fmt(
                                            format_args!(
                                                "internal error: entered unreachable code: {0}",
                                                format_args!(
                                                    "An error occured while mapping an item between mixed domains. The input sample is of the wrong type.",
                                                ),
                                            ),
                                        );
                                    }
                                }
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "An error occured while mapping an item between mixed domains. The output domain is of the wrong type.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "An error occured while mapping an item from a domain to a mixed domain. The input domain is of the wrong type.",
                                ),
                            ),
                        );
                    }
                },
                _onto_opt_fn: |indom, sample, outdom| match indom {
                    _TantaleMixedObj::Real(i) => {
                        match outdom {
                            _TantaleMixedOpt::Int(o) => {
                                match sample {
                                    _TantaleMixedObjTypeDom::Real(s) => {
                                        let mapped = Real::onto(i, s, o);
                                        match mapped {
                                            Ok(m) => Ok(_TantaleMixedOptTypeDom::Int(m)),
                                            Err(e) => Err(e),
                                        }
                                    }
                                    _ => {
                                        ::core::panicking::panic_fmt(
                                            format_args!(
                                                "internal error: entered unreachable code: {0}",
                                                format_args!(
                                                    "An error occured while mapping an item between mixed domains. The input sample is of the wrong type.",
                                                ),
                                            ),
                                        );
                                    }
                                }
                            }
                            _ => {
                                ::core::panicking::panic_fmt(
                                    format_args!(
                                        "internal error: entered unreachable code: {0}",
                                        format_args!(
                                            "An error occured while mapping an item between mixed domains. The output domain is of the wrong type.",
                                        ),
                                    ),
                                );
                            }
                        }
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "An error occured while mapping an item from a domain to a mixed domain. The input domain is of the wrong type.",
                                ),
                            ),
                        );
                    }
                },
            });
        variables
            .push(tantale_core::variable::var::Var {
                name: "b",
                domain_obj: std::rc::Rc::new(_TantaleMixedObj::Nat(Nat::new(0, 1))),
                domain_opt: std::rc::Rc::new(_TantaleMixedOpt::Nat(Nat::new(0, 1))),
                sampler_obj: |dom, rng| match dom {
                    _TantaleMixedObj::Nat(d) => {
                        _TantaleMixedObjTypeDom::Nat(Nat::sample(d, rng))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "An error occured while sampling from a mixed domain. The mixed variant is of wrong type.",
                                ),
                            ),
                        );
                    }
                },
                sampler_opt: |dom, rng| match dom {
                    _TantaleMixedOpt::Nat(d) => {
                        _TantaleMixedOptTypeDom::Nat(uniform_nat(d, rng))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "An error occured while sampling from a mixed domain. The mixed variant is of wrong type.",
                                ),
                            ),
                        );
                    }
                },
                _onto_obj_fn: |_indom, sample, _outdom| match sample {
                    _TantaleMixedOptTypeDom::Nat(s) => {
                        Ok(_TantaleMixedObjTypeDom::Nat(s.clone()))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "The input sample is of the wrong type in a mixed onto mixed (single) function.",
                                ),
                            ),
                        );
                    }
                },
                _onto_opt_fn: |_indom, sample, _outdom| match sample {
                    _TantaleMixedObjTypeDom::Nat(s) => {
                        Ok(_TantaleMixedOptTypeDom::Nat(s.clone()))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "The input sample is of the wrong type in a mixed onto mixed (single) function.",
                                ),
                            ),
                        );
                    }
                },
            });
        variables
            .push(tantale_core::variable::var::Var {
                name: "c",
                domain_obj: std::rc::Rc::new(_TantaleMixedObj::Bool(Bool::new())),
                domain_opt: std::rc::Rc::new(_TantaleMixedOpt::Bool(Bool::new())),
                sampler_obj: |dom, rng| match dom {
                    _TantaleMixedObj::Bool(d) => {
                        _TantaleMixedObjTypeDom::Bool(Bool::sample(d, rng))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "An error occured while sampling from a mixed domain. The mixed variant is of wrong type.",
                                ),
                            ),
                        );
                    }
                },
                sampler_opt: |dom, rng| match dom {
                    _TantaleMixedOpt::Bool(d) => {
                        _TantaleMixedOptTypeDom::Bool(Bool::sample(d, rng))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "An error occured while sampling from a mixed domain. The mixed variant is of wrong type.",
                                ),
                            ),
                        );
                    }
                },
                _onto_obj_fn: |_indom, sample, _outdom| match sample {
                    _TantaleMixedOptTypeDom::Bool(s) => {
                        Ok(_TantaleMixedObjTypeDom::Bool(s.clone()))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "The input sample is of the wrong type in a mixed onto mixed (single) function.",
                                ),
                            ),
                        );
                    }
                },
                _onto_opt_fn: |_indom, sample, _outdom| match sample {
                    _TantaleMixedObjTypeDom::Bool(s) => {
                        Ok(_TantaleMixedOptTypeDom::Bool(s.clone()))
                    }
                    _ => {
                        ::core::panicking::panic_fmt(
                            format_args!(
                                "internal error: entered unreachable code: {0}",
                                format_args!(
                                    "The input sample is of the wrong type in a mixed onto mixed (single) function.",
                                ),
                            ),
                        );
                    }
                },
            });
        variables
    }
}