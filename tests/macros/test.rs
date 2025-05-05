#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
pub mod searchspace {
    pub mod searchspace {
        use tantale_macros::sp;
        use tantale_core::domain::{Real, Int, Nat, Bool, Cat};
        use tantale_core::domain::sampler::{uniform_real, uniform_nat, uniform_int};
        static ACTIVATION: [&str; 3] = ["relu", "tanh", "sigmoid"];
        pub enum _TantaleMixedObj {
            Cat(Cat),
            Int(Int),
            Nat(Nat),
            Bool(Bool),
        }
        impl std::fmt::Display for _TantaleMixedObj {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    Self::Cat(d) => std::fmt::Display::fmt(&d, f),
                    Self::Int(d) => std::fmt::Display::fmt(&d, f),
                    Self::Nat(d) => std::fmt::Display::fmt(&d, f),
                    Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                }
            }
        }
        impl std::fmt::Debug for _TantaleMixedObj {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    Self::Cat(d) => std::fmt::Display::fmt(&d, f),
                    Self::Int(d) => std::fmt::Display::fmt(&d, f),
                    Self::Nat(d) => std::fmt::Display::fmt(&d, f),
                    Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                }
            }
        }
        pub enum _TantaleMixedObjTypeDom {
            Cat(<Cat as tantale_core::Domain>::TypeDom),
            Int(<Int as tantale_core::Domain>::TypeDom),
            Nat(<Nat as tantale_core::Domain>::TypeDom),
            Bool(<Bool as tantale_core::Domain>::TypeDom),
        }
        #[automatically_derived]
        impl ::core::fmt::Debug for _TantaleMixedObjTypeDom {
            #[inline]
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match self {
                    _TantaleMixedObjTypeDom::Cat(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "Cat",
                            &__self_0,
                        )
                    }
                    _TantaleMixedObjTypeDom::Int(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "Int",
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
                    _TantaleMixedObjTypeDom::Bool(__self_0) => {
                        ::core::fmt::Formatter::debug_tuple_field1_finish(
                            f,
                            "Bool",
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
                    <Cat as tantale_core::Domain>::TypeDom,
                >;
                let _: ::core::clone::AssertParamIsClone<
                    <Int as tantale_core::Domain>::TypeDom,
                >;
                let _: ::core::clone::AssertParamIsClone<
                    <Nat as tantale_core::Domain>::TypeDom,
                >;
                let _: ::core::clone::AssertParamIsClone<
                    <Bool as tantale_core::Domain>::TypeDom,
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
                            _TantaleMixedObjTypeDom::Cat(__self_0),
                            _TantaleMixedObjTypeDom::Cat(__arg1_0),
                        ) => __self_0 == __arg1_0,
                        (
                            _TantaleMixedObjTypeDom::Int(__self_0),
                            _TantaleMixedObjTypeDom::Int(__arg1_0),
                        ) => __self_0 == __arg1_0,
                        (
                            _TantaleMixedObjTypeDom::Nat(__self_0),
                            _TantaleMixedObjTypeDom::Nat(__arg1_0),
                        ) => __self_0 == __arg1_0,
                        (
                            _TantaleMixedObjTypeDom::Bool(__self_0),
                            _TantaleMixedObjTypeDom::Bool(__arg1_0),
                        ) => __self_0 == __arg1_0,
                        _ => unsafe { ::core::intrinsics::unreachable() }
                    }
            }
        }
        impl std::fmt::Display for _TantaleMixedObjTypeDom {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    Self::Cat(d) => std::fmt::Display::fmt(&d, f),
                    Self::Int(d) => std::fmt::Display::fmt(&d, f),
                    Self::Nat(d) => std::fmt::Display::fmt(&d, f),
                    Self::Bool(d) => std::fmt::Display::fmt(&d, f),
                }
            }
        }
        impl tantale_core::Domain for _TantaleMixedObj {
            type TypeDom = _TantaleMixedObjTypeDom;
            fn sample(&self, rng: &mut rand::prelude::ThreadRng) -> Self::TypeDom {
                match self {
                    Self::Cat(d) => _TantaleMixedObjTypeDom::Cat(d.sample(rng)),
                    Self::Int(d) => _TantaleMixedObjTypeDom::Int(d.sample(rng)),
                    Self::Nat(d) => _TantaleMixedObjTypeDom::Nat(d.sample(rng)),
                    Self::Bool(d) => _TantaleMixedObjTypeDom::Bool(d.sample(rng)),
                }
            }
            fn is_in(&self, item: &Self::TypeDom) -> bool {
                match self {
                    Self::Cat(d) => {
                        match item {
                            Self::TypeDom::Cat(i) => d.is_in(i),
                            _ => false,
                        }
                    }
                    Self::Int(d) => {
                        match item {
                            Self::TypeDom::Int(i) => d.is_in(i),
                            _ => false,
                        }
                    }
                    Self::Nat(d) => {
                        match item {
                            Self::TypeDom::Nat(i) => d.is_in(i),
                            _ => false,
                        }
                    }
                    Self::Bool(d) => {
                        match item {
                            Self::TypeDom::Bool(i) => d.is_in(i),
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
                    _TantaleMixedObj::Cat(__self_0) => {
                        _TantaleMixedObj::Cat(::core::clone::Clone::clone(__self_0))
                    }
                    _TantaleMixedObj::Int(__self_0) => {
                        _TantaleMixedObj::Int(::core::clone::Clone::clone(__self_0))
                    }
                    _TantaleMixedObj::Nat(__self_0) => {
                        _TantaleMixedObj::Nat(::core::clone::Clone::clone(__self_0))
                    }
                    _TantaleMixedObj::Bool(__self_0) => {
                        _TantaleMixedObj::Bool(::core::clone::Clone::clone(__self_0))
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
                            _TantaleMixedObj::Cat(__self_0),
                            _TantaleMixedObj::Cat(__arg1_0),
                        ) => __self_0 == __arg1_0,
                        (
                            _TantaleMixedObj::Int(__self_0),
                            _TantaleMixedObj::Int(__arg1_0),
                        ) => __self_0 == __arg1_0,
                        (
                            _TantaleMixedObj::Nat(__self_0),
                            _TantaleMixedObj::Nat(__arg1_0),
                        ) => __self_0 == __arg1_0,
                        (
                            _TantaleMixedObj::Bool(__self_0),
                            _TantaleMixedObj::Bool(__arg1_0),
                        ) => __self_0 == __arg1_0,
                        _ => unsafe { ::core::intrinsics::unreachable() }
                    }
            }
        }
        pub fn get_searchpace<'a>() -> Vec<
            tantale_core::variable::var::Var<'a, _TantaleMixedObj, Real>,
        > {
            pub use tantale_core::domain::{Onto, Domain};
            let mut variables: Vec<
                tantale_core::variable::var::Var<'a, _TantaleMixedObj, Real>,
            > = Vec::new();
            variables
                .push(tantale_core::variable::var::Var {
                    name: "a",
                    domain_obj: std::rc::Rc::new(
                        _TantaleMixedObj::Int(Int::new(0, 100)),
                    ),
                    domain_opt: std::rc::Rc::new(Real::new(0.0, 1.0)),
                    sampler_obj: |dom, rng| match dom {
                        _TantaleMixedObj::Int(d) => {
                            _TantaleMixedObjTypeDom::Int(uniform_int(d, rng))
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
                        Real::Real(d) => Real::Real(uniform_real(d, rng)),
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
                    _onto_obj_fn: |indom, sample, outdom| match outdom {
                        _TantaleMixedObj::Real(d) => {
                            let mapped = Real::onto(indom, sample, d);
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
                                        "An error occured while mapping an item from a domain to a mixed domain. The output domain is of the wrong type.",
                                    ),
                                ),
                            );
                        }
                    },
                    _onto_opt_fn: |indom, sample, outdom| match indom {
                        _TantaleMixedObj::Int(d) => {
                            let i = match sample {
                                _TantaleMixedObjTypeDom::Int(i) => i,
                                _ => {
                                    ::core::panicking::panic_fmt(
                                        format_args!(
                                            "internal error: entered unreachable code: {0}",
                                            format_args!(
                                                "An error occured while mapping an item from a mixed domain to a domain. The input item is of the wrong type.",
                                            ),
                                        ),
                                    );
                                }
                            };
                            Int::onto(d, i, outdom)
                        }
                        _ => {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "internal error: entered unreachable code: {0}",
                                    format_args!(
                                        "An error occured while mapping an item from a mixed domain to a domain. The mixed variant is of wrong type.",
                                    ),
                                ),
                            );
                        }
                    },
                });
            variables
                .push(tantale_core::variable::var::Var {
                    name: "b",
                    domain_obj: std::rc::Rc::new(
                        _TantaleMixedObj::Nat(Nat::new(0, 100)),
                    ),
                    domain_opt: std::rc::Rc::new(Real::new(0.0, 1.0)),
                    sampler_obj: |dom, rng| match dom {
                        _TantaleMixedObj::Nat(d) => {
                            _TantaleMixedObjTypeDom::Nat(uniform_nat(d, rng))
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
                        Real::Real(d) => Real::Real(Real::sample(d, rng)),
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
                    _onto_obj_fn: |indom, sample, outdom| match outdom {
                        _TantaleMixedObj::Real(d) => {
                            let mapped = Real::onto(indom, sample, d);
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
                                        "An error occured while mapping an item from a domain to a mixed domain. The output domain is of the wrong type.",
                                    ),
                                ),
                            );
                        }
                    },
                    _onto_opt_fn: |indom, sample, outdom| match indom {
                        _TantaleMixedObj::Nat(d) => {
                            let i = match sample {
                                _TantaleMixedObjTypeDom::Nat(i) => i,
                                _ => {
                                    ::core::panicking::panic_fmt(
                                        format_args!(
                                            "internal error: entered unreachable code: {0}",
                                            format_args!(
                                                "An error occured while mapping an item from a mixed domain to a domain. The input item is of the wrong type.",
                                            ),
                                        ),
                                    );
                                }
                            };
                            Nat::onto(d, i, outdom)
                        }
                        _ => {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "internal error: entered unreachable code: {0}",
                                    format_args!(
                                        "An error occured while mapping an item from a mixed domain to a domain. The mixed variant is of wrong type.",
                                    ),
                                ),
                            );
                        }
                    },
                });
            variables
                .push(tantale_core::variable::var::Var {
                    name: "c",
                    domain_obj: std::rc::Rc::new(
                        _TantaleMixedObj::Cat(Cat::new(&ACTIVATION)),
                    ),
                    domain_opt: std::rc::Rc::new(Real::new(0.0, 1.0)),
                    sampler_obj: |dom, rng| match dom {
                        _TantaleMixedObj::Cat(d) => {
                            _TantaleMixedObjTypeDom::Cat(Cat::sample(d, rng))
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
                        Real::Real(d) => Real::Real(Real::sample(d, rng)),
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
                    _onto_obj_fn: |indom, sample, outdom| match outdom {
                        _TantaleMixedObj::Real(d) => {
                            let mapped = Real::onto(indom, sample, d);
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
                                        "An error occured while mapping an item from a domain to a mixed domain. The output domain is of the wrong type.",
                                    ),
                                ),
                            );
                        }
                    },
                    _onto_opt_fn: |indom, sample, outdom| match indom {
                        _TantaleMixedObj::Cat(d) => {
                            let i = match sample {
                                _TantaleMixedObjTypeDom::Cat(i) => i,
                                _ => {
                                    ::core::panicking::panic_fmt(
                                        format_args!(
                                            "internal error: entered unreachable code: {0}",
                                            format_args!(
                                                "An error occured while mapping an item from a mixed domain to a domain. The input item is of the wrong type.",
                                            ),
                                        ),
                                    );
                                }
                            };
                            Cat::onto(d, i, outdom)
                        }
                        _ => {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "internal error: entered unreachable code: {0}",
                                    format_args!(
                                        "An error occured while mapping an item from a mixed domain to a domain. The mixed variant is of wrong type.",
                                    ),
                                ),
                            );
                        }
                    },
                });
            variables
                .push(tantale_core::variable::var::Var {
                    name: "d",
                    domain_obj: std::rc::Rc::new(_TantaleMixedObj::Bool(Bool::new())),
                    domain_opt: std::rc::Rc::new(Real::new(0.0, 1.0)),
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
                        Real::Real(d) => Real::Real(Real::sample(d, rng)),
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
                    _onto_obj_fn: |indom, sample, outdom| match outdom {
                        _TantaleMixedObj::Real(d) => {
                            let mapped = Real::onto(indom, sample, d);
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
                                        "An error occured while mapping an item from a domain to a mixed domain. The output domain is of the wrong type.",
                                    ),
                                ),
                            );
                        }
                    },
                    _onto_opt_fn: |indom, sample, outdom| match indom {
                        _TantaleMixedObj::Bool(d) => {
                            let i = match sample {
                                _TantaleMixedObjTypeDom::Bool(i) => i,
                                _ => {
                                    ::core::panicking::panic_fmt(
                                        format_args!(
                                            "internal error: entered unreachable code: {0}",
                                            format_args!(
                                                "An error occured while mapping an item from a mixed domain to a domain. The input item is of the wrong type.",
                                            ),
                                        ),
                                    );
                                }
                            };
                            Bool::onto(d, i, outdom)
                        }
                        _ => {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "internal error: entered unreachable code: {0}",
                                    format_args!(
                                        "An error occured while mapping an item from a mixed domain to a domain. The mixed variant is of wrong type.",
                                    ),
                                ),
                            );
                        }
                    },
                });
            variables
        }
    }
}