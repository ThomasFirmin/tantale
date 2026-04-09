extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Token, parse_macro_input, punctuated::Punctuated};
use crate::hpo::{LineStream, parse_sp};

#[cfg(all(feature="py", not(feature="mpi")))]
/// Generates the complete Rust code for the searchspace.
///
/// Creates the public API that users interact with:
/// - Type aliases for ObjType and OptType
/// - The `get_searchspace()` function that returns a Sp<ObjType, OptType>
///
/// # Arguments
///
/// * `ident_mixed_obj` - The objective domain type identifier
/// * `ident_mixed_opt` - The optimizer domain type identifier
/// * `push_statements` - Generated code to push Var instances to the variables vector
///
/// # Returns
///
/// * `Ok` - TokenStream with complete searchspace implementation
/// * `Err` - Token generation error
///
/// # Generated Output
///
/// ```ignore
/// use tantale::core::domain::{Mixed, MixedTypeDom, Domain, NoDomain, onto::Onto};
///
/// pub type ObjType = /* ident_mixed_obj */;
/// pub type OptType = /* ident_mixed_opt */;
///
/// pub fn get_searchspace() -> tantale::core::searchspace::Sp<ObjType, OptType> {
///     let mut variables: Vec<Var<ObjType, OptType>> = Vec::new();
///     // ... all push_statements ...
///     Sp { var: variables.into() }
/// }
/// ```
pub fn py_get_sp_tokens(
    ident_mixed_obj: proc_macro2::Ident,
    ident_mixed_opt: proc_macro2::Ident,
    push_statements: Vec<proc_macro2::TokenStream>,
    const_statements: Vec<proc_macro2::TokenStream>,
    is_grid: bool,
) -> syn::Result<TokenStream> {
    if is_grid {
        Ok(quote! {

            use tantale::core::domain::{Grid,MixedTypeDom,Domain,NoDomain,onto::Onto};
            
            pub type ObjType = #ident_mixed_obj;
            pub type OptType = #ident_mixed_opt;

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "pytantale")]
            pub mod pytantale {
                #[pymodule_export]
                use tantale::python::PyStep;
                #[pymodule_export]
                use super::indices;
                #[pymodule_export]
                use super::extra;
            }

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "indices")]
            pub mod indices {
                #(
                    #[pymodule_export]
                    #const_statements
                )*
            }

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "extra")]
            pub mod extra { }
            
            pub fn get_searchspace() -> tantale::core::searchspace::Sp<#ident_mixed_obj,#ident_mixed_opt>
            {
                #(#push_statements)*

                tantale::core::searchspace::Sp{
                    var : variables.into(),
                }
            }
        }
        .into())
    } else {
        Ok(quote! {

            use tantale::core::domain::{Mixed,MixedTypeDom,Domain,NoDomain,onto::Onto};

            pub type ObjType = #ident_mixed_obj;
            pub type OptType = #ident_mixed_opt;

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "pytantale")]
            pub mod pytantale {
                #[pymodule_export]
                use tantale::python::PyStep;
                #[pymodule_export]
                use super::indices;
                #[pymodule_export]
                use super::extra;
            }

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "indices")]
            pub mod indices {
                #(
                    #[pymodule_export]
                    #const_statements
                )*
            }

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "extra")]
            pub mod extra { }

            pub fn get_searchspace() -> tantale::core::searchspace::Sp<#ident_mixed_obj,#ident_mixed_opt>
            {
                #(#push_statements)*

                tantale::core::searchspace::Sp{
                    var : variables.into(),
                }
            }
        }
        .into())
    }
}

#[cfg(all(feature="py", not(feature="mpi")))]
pub fn pyhpo(input: TokenStream) -> TokenStream {
    let lines = parse_macro_input!(
        input with Punctuated::<LineStream,Token![;]>::parse_terminated
    );

    let lines: Vec<LineStream> = lines.into_iter().collect();

    let (ident_mixed_obj, ident_mixed_opt, _, push_statements, const_statements, _, _, is_grid) =
        parse_sp(lines).unwrap();

    py_get_sp_tokens(ident_mixed_obj, ident_mixed_opt, push_statements, const_statements, is_grid).unwrap()
}


#[cfg(all(feature="py",feature="mpi"))]
/// Generates the complete Rust code for the searchspace.
///
/// Creates the public API that users interact with:
/// - Type aliases for ObjType and OptType
/// - The `get_searchspace()` function that returns a Sp<ObjType, OptType>
///
/// # Arguments
///
/// * `ident_mixed_obj` - The objective domain type identifier
/// * `ident_mixed_opt` - The optimizer domain type identifier
/// * `push_statements` - Generated code to push Var instances to the variables vector
///
/// # Returns
///
/// * `Ok` - TokenStream with complete searchspace implementation
/// * `Err` - Token generation error
///
/// # Generated Output
///
/// ```ignore
/// use tantale::core::domain::{Mixed, MixedTypeDom, Domain, NoDomain, onto::Onto};
///
/// pub type ObjType = /* ident_mixed_obj */;
/// pub type OptType = /* ident_mixed_opt */;
///
/// pub fn get_searchspace() -> tantale::core::searchspace::Sp<ObjType, OptType> {
///     let mut variables: Vec<Var<ObjType, OptType>> = Vec::new();
///     // ... all push_statements ...
///     Sp { var: variables.into() }
/// }
/// ```
pub fn py_get_sp_tokens(
    ident_mixed_obj: proc_macro2::Ident,
    ident_mixed_opt: proc_macro2::Ident,
    push_statements: Vec<proc_macro2::TokenStream>,
    const_statements: Vec<proc_macro2::TokenStream>,
    is_grid: bool,
) -> syn::Result<TokenStream> {
    if is_grid {
        Ok(quote! {

            use tantale::core::domain::{Grid,MixedTypeDom,Domain,NoDomain,onto::Onto};

            pub type ObjType = #ident_mixed_obj;
            pub type OptType = #ident_mixed_opt;

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "pytantale")]
            pub mod pytantale {
                #[pymodule_export]
                use tantale::python::PyStep;
                #[pymodule_export]
                use super::indices;
                #[pymodule_export]
                use super::extra;
            }

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "indices")]
            pub mod indices {
                #(
                    #[pymodule_export]
                    #const_statements
                )*
            }

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "extra")]
            pub mod extra {
                #[tantale::python::pyo3::prelude::pyfunction(crate = "tantale::python::pyo3")]
                pub fn mpi_rank() -> i32 {
                    tantale::core::MPI_RANK.get().cloned().unwrap()
                }
                
                #[tantale::python::pyo3::prelude::pyfunction(crate = "tantale::python::pyo3")]
                pub fn mpi_size() -> i32 {
                    tantale::core::MPI_SIZE.get().cloned().unwrap()
                }
            }

            pub fn get_searchspace() -> tantale::core::searchspace::Sp<#ident_mixed_obj,#ident_mixed_opt>
            {
                #(#push_statements)*

                tantale::core::searchspace::Sp{
                    var : variables.into(),
                }
            }
        }
        .into())
    } else {
        Ok(quote! {

            use tantale::core::domain::{Mixed,MixedTypeDom,Domain,NoDomain,onto::Onto};

            pub type ObjType = #ident_mixed_obj;
            pub type OptType = #ident_mixed_opt;

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "pytantale")]
            pub mod pytantale {
                #[pymodule_export]
                use tantale::python::PyStep;
                #[pymodule_export]
                use super::indices;
                #[pymodule_export]
                use super::extra;
            }

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "indices")]
            pub mod indices {
                #(
                    #[pymodule_export]
                    #const_statements
                )*
            }

            #[tantale::python::pyo3::pymodule(crate = "tantale::python::pyo3", name = "extra")]
            pub mod extra {
                #[tantale::python::pyo3::prelude::pyfunction(crate = "tantale::python::pyo3")]
                pub fn mpi_rank() -> i32 {
                    tantale::core::MPI_RANK.get().cloned().unwrap()
                }
                
                #[tantale::python::pyo3::prelude::pyfunction(crate = "tantale::python::pyo3")]
                pub fn mpi_size() -> i32 {
                    tantale::core::MPI_SIZE.get().cloned().unwrap()
                }
            }

            pub fn get_searchspace() -> tantale::core::searchspace::Sp<#ident_mixed_obj,#ident_mixed_opt>
            {
                #(#push_statements)*

                tantale::core::searchspace::Sp{
                    var : variables.into(),
                }
            }
        }
        .into())
    }
}

#[cfg(all(feature="py", feature="mpi"))]
pub fn pyhpo(input: TokenStream) -> TokenStream {
    let lines = parse_macro_input!(
        input with Punctuated::<LineStream,Token![;]>::parse_terminated
    );

    let lines: Vec<LineStream> = lines.into_iter().collect();

    let (ident_mixed_obj, ident_mixed_opt, _, push_statements, const_statements, _, _, is_grid) =
        parse_sp(lines).unwrap();

    py_get_sp_tokens(ident_mixed_obj, ident_mixed_opt, push_statements, const_statements, is_grid).unwrap()
}
