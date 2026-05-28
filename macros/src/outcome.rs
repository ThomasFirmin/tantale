extern crate proc_macro;

use quote::quote;
use syn::{Field, ItemStruct, parse_macro_input, spanned::Spanned};
use proc_macro2::TokenStream;


pub struct Attributes{
    pub objectives: Vec<TokenStream>,

    pub constraint: Vec<TokenStream>,
    pub cost: Option<TokenStream>,
    pub step: Option<TokenStream>,
    pub is_multi: bool,
    pub is_constrained: bool,
}

impl Attributes {
    pub fn new() -> Self {
        Self {
            objectives: Vec::new(),
            constraint: Vec::new(),
            cost: None,
            step: None,
            is_multi: false,
            is_constrained: false,
        }
    }

    pub fn at_least_one_objective(&self) -> bool {
        !self.objectives.is_empty()
    }

    pub fn add_objective(&mut self, field: &Field, is_max: bool) {
        if !matches!(&field.ty, syn::Type::Path(p) if {
            let ident = &p.path.segments.last().unwrap().ident;
            matches!(ident.to_string().as_str(), "f64")
        }) {
            panic!(
                "{:?}",
                syn::Error::new(
                    field.span(),
                    "Objective fields must be of type f64."
                )
            );
        }
        let ident= &field.ident;
        if is_max {
            self.objectives.push(quote! { o.#ident });
        } else {
            self.objectives.push(quote! { - o.#ident });
        }
        if !self.is_multi && self.objectives.len() > 1 {
            self.is_multi = true;
        }
    }

    pub fn add_constraint(&mut self, field: &Field) {
        if !matches!(&field.ty, syn::Type::Path(p) if {
            let ident = &p.path.segments.last().unwrap().ident;
            matches!(ident.to_string().as_str(), "f64")
        }) {
            panic!(
                "{:?}",
                syn::Error::new(
                    field.span(),
                    "Constraints fields must be of type f64."
                )
            );
        }
        let ident= &field.ident;
        self.constraint.push(quote! { #ident });
        if !self.is_constrained && !self.constraint.is_empty() {
            self.is_constrained = true;
        }
    }

    pub fn add_cost(&mut self, field: &Field) {
        if !matches!(&field.ty, syn::Type::Path(p) if {
            let ident = &p.path.segments.last().unwrap().ident;
            matches!(ident.to_string().as_str(), "f64")
        }) {
            panic!(
                "{:?}",
                syn::Error::new(
                    field.span(),
                    "The Cost field must be of type f64."
                )
            );
        }
        let ident= &field.ident;
        if self.cost.is_some() {
            panic!(
                "{:?}",
                syn::Error::new(
                    field.span(),
                    "Only one Cost should be defined within an Outcome."
                )
            );
        }
        self.cost = Some(quote! { #ident });
    }

    pub fn add_step(&mut self, field: &Field) {
        if !matches!(&field.ty, syn::Type::Path(p) if {
            let ident = &p.path.segments.last().unwrap().ident;
            matches!(ident.to_string().as_str(), "Step")
        }) {
            panic!(
                "{:?}",
                syn::Error::new(
                    field.span(),
                    "Step field must be of type Step."
                )
            );
        }
        let ident= &field.ident;
        if self.step.is_some() {
            panic!(
                "{:?}",
                syn::Error::new(
                    field.span(),
                    "Only one Step should be defined within an Outcome."
                )
            );
        }
        self.step = Some(quote! { #ident });
    }

    pub fn parse_from_field(&mut self, field: &syn::Field) -> Result<(), syn::Error> {
        if field.ident.is_none() {
            return Err(syn::Error::new(field.span(), "Outcome fields must be named."));
        }
        for attr in &field.attrs {
            if attr.path().is_ident("maximize") {
                self.add_objective(field, true);
            } else if attr.path().is_ident("minimize") {
                self.add_objective(field, false);
            } else if attr.path().is_ident("constraint") {
                self.add_constraint(field);
            } else if attr.path().is_ident("cost") {
                self.add_cost(field);
            } else if attr.path().is_ident("step") {
                self.add_step(field);
            }
        }
        Ok(())
    }
}

/// Entry point for the `Outcome` derive macro.
///
/// This function processes a struct and derives implementations for:
/// 1. `Outcome` - Marker trait for objective outputs
/// 2. `FidOutcome` - Multi-fidelity tracking (if Step field exists)
///
pub fn proc_outcome(input: proc_macro::TokenStream) -> proc_macro::TokenStream {

    let mut attr = Attributes::new();

    let input = parse_macro_input!(input as ItemStruct);

    let eident = &input.ident;
    let egenerics = &input.generics;
    if !egenerics.params.is_empty() {
        panic!(
            "{:?}",
            syn::Error::new(
                egenerics.span(),
                "Outcome cannot have generics. Please remove any generic parameters from the struct definition."
            )
        );
    }
    
    // Parse helper attributes for objectives, constraints, costs, and step
    input.fields.iter().for_each(|field| {
        attr.parse_from_field(field).unwrap();
    });
    
    if !attr.at_least_one_objective() {
        panic!(
            "{:?}",
            syn::Error::new(
                input.span(),
                "At least one objective must be defined with #[maximize] or #[minimize]."
            )
        );
    }

    let outcome = match (attr.is_multi, attr.is_constrained, attr.cost.is_some()) {
        (false, false, false) => {
            let obj = &attr.objectives[0];
            quote! {
                impl tantale::core::Outcome for #eident {
                    type Cod = tantale::core::SingleCodomain<Self>;
                    fn codomain() -> Self::Cod {
                        tantale::core::SingleCodomain::new(
                            | o: &#eident | #obj
                        )
                    }
                }
            }
        },
        (true, false, false) => {
            let obj = attr.objectives;
            quote! {
                impl tantale::core::Outcome for #eident {
                    type Cod = tantale::core::MultiCodomain<Self>;
                    fn codomain() -> Self::Cod {
                        tantale::core::MultiCodomain::new(
                            Box::new(
                                [
                                     #( | o: &#eident | #obj ),*
                                ]
                            )
                        )
                    }
                }
            }
        },
        (false, true, false) => {
            let obj = &attr.objectives[0];
            let constraints = attr.constraint;
            quote! {
                impl tantale::core::Outcome for #eident {
                    type Cod = tantale::core::ConstCodomain<Self>;
                    fn codomain() -> Self::Cod {
                        tantale::core::ConstCodomain::new(
                            | o: &#eident | #obj,
                            Box::new(
                                [
                                     #( | o: &#eident | o.#constraints ),*
                                ]
                            )
                        )
                    }
                }
            }
        },
        (false, false, true) => {
            let obj = &attr.objectives[0];
            let cost = attr.cost.unwrap();
            quote! {
                impl tantale::core::Outcome for #eident {
                    type Cod = tantale::core::CostCodomain<Self>;
                    fn codomain() -> Self::Cod {
                        tantale::core::CostCodomain::new(
                            | o: &#eident | #obj,
                            | o: &#eident | o.#cost
                        )
                    }
                }
            }
        },
        (false, true, true) => {
            let obj = &attr.objectives[0];
            let cost = attr.cost.unwrap();
            let constraints = attr.constraint;
            quote! {
                impl tantale::core::Outcome for #eident {
                    type Cod = tantale::core::CostConstCodomain<Self>;
                    fn codomain() -> Self::Cod {
                        tantale::core::CostConstCodomain::new(
                            | o: &#eident | #obj,
                            | o: &#eident | o.#cost,
                            Box::new(
                                [
                                     #( | o: &#eident | o.#constraints ),*
                                ]
                            )
                        )
                    }
                }
            }
        },
        (true, true, false) => {
            let obj = attr.objectives;
            let constraints = attr.constraint;
            quote! {
                impl tantale::core::Outcome for #eident {
                    type Cod = tantale::core::ConstMultiCodomain<Self>;
                    fn codomain() -> Self::Cod {
                        tantale::core::ConstMultiCodomain::new(
                            Box::new(
                                [
                                     #( | o: &#eident | #obj ),*
                                ]
                            ),
                            Box::new(
                                [
                                     #( | o: &#eident | o.#constraints ),*
                                ]
                            )
                        )
                    }
                }
            }
        },
        (true, false, true) => {
            let obj = attr.objectives;
            let cost = attr.cost.unwrap();
            quote! {
                impl tantale::core::Outcome for #eident {
                    type Cod = tantale::core::CostMultiCodomain<Self>;
                    fn codomain() -> Self::Cod {
                        tantale::core::CostMultiCodomain::new(
                            Box::new(
                                [
                                     #( | o: &#eident | #obj ),*
                                ]
                            ),
                            | o: &#eident | o.#cost,
                        )
                    }
                }
            }
        },
        (true, true, true) => {
            let obj = attr.objectives;
            let constraints = attr.constraint;
            let cost = attr.cost.unwrap();
            quote! {
                impl tantale::core::Outcome for #eident {
                    type Cod = tantale::core::CostConstMultiCodomain<Self>;
                    fn codomain() -> Self::Cod {
                        tantale::core::CostConstMultiCodomain::new(
                            Box::new(
                                [
                                     #( | o: &#eident | #obj ),*
                                ]
                            ),
                            | o: &#eident | o.#cost,
                            Box::new(
                                [
                                     #( | o: &#eident | o.#constraints ),*
                                ]
                            )
                        )
                    }
                }
            }
        },
    };

    let fid_outcome = if let Some(step) = attr.step {
        quote! {
            impl tantale::core::FidOutcome for #eident {
                fn get_step(&self) -> tantale::core::EvalStep { self.#step.into() }
            }
        }
    } else {
        TokenStream::new()
    };

    quote! {
        #outcome
        #fid_outcome
    }.into()
}
