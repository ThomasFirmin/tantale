use proc_macro::TokenStream;
use syn::{Generics, Item, Type, parse_macro_input};

use crate::schema_crate_path;

fn add_basic_bounds(generics: &mut Generics, p: &Type, domain: &Option<Type>) {
    let tantale_crate = schema_crate_path();
    let where_clause = generics.make_where_clause();
    where_clause.predicates.push(syn::parse_quote! {
        #tantale_crate::utils::aliases::SOpt<#p>: #tantale_crate::HasX<X = <#p as #tantale_crate::Problem>::XOpt>
    });

    if let Some(opt) = domain {
        where_clause.predicates.push(
            syn::parse_quote!(
                #p: #tantale_crate::Problem<XOpt = ::std::sync::Arc<[<#opt as #tantale_crate::Domain>::Element]>>
            )
        );
        where_clause.predicates.push(syn::parse_quote!(
            <#p as #tantale_crate::Problem>::Scp: #tantale_crate::Linked<Opt = #opt>
        ));
        where_clause.predicates.push(syn::parse_quote!(
            #opt: #tantale_crate::Domain
        ));
    } else {
        where_clause.predicates.push(
            syn::parse_quote!(
                #p: #tantale_crate::Problem<XOpt = ::std::sync::Arc<[#tantale_crate::utils::aliases::TOpt<#p>]>>
            )
        );
    }

    where_clause.predicates.push(syn::parse_quote!(
        <#p as #tantale_crate::Problem>::Scp: #tantale_crate::utils::has_trait::HasVariables
    ));

    where_clause.predicates.push(syn::parse_quote!(
        <#p as #tantale_crate::Problem>::Id: #tantale_crate::StepId
    ));

    where_clause.predicates.push(syn::parse_quote! {
        #tantale_crate::utils::aliases::SOpt<#p>: #tantale_crate::HasX<X = <#p as #tantale_crate::Problem>::XOpt> + #tantale_crate::HasFidelity + #tantale_crate::HasStep + #tantale_crate::HasStepId
    });

    where_clause.predicates.push(syn::parse_quote! {
        #tantale_crate::utils::aliases::SObj<#p>: #tantale_crate::HasFidelity + #tantale_crate::HasStep + #tantale_crate::HasStepId
    });

    where_clause.predicates.push(syn::parse_quote! {
        #tantale_crate::utils::aliases::Shape<#p>: #tantale_crate::HasFidelity + #tantale_crate::HasStep + #tantale_crate::HasStepId
    });

    where_clause.predicates.push(syn::parse_quote! {
        #tantale_crate::utils::aliases::CompShape<#p>: #tantale_crate::HasFidelity + #tantale_crate::HasStep + #tantale_crate::HasStepId
    });

    where_clause.predicates.push(syn::parse_quote! {
        #tantale_crate::utils::aliases::Out<#p>: #tantale_crate::FidOutcome
    });

    // Add the remaining basic bounds here.
}

fn add_bounds_to_item(item: &mut Item, p: &Type, opt: &Option<Type>) -> syn::Result<()> {
    let generics = match item {
        Item::Impl(item) => &mut item.generics,
        Item::Struct(item) => &mut item.generics,
        Item::Enum(item) => &mut item.generics,
        Item::Trait(item) => &mut item.generics,
        Item::Fn(item) => &mut item.sig.generics,
        _ => {
            return Err(syn::Error::new_spanned(
                item,
                "expected an impl, struct, enum, function or trait",
            ));
        }
    };

    add_basic_bounds(generics, p, opt);

    Ok(())
}

pub fn fidelity_bounds(attr: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with syn::punctuated::Punctuated::<Type, syn::Token![,]>::parse_terminated);

    let mut args = args.into_iter();
    let Some(p) = args.next() else {
        return syn::Error::new(proc_macro2::Span::call_site(), "expected P parameter")
            .into_compile_error()
            .into();
    };
    let opt = args.next();

    let mut item = syn::parse_macro_input!(input as syn::Item);

    if let Err(error) = add_bounds_to_item(&mut item, &p, &opt) {
        return error.into_compile_error().into();
    }

    quote::quote!(#item).into()
}
