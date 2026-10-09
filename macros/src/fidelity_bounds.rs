use proc_macro::TokenStream;
use quote::quote;
use syn::{
    ItemImpl, parse_macro_input, Type,
};

pub fn fidelity_bounds(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(
        attr with
        syn::punctuated::Punctuated::<Type, syn::Token![,]>::parse_terminated
    );

    let mut item = parse_macro_input!(item as ItemImpl);

    let mut args = args.into_iter();
    let p = args.next().expect("expected P parameter");
    let opt = args.next().expect("expected Opt parameter");

    let where_clause = item.generics.make_where_clause();

    if let Some(info) = args.next() {
        where_clause.predicates.push(
            syn::parse_quote!(
                #p: ::tantale_core::Problem<XOpt = ::std::sync::Arc<[<#opt as ::tantale_core::Domain>::Element]>, Info=#info>
            )
        );
    } else {
        where_clause.predicates.push(
            syn::parse_quote!(
                #p: ::tantale_core::Problem<XOpt = ::std::sync::Arc<[<#opt as ::tantale_core::Domain>::Element]>>
            )
        );
    }

    where_clause.predicates.push(syn::parse_quote! {
        <#p as ::tantale_core::Problem>::Scp: ::tantale_core::Linked<Opt = #opt> + ::tantale_core::utils::has_trait::HasVariables
    });

    where_clause.predicates.push(syn::parse_quote! {
        #opt: ::tantale_core::Domain
    });

    where_clause.predicates.push(syn::parse_quote! {
        <#p as ::tantale_core::Problem>::Id: ::tantale_core::StepId
    });

    where_clause.predicates.push(syn::parse_quote! {
        ::tantale_core::utils::aliases::SOpt<#p>: ::tantale_core::HasX<X = <#p as ::tantale_core::Problem>::XOpt> + ::tantale_core::HasFidelity + ::tantale_core::HasStep
    });

    where_clause.predicates.push(syn::parse_quote! {
        ::tantale_core::utils::aliases::SObj<#p>: ::tantale_core::HasFidelity + ::tantale_core::HasStep
    });

    where_clause.predicates.push(syn::parse_quote! {
        ::tantale_core::utils::aliases::Shape<#p>: ::tantale_core::HasFidelity + ::tantale_core::HasStep
    });

    where_clause.predicates.push(syn::parse_quote! {
        ::tantale_core::utils::aliases::CompShape<#p>: ::tantale_core::HasFidelity + ::tantale_core::HasStep
    });

    where_clause.predicates.push(syn::parse_quote! {
        ::tantale_core::utils::aliases::Out<#p>: ::tantale_core::FidOutcome
    });

    TokenStream::from(quote! {
        #item
    })
}