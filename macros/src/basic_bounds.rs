use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Type, ItemImpl, parse_macro_input,
};

pub fn basic_bounds(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with syn::punctuated::Punctuated::<Type, syn::Token![,]>::parse_terminated);
    let item = parse_macro_input!(item as ItemImpl);

    let mut args = args.into_iter();
    let p = args.next().expect("expected P parameter");
    let opt = args.next().expect("expected Opt parameter");

    let mut item = item;

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
        ::tantale_core::utils::aliases::SOpt<#p>: ::tantale_core::HasX<X = <#p as ::tantale_core::Problem>::XOpt>
    });
    where_clause.predicates.push(
        syn::parse_quote!(
            <#p as ::tantale_core::Problem>::Scp: ::tantale_core::Linked<Opt = #opt> + ::tantale_core::utils::has_trait::HasVariables
        )
    );
    where_clause.predicates.push(
        syn::parse_quote!(
            #opt: ::tantale_core::Domain
        )
    );

    TokenStream::from(quote! {
        #item
    })
}