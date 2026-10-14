use proc_macro::TokenStream;
use syn::{
    Generics, Item, Type, parse_macro_input,
};


fn add_basic_bounds(
    generics: &mut Generics,
    p: &Type,
    domain: &Option<Type>,
) {
    let where_clause = generics.make_where_clause();
    where_clause.predicates.push(syn::parse_quote! {
        ::tantale_core::utils::aliases::SOpt<#p>: ::tantale_core::HasX<X = <#p as ::tantale_core::Problem>::XOpt>
    });
    
    if let Some(opt) = domain {
        where_clause.predicates.push(
            syn::parse_quote!(
                #p: ::tantale_core::Problem<XOpt = ::std::sync::Arc<[<#opt as ::tantale_core::Domain>::Element]>>
            )
        );
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
    } else {
        where_clause.predicates.push(
            syn::parse_quote!(
                #p: ::tantale_core::Problem<XOpt = ::std::sync::Arc<[::tantale_core::utils::aliases::TOpt<#p>]>>
            )
        );
        where_clause.predicates.push(
            syn::parse_quote!(
                <#p as ::tantale_core::Problem>::Scp: ::tantale_core::utils::has_trait::HasVariables
            )
        );
    }

    // Add the remaining basic bounds here.
}

fn add_bounds_to_item(
    item: &mut Item,
    p: &Type,
    opt: &Option<Type>,
) -> syn::Result<()> {
    let generics = match item {
        Item::Impl(item) => &mut item.generics,
        Item::Struct(item) => &mut item.generics,
        Item::Enum(item) => &mut item.generics,
        Item::Trait(item) => &mut item.generics,

        _ => {
            return Err(syn::Error::new_spanned(
                item,
                "expected an impl, struct, enum, or trait",
            ));
        }
    };

    add_basic_bounds(generics, p, opt);

    Ok(())
}

pub fn basic_bounds(attr: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with syn::punctuated::Punctuated::<Type, syn::Token![,]>::parse_terminated);

    let mut args = args.into_iter();
    let Some(p) = args.next() else {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "expected P parameter",
        )
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