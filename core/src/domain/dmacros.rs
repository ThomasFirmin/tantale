#[macro_export]
macro_rules! mixed_sampler {
    ($(let $var:ident : $type:ident = $sampler:ident),+) => {
        $(
            paste::paste!(
            fn [<_tantale_wrapped_ $sampler>]<'a>(
                domain: &$crate::domain::base::BaseDom,
                rng: &mut rand::prelude::ThreadRng) ->
                <$crate::domain::base::BaseDom as $crate::domain::Domain>::TypeDom<'a>
            {
                if let $crate::domain::base::BaseDom::$type(d) = domain{
                    $crate::domain::base::BaseTypeDom::$type($sampler(&d,rng))
                }
                else{
                    panic!("Wrapped sampler called with the wrong BaseDom variant.")
                }
            }
            let $var = [<_tantale_wrapped_ $sampler>];
            )
        )+
    };
}

#[macro_export]
macro_rules! single_lhs {
    ($domain : expr) => {
        (Some($domain), None)
    };
    ($domain : expr => $sampler:ident) => {
        (Some($domain), $sampler)
    };
}

#[macro_export]
macro_rules! single_rhs {
    ($domain : expr) => {
        (Some($domain), None)
    };
    ($domain : expr => $sampler:ident) => {
        (Some($domain), Some($sampler))
    };
    (=> $sampler:ident) => {
        (None, Some($sampler))
    };
}

#[macro_export]
macro_rules! mixed_lhs {
    ($domain:expr) => {
        ($domain,None)
    };
    ($domain:expr => $sampler:ident) => {
        ($domain,Some(paste!([<_tantale_wrapped_ $sampler>])))
    };
}

#[macro_export]
macro_rules! mixed_rhs {
    ($domain:expr) => {
        (Some($domain), None)
    };
    ($domain:expr => $sampler:ident) => {
        (Some($domain), Some(paste!([<_tantale_wrapped_ $sampler>])))
    };
    (=>$sampler:ident) => {
        (None,Some(paste!([<_tantale_wrapped_ $sampler>])))
    };
    () => {
        (None,None)
    };
}

#[macro_export]
macro_rules! get_domain {
    (
            | name        | mixed   | mixed   |$(
            | $name:ident | $obj:tt | $opt:tt |)+
    ) => {
        $(
            let $name: () = (
                stringify!($name),
                $crate::core::domain::mixed_lhs!($obj),
                $crate::core::domain::mixed_lhs!($obj));
        )+
    };
    (
        | name        | single  | mixed   |$(
        | $name:ident | $obj:tt | $opt:tt |)+
    ) => {
        $(
            let (paste!([<obj_ $name>]),paste!([<obj_samp_ $name>])) = $crate::core::domain::single_lhs!($obj);
            let (paste!([<opt_ $name>]),paste!([<opt_samp_ $name>])) = $crate::core::domain::mixed_rhs!($opt);
        )+
    };
    (
        | name        | mixed   | single  |$(
        | $name:ident | $obj:tt | $opt:tt |)+
    ) => {
        $(
            let (paste!([<obj_ $name>]),paste!([<obj_samp_ $name>])) = $crate::core::domain::mixed_lhs!($obj);
            let (paste!([<opt_ $name>]),paste!([<opt_samp_ $name>])) = $crate::core::domain::single_rhs!($opt);
        )+
    };
    (
        | name        | single   | single  |$(
        | $name:ident | $obj:tt | $opt:tt |)+
    ) => {
        $(
            let (paste!([<obj_ $name>]),paste!([<obj_samp_ $name>])) = $crate::core::domain::single_lhs!($obj);
            let (paste!([<opt_ $name>]),paste!([<opt_samp_ $name>])) = $crate::core::domain::single_rhs!($opt);
        )+
    };
}
