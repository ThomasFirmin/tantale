use super::init_dom::*;
use paste::paste;

// make_test!(
// lower ; 0.0 => 100,
// mid ; 0.5 => 110,
// upper ; 1.0 => 120;
// real        => int,
// 
// ...
// )
macro_rules! make_test {
    ($(
        $(
            $name:ident; $in:expr => $out:expr
        ),*;
        $dom1:ident => $dom2:ident
    ),*
    ) => {
        paste!{
            $(
                $(
                    #[test]
                    fn [<$dom1 _into_ $dom2 _ $name>]() {
                        let domain_1 = [<get_domain_ $dom1>]();
                        let domain_2 = [<get_domain_ $dom2 _2>]();

                        let point = $in;
                        
                        let mapped = domain_1
                            .onto(&point, &domain_2)
                            .expect(concat!("Error in mapping ",stringify!($name)," from ", stringify!($dom1)," to ",stringify!($dom2),"."));
                        assert_eq!(
                            mapped, $out,
                            concat!("Mapping ",stringify!($name)," of ", stringify!($dom1)," to ",stringify!($dom2)," does not match.")
                        )
                    }

                    #[test]
                    fn [<base_ $dom1 _into_ $dom2 _ $name>]() {
                        let domain_1 = [<get_domain_ $dom1>]();
                        let input = $in;
                        let (domain_1,input) = [<get_domain_base_ $dom1>](domain_1,input);
                        
                        let domain_2 = [<get_domain_ $dom2 _2>]();
                        let output = $out;

                        let mapped = domain_1
                            .onto(&input, &domain_2)
                            .expect(concat!("Error in mapping ",stringify!($name)," from ", stringify!($dom1)," to ",stringify!($dom2),"."));
                        assert_eq!(
                            mapped, output,
                            concat!("Mapping ",stringify!($name)," of ", stringify!($dom1)," to ",stringify!($dom2)," does not match.")
                        )
                    }

                    #[test]
                    fn [<$dom1 _into_base_ $dom2 _ $name>]() {
                        let domain_1 = [<get_domain_ $dom1>]();
                        let input = $in;
                        
                        let domain_2 = [<get_domain_ $dom2 _2>]();
                        let output = $out;
                        let (domain_2,output) = [<get_domain_base_ $dom2>](domain_2,output);
                        
                        let mapped = domain_1
                            .onto(&input, &domain_2)
                            .expect(concat!("Error in mapping ",stringify!($name)," from ", stringify!($dom1)," to ",stringify!($dom2),"."));
                        assert_eq!(
                            mapped, output,
                            concat!("Mapping ",stringify!($name)," of ", stringify!($dom1)," to ",stringify!($dom2)," does not match.")
                        )
                    }

                    #[test]
                    fn [<base_ $dom1 _into_base_ $dom2 _ $name>]() {
                        let domain_1 = [<get_domain_ $dom1>]();
                        let input = $in;
                        let (domain_1,input) = [<get_domain_base_ $dom1>](domain_1,input);
                        
                        let domain_2 = [<get_domain_ $dom2 _2>]();
                        let output = $out;
                        let (domain_2,output) = [<get_domain_base_ $dom2>](domain_2,output);
                        
                        let mapped = domain_1
                            .onto(&input, &domain_2)
                            .expect(concat!("Error in mapping ",stringify!($name)," from ", stringify!($dom1)," to ",stringify!($dom2),"."));
                        assert_eq!(
                            mapped, output,
                            concat!("Mapping ",stringify!($name)," of ", stringify!($dom1)," to ",stringify!($dom2)," does not match.")
                        )
                    }
                )+
            )+
        }
    };
}

// ___---___REAL___---___ //
make_test!(
    lower ; 0.0 => 80.0,
    mid ; 5.0 => 90.0,
    upper ; 10.0 => 100.0;
    real      => real,

    lower ; 0.0 => 80,
    mid ; 5.0 => 90,
    upper ; 10.0 => 100;
    real      => nat ,

    lower ; 0.0 => 80,
    mid ; 5.0 => 90,
    upper ; 10.0 => 100;
    real      => int ,

    lower ; 0.0 => false,
    mid ; 5.0 => false,
    upper ; 10.0 => true;
    real      => bool ,

    lower ; 0.0 => "relu",
    mid ; 5.0 => "tanh",
    upper ; 10.0 => "sigmoid";
    real      => cat ,

    lower ; 0.0 => 0.0,
    mid ; 5.0 => 0.5,
    upper ; 10.0 => 1.0;
    real      => unit
);

// ___---___NAT___---___ //
make_test!(
    lower ; 1 => 80.0,
    mid ; 6 => 90.0,
    upper ; 11 => 100.0;
    nat      => real ,

    lower ; 1 => 80,
    mid ; 6 => 90,
    upper ; 11 => 100;
    nat      => nat ,

    lower ; 1 => 80,
    mid ; 6 => 90,
    upper ; 11 => 100;
    nat      => int ,

    lower ; 1 => false,
    mid ; 6 => false,
    upper ; 11 => true;
    nat      => bool ,

    lower ; 1 => "relu",
    mid ; 6 => "tanh",
    upper ; 11 => "sigmoid";
    nat      => cat ,

    lower ; 1 => 0.0,
    mid ; 6 => 0.5,
    upper ; 11 => 1.0;
    nat      => unit 
);

// ___---___INT___---___ //
make_test!(
    lower ; 0 => 80.0,
    mid ; 5 => 90.0,
    upper ; 10 => 100.0;
    int      => real ,

    lower ; 0 => 80,
    mid ; 5 => 90,
    upper ; 10 => 100;
    int      => nat ,

    lower ; 0 => 80,
    mid ; 5 => 90,
    upper ; 10 => 100;
    int      => int ,

    lower ; 0 => false,
    mid ; 5 => false,
    upper ; 10 => true;
    int      => bool ,

    lower ; 0 => "relu",
    mid ; 5 => "tanh",
    upper ; 10 => "sigmoid";
    int      => cat ,

    lower ; 0 => 0.0,
    mid ; 5 => 0.5,
    upper ; 10 => 1.0;
    int      => unit
);

// ___---___BOOL___---___ //
make_test!(
    lower ; false => 80.0,
    upper ; true => 100.0;
    bool      => real ,

    lower ; false => 80,
    upper ; true => 100;
    bool      => nat ,

    lower ; false => 80,
    upper ; true => 100;
    bool      => int ,

    lower ; false => 0.0,
    upper ; true => 1.0;
    bool      => unit
);

// ___---___CAT___---___ //
make_test!(
    lower ; "relu" => 80.0,
    mid ; "tanh" => 90.0,
    upper ; "sigmoid" => 100.0;
    cat      => real ,

    lower ; "relu" => 80,
    mid ; "tanh" => 90,
    upper ; "sigmoid" => 100;
    cat      => nat,

    lower ; "relu" => 80,
    mid ; "tanh" => 90,
    upper ; "sigmoid" => 100;
    cat      => int ,

    lower ; "relu" => 0.0,
    mid ; "tanh" => 0.5,
    upper ; "sigmoid" => 1.0;
    cat      => unit
);

// ___---___UNIT___---___ //
make_test!(
    lower ; 0.0 => 80.0,
    mid ; 0.5 => 90.0,
    upper ; 1.0 => 100.0;
    unit      => real ,

    lower ; 0.0 => 80,
    mid ; 0.5 => 90,
    upper ; 1.0 => 100;
    unit      => nat ,

    lower ; 0.0 => 80,
    mid ; 0.5 => 90,
    upper ; 1.0 => 100;
    unit      => int ,

    lower ; 0.0 => false,
    mid ; 0.5 => false,
    upper ; 1.0 => true;
    unit      => bool ,

    lower ; 0.0 => "relu",
    mid ; 0.5 => "tanh",
    upper ; 1.0 => "sigmoid";
    unit => cat
);