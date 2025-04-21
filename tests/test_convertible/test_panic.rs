use super::init_dom::*;
use paste::paste;

macro_rules! make_oob_test {
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
                    #[should_panic]
                    fn [<$dom1 _into_ $dom2 _oob_ $name>]() {
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
                )+
            )+
        }
    };
}

// ___---___REAL___---___ //
make_oob_test!(
    lower ; -1.0 => 80.0,
    upper ; 11.0 => 100.0;
    real      => real,
    
    lower ; -1.0 => 80,
    upper ; 11.0 => 100;
    real      => nat,
    
    lower ; -1.0 => 80,
    upper ; 11.0 => 100;
    real      => int,
    
    lower ; -1.0 => false,
    upper ; 11.0 => true;
    real      => bool,
    
    lower ; -1.0 => "relu",
    upper ; 11.0 => "sigmoid";
    real      => cat,
    
    lower ; -1.0 => 0.0,
    upper ; 11.0 => 1.0;
    real      => unit
);

// ___---___NAT___---___ //
make_oob_test!(
    lower ; 0 => 80.0,
    upper ; 12 => 100.0;
    nat      => real,
    
    lower ; 0 => 80,
    upper ; 12 => 100;
    nat      => nat,
    
    lower ; 0 => 80,
    upper ; 12 => 100;
    nat      => int,
    
    lower ; 0 => false,
    upper ; 12 => true;
    nat      => bool,
    
    lower ; 0 => "relu",
    upper ; 12 => "sigmoid";
    nat      => cat,
    
    lower ; 0 => 0.0,
    upper ; 12 => 1.0;
    nat      => unit
);

// ___---___INT___---___ //
make_oob_test!(
    lower ; -1 => 80.0,
    upper ; 11 => 100.0;
    int      => real,
    
    lower ; -1 => 80,
    upper ; 11 => 100;
    int      => nat,
    
    lower ; -1 => 80,
    upper ; 11 => 100;
    int      => int,
    
    lower ; -1 => false,
    upper ; 11 => true;
    int      => bool,
    
    lower ; -1 => "relu",
    upper ; 11 => "sigmoid";
    int      => cat,
    
    lower ; -1 => 0.0,
    upper ; 11 => 1.0;
    int      => unit
);

// ___---___CAT___---___ //
make_oob_test!(
    lower ; "pineapple" => 80.0;
    cat      => real,
    
    lower ; "pineapple" => 80;
    cat      => nat,
    
    lower ; "pineapple" => 80;
    cat      => int,
    
    lower ; "pineapple" => 0.0;
    cat      => unit
);

// ___---___UNIT___---___ //
make_oob_test!(
    lower ; -1.0 => 80.0,
    upper ; 1.1 => 100.0;
    unit      => real,
    
    lower ; -1.0 => 80,
    upper ; 1.1 => 100;
    unit      => nat,
    
    lower ; -1.0 => 80,
    upper ; 1.1 => 100;
    unit      => int,
    
    lower ; -1.0 => false,
    upper ; 1.1 => true;
    unit      => bool,
    
    lower ; -1.0 => "relu",
    upper ; 1.1 => "sigmoid";
    unit      => cat
    
);