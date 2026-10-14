pub mod one_step_func{
    use serde::{Deserialize, Serialize};
    use tantale::{core::{
        BaseSol, Domain, EmptyInfo, OneStep, SId, utils::aliases::Elem
    }, macros::{Outcome, Recordable, Schema}};
    use std::sync::Arc;

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct OutStruct {
        #[maximize]
        out: f64,
    }

    fn example<ObjType:Domain>(_tantale_in: Arc<[Elem<ObjType>]>) -> OutStruct {
            OutStruct { out: 42.0 }
        }

    pub fn get_function<ObjType:Domain>() -> OneStep<BaseSol<SId, ObjType, EmptyInfo>, OutStruct> {
        OneStep::new(example::<ObjType>)
    }

    pub type FuncType<ObjType> = OneStep<BaseSol<SId, ObjType, EmptyInfo>, OutStruct>;
    pub type SolType<ObjType> = BaseSol<SId, ObjType, EmptyInfo>;
}

pub mod one_step_func_with_out{
    use tantale::{core::{BaseSol, Domain, EmptyInfo, OneStep, Outcome, SId, utils::aliases::Elem}};
    use std::sync::Arc;

    fn example<ObjType:Domain, Out:Outcome>(_tantale_in: Arc<[Elem<ObjType>]>) -> Out {
            unreachable!("This function is only used to test the OneStep type, it should not be called");
        }

    pub fn get_function<ObjType:Domain, Out:Outcome>() -> OneStep<BaseSol<SId, ObjType, EmptyInfo>, Out> {
        OneStep::new(example::<ObjType, Out>)
    }

    pub type FuncType<ObjType, Out> = OneStep<BaseSol<SId, ObjType, EmptyInfo>, Out>;
    pub type SolType<ObjType> = BaseSol<SId, ObjType, EmptyInfo>;
}

pub mod multi_step_func{
    use serde::{Deserialize, Serialize};
    use tantale::{core::{
        Domain, EmptyInfo, Fidelity, FidelitySol, FuncState, MultiStep, StepSId, Step, 
        utils::aliases::Elem}, 
        macros::{Outcome, Recordable, Schema}
    };
    use std::sync::Arc;

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct FidOutEvaluator {
        #[maximize]
        pub obj: f64,
        #[step]
        pub fid: Step,
    }

    #[derive(Serialize, Deserialize)]
    pub struct FnState {
        pub state: isize,
    }
    impl FuncState for FnState {
        fn save(&self, path: std::path::PathBuf) -> std::io::Result<()> {
            let mut file = std::fs::File::create(path.join("fn_state.mp"))?;
            rmp_serde::encode::write(&mut file, &self).unwrap();
            Ok(())
        }
        fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
            let file_path = path.join("fn_state.mp");
            let file = std::fs::File::open(file_path)?;
            let state = rmp_serde::decode::from_read(file).unwrap();
            Ok(state)
        }
    }

    fn example<ObjType:Domain>(_tantale_in: Arc<[Elem<ObjType>]>, _fid: Fidelity, _state: Option<FnState>) -> (FidOutEvaluator, FnState) {
        (FidOutEvaluator { obj: 42.0, fid: Step::Partially(1) }, FnState { state: 1 })
    }

    pub fn get_function<ObjType:Domain>() -> MultiStep<FidelitySol<StepSId, ObjType, EmptyInfo>, FidOutEvaluator, FnState> {
        MultiStep::new(example::<ObjType>)
    }

    pub type FuncType<ObjType> = MultiStep<FidelitySol<StepSId, ObjType, EmptyInfo>, FidOutEvaluator, FnState>;
    pub type SolType<ObjType> = FidelitySol<StepSId, ObjType, EmptyInfo>;
}

pub mod multi_step_func_with_out{
    use serde::{Deserialize, Serialize};
    use tantale::{core::{
        FidelitySol, Domain, EmptyInfo, FidOutcome, Fidelity, FuncState, MultiStep, StepSId, Step,
        utils::aliases::Elem
    },
    macros::{Outcome, Recordable, Schema}};
    use std::sync::Arc;

    #[derive(Outcome, Schema, Recordable, Debug, Serialize, Deserialize)]
    pub struct FidOutEvaluator {
        #[maximize]
        pub obj: f64,
        #[step]
        pub fid: Step,
    }

    #[derive(Serialize, Deserialize)]
    pub struct FnState {
        pub state: isize,
    }
    impl FuncState for FnState {
        fn save(&self, path: std::path::PathBuf) -> std::io::Result<()> {
            let mut file = std::fs::File::create(path.join("fn_state.mp"))?;
            rmp_serde::encode::write(&mut file, &self).unwrap();
            Ok(())
        }
        fn load(path: std::path::PathBuf) -> std::io::Result<Self> {
            let file_path = path.join("fn_state.mp");
            let file = std::fs::File::open(file_path)?;
            let state = rmp_serde::decode::from_read(file).unwrap();
            Ok(state)
        }
    }

    fn example<ObjType:Domain, Out:FidOutcome>(_tantale_in: Arc<[Elem<ObjType>]>, _fid: Fidelity, _state: Option<FnState>) -> (Out, FnState) {
        unreachable!("This function is only used to test the MultiStep type, it should not be called");
    }

    pub fn get_function<ObjType:Domain, Out: FidOutcome>() -> MultiStep<FidelitySol<StepSId, ObjType, EmptyInfo>, Out, FnState> {
        MultiStep::new(example::<ObjType, Out>)
    }

    pub type FuncType<ObjType, Out> = MultiStep<FidelitySol<StepSId, ObjType, EmptyInfo>, Out, FnState>;
    pub type SolType<ObjType> = FidelitySol<StepSId, ObjType, EmptyInfo>;
}


pub mod sp_mixed_to_single {
    use tantale::core::{
        domain::{Bool, Cat, Int, Nat, Real},
        distribution::{Bernoulli, Uniform},
    };
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        b | Nat(0,100, Uniform)                       | Real(0.0,1.0, Uniform)                 ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform) | Real(0.0,1.0, Uniform)                 ;
        d | Bool(Bernoulli(0.5))                      | Real(0.0,1.0, Uniform)                 ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_single_to_mixed {
    use tantale::core::domain::{Bool, Cat, Int, Nat, Real};
    use tantale::core::distribution::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform)          ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                    ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_mixed_to_nodomain {
    use tantale::core::domain::{Bool, Cat, Int, Nat};
    use tantale::core::distribution::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(0,100, Uniform)       | ;
        b | Nat(0,100, Uniform)       | ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
        d | Bool(Bernoulli(0.5))           | ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_single_to_nodomain {
    use tantale::core::domain::Real;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | ;
        b | Real(0.0,1.0, Uniform) | ;
        c | Real(0.0,1.0, Uniform) | ;
        d | Real(0.0,1.0, Uniform) | ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_single_to_single {
    use tantale::core::domain::{Int, Real};
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        b | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        c | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
        d | Real(0.0,1.0, Uniform) | Int(0,100, Uniform);
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_single_to_mixed_first_hole {
    use tantale::core::domain::{Bool, Cat, Nat, Real};
    use tantale::core::distribution::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                                           ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                      ;
    );
}

pub mod sp_single_to_mixed_second_hole {
    use tantale::core::domain::{Bool, Cat, Int, Real};
    use tantale::core::distribution::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                       ;
        b | Real(0.0,1.0, Uniform) |                                           ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                      ;
    );
}

pub mod sp_single_to_mixed_last_hole {
    use tantale::core::domain::{Cat, Int, Nat, Real};
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | Int(0,100, Uniform)                       ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d | Real(0.0,1.0, Uniform) |                                           ;
    );
}

pub mod sp_single_to_mixed_two_hole {
    use tantale::core::domain::{Cat, Nat, Real};
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) |                                           ;
        b | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
        c | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d | Real(0.0,1.0, Uniform) |                                           ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_mixed_to_single_first_hole {
    use tantale::core::domain::{Bool, Cat, Nat, Real};
    use tantale::core::distribution::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform)                      |                        ;
        b | Nat(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform)   | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                        | Real(0.0,1.0, Uniform) ;
    );
}

pub mod sp_mixed_to_single_second_hole {
    use tantale::core::domain::{Bool, Cat, Int, Real};
    use tantale::core::distribution::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        b | Real(0.0,1.0, Uniform)                      |                        ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform)   | Real(0.0,1.0, Uniform) ;
        d | Bool(Bernoulli(0.5))                        | Real(0.0,1.0, Uniform) ;
    );
}

pub mod sp_mixed_to_single_last_hole {
    use tantale::core::domain::{Cat, Int, Nat, Real};
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        b | Nat(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform)   | Real(0.0,1.0, Uniform) ;
        d | Real(0.0,1.0, Uniform)                      |                        ;
    );
}

pub mod sp_mixed_to_single_two_holes {
    use tantale::core::domain::{Cat, Nat, Real};
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform)                      |                        ;
        b | Nat(0,100, Uniform)                         | Real(0.0,1.0, Uniform) ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform)   | Real(0.0,1.0, Uniform) ;
        d | Real(0.0,1.0, Uniform)                      |                        ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_repeats {

    use tantale::core::domain::{Bool, Cat, Nat, Real};
    use tantale::core::distribution::{Bernoulli, Uniform};
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 6;
    pub const A_INDEX: (usize, usize) = (0, 3);
    pub const B_INDEX: usize = 3;
    pub const C_INDEX: usize = 4;
    pub const D_INDEX: usize = 5;

    hpo!(
        a{3}| Real(0.0,1.0, Uniform) |                                           ;
        b   | Real(0.0,1.0, Uniform) | Nat(0,100, Uniform)                       ;
        c   | Real(0.0,1.0, Uniform) | Cat(["relu", "tanh", "sigmoid"], Uniform) ;
        d   | Real(0.0,1.0, Uniform) | Bool(Bernoulli(0.5))                      ;
    );
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

pub mod sp_only_real {
    use tantale::core::domain::Real;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.0,1.0, Uniform) | ;
        b | Real(0.0,1.0, Uniform) | ;
        c | Real(0.0,1.0, Uniform) | ;
        d | Real(0.0,1.0, Uniform) | ;
    );
}

pub mod sp_only_int {
    use tantale::core::domain::Int;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(-100,100,Uniform) | ;
        b | Int(-100,100,Uniform) | ;
        c | Int(-100,100,Uniform) | ;
        d | Int(-100,100,Uniform) | ;
    );
}
pub mod sp_only_nat {
    use tantale::core::domain::Nat;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Nat(0,100, Uniform) | ;
        b | Nat(0,100, Uniform) | ;
        c | Nat(0,100, Uniform) | ;
        d | Nat(0,100, Uniform) | ;
    );
}

pub mod sp_only_unit {
    use tantale::core::domain::Unit;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Unit(Uniform) | ;
        b | Unit(Uniform) | ;
        c | Unit(Uniform) | ;
        d | Unit(Uniform) | ;
    );
}
pub mod sp_only_bool {
    use tantale::core::domain::Bool;
    use tantale::core::distribution::Bernoulli;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Bool(Bernoulli(0.5)) | ;
        b | Bool(Bernoulli(0.5)) | ;
        c | Bool(Bernoulli(0.5)) | ;
        d | Bool(Bernoulli(0.5)) | ;
    );
}

pub mod sp_only_cat {
    use tantale::core::domain::Cat;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
        b | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
        c | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
        d | Cat(["relu", "tanh", "sigmoid"], Uniform) | ;
    );
}

pub mod sp_only_real_log {
    use tantale::core::domain::Real;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Real(0.1,1.0, Uniform) | Log ;
        b | Real(0.0,1.0, Uniform) |     ;
        c | Real(0.0,1.0, Uniform) |     ;
        d | Real(0.0,1.0, Uniform) |     ;
    );
}

pub mod sp_only_int_log {
    use tantale::core::domain::Int;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Int(-100,100,Uniform) |     ;
        b | Int(1,100,Uniform)    | Log ;
        c | Int(-100,100,Uniform) |     ;
        d | Int(-100,100,Uniform) |     ;
    );
}
pub mod sp_only_nat_log {
    use tantale::core::domain::Nat;
    use tantale::core::distribution::Uniform;
    use tantale::macros::hpo;

    pub const SP_SIZE: usize = 4;
    pub const A_INDEX: usize = 0;
    pub const B_INDEX: usize = 1;
    pub const C_INDEX: usize = 2;
    pub const D_INDEX: usize = 3;

    hpo!(
        a | Nat(0,100, Uniform) |      ;
        b | Nat(0,100, Uniform) |      ;
        c | Nat(0,100, Uniform) |      ;
        d | Nat(1,100, Uniform) | Log  ;
    );
}