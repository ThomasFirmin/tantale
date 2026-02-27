use mpi::traits::Communicator;
use tantale::core::{EmptyInfo, Searchspace, SingleCodomain, stop::Calls};
use tantale_algos::RandomSearch;
use tantale_core::{
    FidelitySol, Mixed, MixedTypeDom, SId, Sp, Stepped,
    checkpointer::NoCheck,
    domain::{NoDomain, TypeDom},
    experiment::{
        DistEvaluate, DistOutShapeEvaluate,
        mpi::{
            utils::{FXMessage, MPIProcess, SendRec},
            worker::{FidWorker, Worker},
        },
        sequential::seqfidevaluator::FidDistSeqEvaluator,
    },
    objective::Step,
    solution::{HasId, HasStep, IntoComputed, Lone, SolutionShape},
};

use std::sync::Arc;

mod init_func {
    use serde::{Deserialize, Serialize};
    use tantale::core::{Step, objective::outcome::FuncState};
    use tantale::macros::Outcome;

    #[derive(Serialize, Deserialize)]
    pub struct FnState {
        pub state: isize,
    }
    impl FuncState for FnState {}

    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct FidOutEvaluator {
        pub obj: f64,
        pub fid: Step,
    }

    impl PartialEq for FidOutEvaluator {
        fn eq(&self, other: &Self) -> bool {
            self.obj == other.obj && self.fid == other.fid
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct Neuron {
        pub number: i64,
        pub activation: String,
    }

    pub fn plus_one_int(x: i64) -> (i64, i64) {
        (x, x + 1)
    }

    pub fn int_plus_nat(x: i64, y: u64) -> (i64, u64, i64) {
        (x, y, x + (y as i64))
    }

    pub mod sp_evaluator {
        use super::{FidOutEvaluator, FnState, Neuron, int_plus_nat, plus_one_int};
        use tantale_core::{
            Bool, Cat, Int, Nat, Real,
            objective::Step,
            sampler::{Bernoulli, Uniform},
        };
        use tantale_macros::objective;

        objective!(
            pub fn example() -> (FidOutEvaluator, FnState) {
                let _rank = [! MPI_RANK !];
                let _size = [! MPI_SIZE !];

                let _a = [! a | Int(0,100, Uniform) | !];
                let _b = [! b | Nat(0,100, Uniform) | !];
                let _c = [! c | Cat(["relu", "tanh", "sigmoid"], Uniform) | !];
                let _d = [! d | Bool(Bernoulli(0.5)) | !];

                let _e = plus_one_int([! e | Int(0,100,Uniform) | !]);
                let _f = int_plus_nat([! f | Int(0,100,Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

                let _layer = Neuron{
                    number: [! h | Int(0,100, Uniform) | !],
                    activation: [! i | Cat(["relu", "tanh", "sigmoid"], Uniform) | !],
                };

                let _k = [! k_{4} | Nat(0,100, Uniform) | !];

                let mut state = match [! STATE !]{
                    Some(s) => s,
                    None => FnState { state: 0 },
                };
                state.state += 1;
                println!("STATE : {}",state.state);
                let evalstate = if state.state == 5 {Step::Evaluated.into()} else{Step::Partially(state.state).into()};
                (
                    FidOutEvaluator{
                        obj: [! j | Real(1000.0,2000.0, Uniform) | !],
                        fid: evalstate,
                    },
                    state
                )

            }
        );
    }
}

use init_func::{FidOutEvaluator, FnState, sp_evaluator};

fn main() {
    eprintln!("INFO : Running test_seq_evaluator.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let proc = MPIProcess::new();

    if proc.rank != 0 {
        let wkr = FidWorker::new(sp_evaluator::get_function(), None, &proc);
        <FidWorker<
            '_,
            SId,
            Arc<[TypeDom<sp_evaluator::ObjType>]>,
            FidOutEvaluator,
            FnState,
            NoCheck,
        > as Worker<SId>>::run(wkr);
    } else {
        // Define send/rec utilitaries and parameters
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<'_, FXMessage<SId, _>, _, _, _, _, _>::new(config, &proc);

        let sp = sp_evaluator::get_searchspace();
        let func = sp_evaluator::example;
        let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
        let obj = Arc::new(Stepped::new(func));
        let sinfo = std::sync::Arc::new(EmptyInfo {});
        let mut stop = Calls::new(50);

        let mut rng = rand::rng();
        let pair = <Sp<Mixed, NoDomain> as Searchspace<
            FidelitySol<SId, Mixed, EmptyInfo>,
            SId,
            EmptyInfo,
        >>::vec_sample_pair(&sp, &mut rng, 4, sinfo.clone());
        let sobj_bis: Vec<(SId, Arc<[tantale_core::MixedTypeDom]>)> = pair
            .iter()
            .map(|s| (s.id(), s.get_sobj().x.clone()))
            .collect();
        let sopt_bis: Vec<(SId, Arc<[tantale_core::MixedTypeDom]>)> = pair
            .iter()
            .map(|s| (s.id(), s.get_sopt().x.clone()))
            .collect();
        let mut eval = FidDistSeqEvaluator::new(pair, proc.size as usize);

        let out = <FidDistSeqEvaluator<
            SId,
            EmptyInfo,
            Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        > as DistEvaluate<
            FidelitySol<SId, Mixed, EmptyInfo>,
            SId,
            RandomSearch,
            Sp<Mixed, NoDomain>,
            FidOutEvaluator,
            Calls,
            Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
            _,
            Option<
                DistOutShapeEvaluate<
                    SId,
                    EmptyInfo,
                    Sp<Mixed, NoDomain>,
                    FidelitySol<SId, Mixed, EmptyInfo>,
                    SingleCodomain<FidOutEvaluator>,
                    FidOutEvaluator,
                >,
            >,
        >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);
        let (_, (comp, raw)) = out.unwrap();

        assert_eq!(stop.calls(), 0, "Number of calls is wrong.");
        let comobjid = sobj_bis
            .iter()
            .find(|(id, _)| &comp.get_id() == id)
            .unwrap();
        let comoptid = sopt_bis
            .iter()
            .find(|(id, _)| &comp.get_id() == id)
            .unwrap();
        let rawobjid = sobj_bis.iter().find(|(id, _)| &raw.0 == id);
        let rawoptid = sopt_bis.iter().find(|(id, _)| &raw.0 == id);

        assert!(
            Arc::ptr_eq(&comp.get_sobj().sol.x, &comobjid.1),
            "Obj Partial and Computed do not point to the same solutions."
        );
        assert!(
            Arc::ptr_eq(&comp.get_sopt().sol.x, &comoptid.1),
            "Opt Partial and Computed do not point to the same solutions."
        );
        assert!(
            rawobjid.is_some(),
            "Obj Id Raw and Partial do not point to the same solutions."
        );
        assert!(
            rawoptid.is_some(),
            "Opt Id Raw and Partial do not point to the same solutions."
        );

        let mut step = comp.step();
        eval.update(IntoComputed::extract(comp).0);

        while step != Step::Evaluated {
            let out = <FidDistSeqEvaluator<
                SId,
                EmptyInfo,
                Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
            > as DistEvaluate<
                FidelitySol<SId, Mixed, EmptyInfo>,
                SId,
                RandomSearch,
                Sp<Mixed, NoDomain>,
                FidOutEvaluator,
                Calls,
                Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
                _,
                Option<
                    DistOutShapeEvaluate<
                        SId,
                        EmptyInfo,
                        Sp<Mixed, NoDomain>,
                        FidelitySol<SId, Mixed, EmptyInfo>,
                        SingleCodomain<FidOutEvaluator>,
                        FidOutEvaluator,
                    >,
                >,
            >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);
            let (_, (comp, _)) = out.unwrap();
            step = comp.step();
            eval.update(IntoComputed::extract(comp).0);
        }

        assert_eq!(stop.calls(), 1, "Number of calls is wrong.");
        proc.world.abort(42)
    }
}
