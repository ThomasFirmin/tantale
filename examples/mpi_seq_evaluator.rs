use mpi::traits::Communicator;
use tantale::core::{EmptyInfo, Searchspace, SingleCodomain, stop::Calls};
use tantale_algos::RandomSearch;
use tantale_core::{
    BaseSol, Mixed, MixedTypeDom, Objective, SId, Sp,
    domain::{NoDomain, TypeDom},
    experiment::{
        DistEvaluate, OutShapeEvaluate,
        mpi::{
            utils::{MPIProcess, SendRec, XMessage},
            worker::{BaseWorker, Worker},
        },
        sequential::seqevaluator::DistSeqEvaluator,
    },
    solution::{HasId, Lone, SolutionShape},
};

use std::sync::Arc;

mod init_func {
    use serde::{Deserialize, Serialize};
    use tantale::macros::Outcome;

    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct OutEvaluator {
        pub obj: f64,
    }

    impl PartialEq for OutEvaluator {
        fn eq(&self, other: &Self) -> bool {
            self.obj == other.obj
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
        use super::{Neuron, OutEvaluator, int_plus_nat, plus_one_int};
        use tantale::core::{Bool, Cat, Int, Nat, Real};
        use tantale::macros::objective;
        use tantale_core::sampler::{Bernoulli, Uniform};

        objective!(
            pub fn example() -> OutEvaluator {
                let _a = [! a | Int(0,100, Uniform) | !];
                let _b = [! b | Nat(0,100, Uniform) | !];
                let _c = [! c | Cat(["relu", "tanh", "sigmoid"],Uniform) | !];
                let _d = [! d | Bool(Bernoulli(0.5)) | !];

                let _e = plus_one_int([! e | Int(0,100, Uniform) | !]);
                let _f = int_plus_nat([! f | Int(0,100, Uniform) | !], [! g | Nat(0,100, Uniform) | !]);

                let _layer = Neuron{
                    number: [! h | Int(0,100,Uniform) | !],
                    activation: [! i | Cat(["relu", "tanh", "sigmoid"],Uniform) | !],
                };

                let _k = [! k_{4} | Nat(0,100,Uniform) | !];

                OutEvaluator{
                    obj: [! j | Real(1000.0,2000.0,Uniform) | !]
                }
            }
        );
    }
}

use init_func::{OutEvaluator, sp_evaluator};

fn main() {
    eprintln!("INFO : Running test_seq_evaluator.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let proc = MPIProcess::new();

    if proc.rank != 0 {
        let wkr = BaseWorker::new(sp_evaluator::get_function(), &proc);
        <BaseWorker<'_, Arc<[TypeDom<sp_evaluator::ObjType>]>, OutEvaluator> as Worker<SId>>::run(
            wkr,
        );
    } else {
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<'_, XMessage<SId, _>, _, _, _, _, _>::new(config, &proc);

        let sp = sp_evaluator::get_searchspace();
        let func = sp_evaluator::example;
        let cod = SingleCodomain::new(|o: &OutEvaluator| o.obj);
        let obj = Arc::new(Objective::new(func));
        let sinfo = std::sync::Arc::new(EmptyInfo {});
        let mut stop = Calls::new(50);

        let mut rng = rand::rng();
        let pair = <Sp<Mixed, NoDomain> as Searchspace<
            BaseSol<SId, Mixed, EmptyInfo>,
            SId,
            EmptyInfo,
        >>::vec_sample_pair(&sp, &mut rng, 4, sinfo.clone());
        let sobj_bis: Vec<(SId, Arc<[tantale_core::MixedTypeDom]>)> = pair
            .iter()
            .map(|s| (s.get_id(), s.get_sobj().x.clone()))
            .collect();
        let sopt_bis: Vec<(SId, Arc<[tantale_core::MixedTypeDom]>)> = pair
            .iter()
            .map(|s| (s.get_id(), s.get_sopt().x.clone()))
            .collect();
        let mut eval = DistSeqEvaluator::new(pair);

        let out = <DistSeqEvaluator<
            SId,
            EmptyInfo,
            Lone<BaseSol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        > as DistEvaluate<
            BaseSol<SId, Mixed, EmptyInfo>,
            SId,
            RandomSearch,
            Sp<Mixed, NoDomain>,
            OutEvaluator,
            Calls,
            Objective<Arc<[MixedTypeDom]>, OutEvaluator>,
            _,
            Option<
                OutShapeEvaluate<
                    SId,
                    EmptyInfo,
                    Sp<Mixed, NoDomain>,
                    BaseSol<SId, Mixed, EmptyInfo>,
                    SingleCodomain<OutEvaluator>,
                    OutEvaluator,
                >,
            >,
        >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop);
        let (comp, raw) = out.unwrap();

        assert_eq!(stop.calls(), 1, "Number of calls is wrong.");
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

        proc.world.abort(42);
    }
}
