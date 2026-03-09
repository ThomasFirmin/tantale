use mpi::traits::Communicator;
use tantale::core::{
    EmptyInfo, Searchspace, SingleCodomain, experiment::BatchEvaluator, stop::Calls,
};
use tantale::algos::{BatchRandomSearch, RSInfo};
use tantale::core::{
    BaseSol, Codomain, Mixed, MixedTypeDom, Objective, SId, Sp,
    domain::{NoDomain, TypeDom},
    experiment::{
        DistEvaluate, OutBatchEvaluate,
        mpi::{
            utils::{MPIProcess, SendRec, XMessage},
            worker::{BaseWorker, Worker},
        },
    },
    solution::{Batch, HasId, Lone, SolutionShape},
};

use std::{collections::HashMap, sync::Arc};

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
        use tantale::core::sampler::{Bernoulli, Uniform};

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

type BBatch =
    Batch<SId, EmptyInfo, RSInfo, Lone<BaseSol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>>;

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
        let info = std::sync::Arc::new(RSInfo { iteration: 0 });
        let sinfo = std::sync::Arc::new(EmptyInfo {});
        let mut stop = Calls::new(50);
        let mut acc = SingleCodomain::new_accumulator();

        let mut rng = rand::rng();
        let sobj = <Sp<Mixed, NoDomain> as Searchspace<
            BaseSol<SId, Mixed, EmptyInfo>,
            SId,
            EmptyInfo,
        >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
        let pair = sp.vec_onto_obj(sobj);
        let sobj_bis: Vec<(SId, Arc<[tantale::core::MixedTypeDom]>)> = pair
            .iter()
            .map(|s| (s.id(), s.get_sobj().x.clone()))
            .collect();
        let sopt_bis: Vec<(SId, Arc<[tantale::core::MixedTypeDom]>)> = pair
            .iter()
            .map(|s| (s.id(), s.get_sopt().x.clone()))
            .collect();
        let batch: BBatch = Batch::new(pair, info.clone());
        let mut eval = BatchEvaluator::new(batch);

        let (bcomp, braw) =
            <BatchEvaluator<
                SId,
                EmptyInfo,
                RSInfo,
                Lone<BaseSol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
            > as DistEvaluate<
                BaseSol<SId, Mixed, EmptyInfo>,
                SId,
                BatchRandomSearch,
                Sp<Mixed, NoDomain>,
                OutEvaluator,
                Calls,
                Objective<Arc<[MixedTypeDom]>, OutEvaluator>,
                _,
                OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, BaseSol<SId, _, _>, _, _>,
            >>::evaluate(&mut eval, &mut sendrec, &obj, &cod, &mut stop, &mut acc);

        let mut hcobj = HashMap::new();
        let mut hsobj: HashMap<SId, Arc<[tantale::core::MixedTypeDom]>> = HashMap::new();
        let mut hcopt = HashMap::new();
        let mut hsopt: HashMap<SId, Arc<[tantale::core::MixedTypeDom]>> = HashMap::new();

        let compiter = (&bcomp).into_iter();

        sobj_bis
            .into_iter()
            .zip(sopt_bis)
            .zip(compiter)
            .for_each(|((sobj, sopt), pair)| {
                hsobj.insert(sobj.0, sobj.1);
                hsopt.insert(sopt.0, sopt.1);
                hcobj.insert(pair.get_sobj().id(), pair.get_sobj());
                hcopt.insert(pair.get_sopt().id(), pair.get_sopt());
            });

        assert_eq!(bcomp.pairs.len(), 20, "Number of shapes is wrong.");
        assert_eq!(bcomp.size(), 20, "Size of Computed batch is wrong");
        assert_eq!(braw.size(), 20, "Size of Out batch is wrong");

        assert_eq!(
            hcobj.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hcobj"
        );
        assert_eq!(
            hsobj.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hsobj"
        );
        assert_eq!(
            hcopt.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hcopt"
        );
        assert_eq!(
            hsopt.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hsopt"
        );
        assert_eq!(stop.calls(), 20, "Number of calls is wrong.");

        (&bcomp).into_iter().for_each(|pair| {
            let id = pair.id();
            let cobj = hcobj.get(&id).unwrap();
            let copt = hcopt.get(&id).unwrap();
            let sobj = hsobj.get(&id).unwrap();
            let sopt = hsopt.get(&id).unwrap();

            assert!(
                Arc::ptr_eq(&pair.get_sobj().sol.x, sobj),
                "Obj Partial do not point to the same solutions."
            );
            assert!(
                Arc::ptr_eq(&pair.get_sobj().sol.x, &cobj.sol.x),
                "Obj Computed do not point to the same solutions."
            );
            assert!(
                Arc::ptr_eq(&pair.get_sopt().sol.x, sopt),
                "Opt Partial do not point to the same solutions."
            );
            assert!(
                Arc::ptr_eq(&pair.get_sopt().sol.x, &copt.sol.x),
                "Opt Computed do not point to the same solutions."
            );
        });

        proc.world.abort(42);
    }
}
