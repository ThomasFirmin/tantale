use tantale::core::{
    experiment::BatchEvaluator,
    stop::Calls,
    EmptyInfo, Searchspace, SingleCodomain, Solution
};
use tantale_algos::RSInfo;
use tantale_core::{
    BaseDom, BasePartial, SId, Sp, experiment::{DistEvaluate, mpi::{utils::{MPIProcess,SendRec,XMessage}, worker::{BaseWorker, Worker}}}, solution::Batch
};

use std::{
    collections::HashMap,
    sync::Arc,
};

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
        use super::{int_plus_nat, plus_one_int, Neuron, OutEvaluator};
        use tantale::core::{Bool, Cat, Int, Nat, Real};
        use tantale::macros::objective;

        objective!(
            pub fn example() -> OutEvaluator {
                let _a = [! a | Int(0,100) | !];
                let _b = [! b | Nat(0,100) | !];
                let _c = [! c | Cat(&["relu", "tanh", "sigmoid"]) | !];
                let _d = [! d | Bool() | !];

                let _e = plus_one_int([! e | Int(0,100) | !]);
                let _f = int_plus_nat([! f | Int(0,100) | !], [! g | Nat(0,100) | !]);

                let _layer = Neuron{
                    number: [! h | Int(0,100) | !],
                    activation: [! i | Cat(&["relu", "tanh", "sigmoid"]) | !],
                };

                let _k = [! k_{4} | Nat(0,100) | !];

                OutEvaluator{
                    obj: [! j | Real(1000.0,2000.0) | !]
                }
            }
        );
    }
}

use init_func::{sp_evaluator, OutEvaluator};

fn main() {
    eprintln!("INFO : Running test_seq_evaluator.");

    if std::env::var("OMPI_COMM_WORLD_SIZE").is_err() {
        eprintln!("Skipping MPI test (not under mpirun)");
        return;
    }

    let proc = MPIProcess::new();

    if proc.rank != 0 {
        let wkr = BaseWorker::new(sp_evaluator::get_function(), &proc);
        <BaseWorker<'_, BaseDom, OutEvaluator> as Worker<SId, BaseDom>>::run(wkr);
    }
    else{
        let config = bincode::config::standard(); // Bytes encoding config
        let mut sendrec = SendRec::<'_,XMessage<SId,_>,_,_,_,_,_>::new(config, &proc);

        let sp = sp_evaluator::get_searchspace();
        let cod = SingleCodomain::new(|o: &OutEvaluator| o.obj);
        let obj = sp_evaluator::get_function();
        let info = std::sync::Arc::new(RSInfo { iteration: 0 });
        let sinfo = std::sync::Arc::new(EmptyInfo {});
        let mut stop = Calls::new(50);
        
        let mut rng = rand::rng();
        let sobj: Vec<BasePartial<_, _, _>> = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
        let sopt: Vec<BasePartial<_, _, _>> = sp.vec_onto_obj(&sobj);
        let sobj_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = sobj
            .iter()
            .map(|s: &BasePartial<_, _, _>| (s.get_id(), s.x.clone()))
            .collect();
        let sopt_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = sopt
            .iter()
            .map(|s: &BasePartial<_, _, _>| (s.get_id(), s.x.clone()))
            .collect();
        let batch = Batch::new(sobj, sopt, info.clone());
        let mut eval = BatchEvaluator::new(batch);

        let (braw, bcomp) =
            <BatchEvaluator<_, _, _, _, _, _> as DistEvaluate<
                _,
                _,
                _,
                _,
                _,
                _,
                Calls,
                _,
                OutEvaluator,
                Sp<_, _>,
                _,
                _,
                _,
            >>::evaluate(&mut eval, &mut sendrec, &obj, &cod,&mut stop);

        let mut hcobj = HashMap::new();
        let mut hsobj: HashMap<SId, Arc<[tantale_core::BaseTypeDom]>> = HashMap::new();
        let mut hcopt = HashMap::new();
        let mut hsopt: HashMap<SId, Arc<[tantale_core::BaseTypeDom]>> = HashMap::new();

        sobj_bis
            .into_iter()
            .zip(sopt_bis)
            .zip(bcomp.cobj.iter())
            .zip(bcomp.copt.iter())
            .for_each(|(((sobj, sopt), cobj), copt)| {
                hsobj.insert(sobj.0, sobj.1);
                hsopt.insert(sopt.0, sopt.1);
                hcobj.insert(cobj.get_id(), cobj);
                hcopt.insert(copt.get_id(), copt);
            });

        assert_eq!(
            bcomp.cobj.len(),
            20,
            "Number of solutions is wrong for cobj"
        );
        assert_eq!(
            bcomp.copt.len(),
            20,
            "Number of solutions is wrong for copt"
        );
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

        assert!(
            bcomp.cobj.iter().all(|sol| {
                let id = sol.get_id();
                let c = &hcobj.get(&id).unwrap();
                let s = &hsobj.get(&id).unwrap();
                Arc::ptr_eq(&c.get_sol().x, s)
            }),
            "Computed, Partial and Linked do not point to the same Obj solution."
        );

        assert!(
            bcomp.copt.iter().all(|sol| {
                let id = sol.get_id();
                let c = &hcopt.get(&id).unwrap();
                let s = &hsopt.get(&id).unwrap();
                Arc::ptr_eq(&c.get_sol().x, s)
            }),
            "Computed and Partial do not point to the same Opt solution."
        );
    }
}
