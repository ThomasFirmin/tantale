use tantale::core::{
    experiment::{mpi::tools, ThrBatchEvaluator},
    stop::Calls,
    EmptyInfo, Objective, Searchspace, SingleCodomain, Solution,
};
use tantale_algos::{RSInfo, RandomSearch};
use tantale_core::{
    experiment::{mpi::tools::MPIProcess, DistEvaluate},
    solution::Batch,
    Sp,
};

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
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
    pub struct Point {
        pub x: f64,
        pub y: f64,
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

    let func = sp_evaluator::example;
    let cod = SingleCodomain::new(|o: &OutEvaluator| o.obj);
    let obj = Arc::new(Objective::new(cod, func));

    if !tools::launch_worker(&proc, &obj) {
        eprintln!("TEEESST");
        let sp = sp_evaluator::get_searchspace();
        let sinfo = std::sync::Arc::new(EmptyInfo {});
        let info = std::sync::Arc::new(RSInfo { iteration: 0 });
        let stop = Arc::new(Mutex::new(Calls::new(50)));

        let mut rng = rand::rng();
        let sobj = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
        let sopt = sp.vec_onto_obj(sobj.clone());
        let batch = Batch::new(sobj, sopt, info.clone());
        let mut eval = ThrBatchEvaluator::new(batch);

        let (batch_raw, batch_comp) =
            <ThrBatchEvaluator<_, _, _, _, _, _> as DistEvaluate<
                RandomSearch,
                Calls,
                _,
                _,
                OutEvaluator,
                _,
                Sp<_, _>,
            >>::evaluate(&mut eval, &proc, obj.clone(), stop.clone());

        let mut hcobj = HashMap::new();
        let mut hsobj = HashMap::new();
        let mut hcopt = HashMap::new();
        let mut hsopt = HashMap::new();
        let mut hlink = HashMap::new();

        batch_comp
            .cobj
            .iter()
            .zip(eval.batch.sobj.iter())
            .zip(batch_raw.robj.iter())
            .zip(batch_comp.copt.iter())
            .zip(eval.batch.sopt.iter())
            .for_each(|((((c, s), l), x), y)| {
                let cid = c.get_id().id;
                let sid = s.get_id().id;
                let xid = x.get_id().id;
                let yid = y.get_id().id;
                let lid = l.sol.get_id().id;
                hcobj.insert(cid, c.clone());
                hsobj.insert(sid, s.clone());
                hcopt.insert(xid, x.clone());
                hsopt.insert(yid, y.clone());
                hlink.insert(lid, l);
            });

        assert_eq!(
            batch_comp.cobj.len(),
            20,
            "Number of solutions is wrong for cobj"
        );
        assert_eq!(
            eval.batch.sobj.len(),
            20,
            "Number of solutions is wrong for sobj"
        );
        assert_eq!(
            batch_comp.copt.len(),
            20,
            "Number of solutions is wrong for copt"
        );
        assert_eq!(
            eval.batch.sopt.len(),
            20,
            "Number of solutions is wrong for sopt"
        );
        assert_eq!(
            batch_raw.robj.len(),
            20,
            "Number of solutions is wrong for robj"
        );
        assert_eq!(
            batch_raw.ropt.len(),
            20,
            "Number of solutions is wrong for ropt"
        );

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
        assert_eq!(
            hlink.len(),
            20,
            "Some IDs might be duplicated. Number of solutions is wrong for hlink"
        );

        assert_eq!(
            stop.lock().unwrap().calls(),
            20,
            "Number of calls is wrong."
        );

        assert!(
            batch_comp.cobj.iter().all(|sol| {
                let id = sol.get_id().id;
                let c = &hcobj.get(&id).unwrap();
                let s = &hsobj.get(&id).unwrap();
                let l = &hlink.get(&id).unwrap();
                Arc::ptr_eq(&c.get_sol(), s)
                    && Arc::ptr_eq(&l.sol, s)
                    && Arc::ptr_eq(&l.sol, &c.get_sol())
            }),
            "Computed, Partial and Linked do not point to the same Obj solution."
        );

        assert!(
            batch_comp.copt.iter().all(|sol| {
                let id = sol.get_id().id;
                let c = &hcopt.get(&id).unwrap();
                let s = &hsopt.get(&id).unwrap();
                Arc::ptr_eq(&c.get_sol(), s)
            }),
            "Computed and Partial do not point to the same Opt solution."
        );

        assert!(batch_comp.cobj.iter().all(
            |sol|
            {
                let id = sol.get_id().id;
                let c = &hcobj.get(&id).unwrap();
                let s = &hcopt.get(&id).unwrap();
                let l = &hlink.get(&id).unwrap();
                Arc::ptr_eq(&c.get_y(), &s.get_y()) &&
                c.get_y().value == s.get_y().value &&
                c.get_y().value == l.out.obj &&
                s.get_y().value == l.out.obj
            }
            ),
            "Computed Obj, Computed Opt, and Linked do not point to the same codomain, or codomains are not equal."
        );
    }
}
