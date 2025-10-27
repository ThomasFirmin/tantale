use tantale_algos::{RSInfo, RandomSearch};
use tantale_core::{
    experiment::{BatchEvaluator, MonoEvaluate, ThrEvaluate},
    solution::Batch,
    stop::Calls,
    BasePartial, EmptyInfo, Objective, SId, Searchspace, SingleCodomain, Solution, Sp,
};

use super::init_func::sp_evaluator;
use crate::init_func::OutEvaluator;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[test]
fn test_seq_evaluator() {
    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = SingleCodomain::new(|o: &OutEvaluator| o.obj);
    let obj = Arc::new(Objective::new(cod, func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
    let sopt = sp.vec_onto_obj(sobj.clone());
    let batch = Batch::new(sobj, sopt, info.clone());
    let mut eval = BatchEvaluator::new(batch);

    let (braw, bcomp) = <BatchEvaluator<BasePartial<SId, _, _>, _, _, _, _, _> as MonoEvaluate<
        RandomSearch,
        Calls,
        sp_evaluator::ObjType,
        sp_evaluator::OptType,
        OutEvaluator,
        _,
        Sp<_, _>,
    >>::evaluate(&mut eval, obj.clone(), stop.clone());

    let mut hcobj = HashMap::new();
    let mut hsobj = HashMap::new();
    let mut hcopt = HashMap::new();
    let mut hsopt = HashMap::new();
    let mut hlink = HashMap::new();

    bcomp
        .cobj
        .iter()
        .zip(eval.batch.sobj.iter())
        .zip(braw.robj.iter())
        .zip(bcomp.copt.iter())
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
    assert_eq!(braw.robj.len(), 20, "Number of solutions is wrong for robj");
    assert_eq!(braw.ropt.len(), 20, "Number of solutions is wrong for ropt");
    assert_eq!(braw.size(), 20, "Size of RawSol batch is wrong");
    assert_eq!(
        eval.batch.sobj.len(),
        20,
        "Number of solutions is wrong for sobj"
    );
    assert_eq!(
        eval.batch.sopt.len(),
        20,
        "Number of solutions is wrong for sopt"
    );
    assert_eq!(eval.batch.size(), 20, "Size of Partial batch is wrong");

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
        bcomp.cobj.iter().all(|sol| {
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
        bcomp.copt.iter().all(|sol| {
            let id = sol.get_id().id;
            let c = &hcopt.get(&id).unwrap();
            let s = &hsopt.get(&id).unwrap();
            Arc::ptr_eq(&c.get_sol(), s)
        }),
        "Computed and Partial do not point to the same Opt solution."
    );

    assert!(bcomp.cobj.iter().all(
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

#[test]
fn test_seq_par_evaluator() {
    use tantale::core::experiment::ThrBatchEvaluator;

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = SingleCodomain::new(|o: &OutEvaluator| o.obj);
    let obj = Arc::new(Objective::new(cod, func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
    let sopt = sp.vec_onto_obj(sobj.clone());
    let batch = Batch::new(sobj, sopt, info.clone());
    let mut eval = ThrBatchEvaluator::new(batch);

    assert_eq!(eval.batch.sopt.len(), 20, "PROBLEM");
    assert_eq!(eval.batch.sobj.len(), 20, "PROBLEMA");

    let (braw, bcomp) = <ThrBatchEvaluator<_, _, _, _, _, _> as ThrEvaluate<
        RandomSearch,
        Calls,
        sp_evaluator::ObjType,
        sp_evaluator::OptType,
        OutEvaluator,
        _,
        Sp<_, _>,
    >>::evaluate(&mut eval, obj.clone(), stop.clone());
    let mut hcobj = HashMap::new();
    let mut hsobj = HashMap::new();
    let mut hcopt = HashMap::new();
    let mut hsopt = HashMap::new();
    let mut hlink = HashMap::new();

    bcomp
        .cobj
        .iter()
        .zip(eval.batch.sobj.iter())
        .zip(braw.robj.iter())
        .zip(bcomp.copt.iter())
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
        eval.batch.sobj.len(),
        20,
        "Number of solutions is wrong for sobj"
    );
    assert_eq!(
        eval.batch.sopt.len(),
        20,
        "Number of solutions is wrong for sopt"
    );
    assert_eq!(eval.batch.size(), 20, "Size of Partial batch is wrong");
    assert_eq!(braw.robj.len(), 20, "Number of solutions is wrong for robj");
    assert_eq!(braw.ropt.len(), 20, "Number of solutions is wrong for ropt");
    assert_eq!(braw.size(), 20, "Size of RawSol batch is wrong");
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
        bcomp.cobj.iter().all(|sol| {
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
        bcomp.copt.iter().all(|sol| {
            let id = sol.get_id().id;
            let c = &hcopt.get(&id).unwrap();
            let s = &hsopt.get(&id).unwrap();
            Arc::ptr_eq(&c.get_sol(), s)
        }),
        "Computed and Partial do not point to the same Opt solution."
    );

    assert!(bcomp.cobj.iter().all(
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
