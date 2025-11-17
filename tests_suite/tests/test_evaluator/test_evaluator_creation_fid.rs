use tantale_algos::RSInfo;
use tantale_core::{
    BaseDom, EmptyInfo, FidBasePartial, SId, Searchspace, SingleCodomain, Solution, Sp, Stepped, experiment::{FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, ThrEvaluate}, solution::{Batch, partial::FidelityPartial}, stop::Calls
};

use super::init_func::{sp_evaluator_fid, FnState};
use crate::init_func::FidOutEvaluator;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[test]
fn test_seq_evaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);

    let mut rng = rand::rng();
    let sobj: Vec<FidBasePartial<_, _, _>> = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
    let sopt: Vec<FidBasePartial<_, _, _>> = sp.vec_onto_obj(&sobj);
    let sobj_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = sobj
        .iter()
        .map(|s: &FidBasePartial<_, _, _>| (s.get_id(), s.x.clone()))
        .collect();
    let sopt_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = sopt
        .iter()
        .map(|s: &FidBasePartial<_, _, _>| (s.get_id(), s.x.clone()))
        .collect();
    let batch = Batch::new(sobj, sopt, info.clone());
    let mut eval = FidBatchEvaluator::new(batch);

    let (braw, bcomp) = <FidBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > as MonoEvaluate<
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        Calls,
        SingleCodomain<FidOutEvaluator>,
        FidOutEvaluator,
        Sp<BaseDom, BaseDom>,
        Stepped<BaseDom, FidOutEvaluator, FnState>,
        Batch<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, BaseDom, EmptyInfo, RSInfo>,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);

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
    assert_eq!(stop.calls(), 0, "Number of calls is wrong.");

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

    let (vobj,vopt) = bcomp.into_iter().map(
        |(sj,st)|
        {
            let mut obj = sj.sol;
            let mut opt = st.sol;
            obj.discard(&mut opt);
            (obj,opt)
        }
    ).collect();
    let batch = Batch::new(vobj, vopt, info.clone());
    let mut eval = FidBatchEvaluator::new(batch);

    <FidBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > as MonoEvaluate<
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        Calls,
        SingleCodomain<FidOutEvaluator>,
        FidOutEvaluator,
        Sp<BaseDom, BaseDom>,
        Stepped<BaseDom, FidOutEvaluator, FnState>,
        Batch<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, BaseDom, EmptyInfo, RSInfo>,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);

    assert_eq!(
        stop.calls(),
        20,
        "Number of calls is wrong."
    );

}

#[test]
fn test_seq_par_evaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = Arc::new(SingleCodomain::new(|o: &FidOutEvaluator| o.obj));
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj: Vec<FidBasePartial<_, _, _>> = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
    let sopt: Vec<FidBasePartial<_, _, _>> = sp.vec_onto_obj(&sobj);
    let sobj_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = sobj
        .iter()
        .map(|s: &FidBasePartial<_, _, _>| (s.get_id(), s.x.clone()))
        .collect();
    let sopt_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = sopt
        .iter()
        .map(|s: &FidBasePartial<_, _, _>| (s.get_id(), s.x.clone()))
        .collect();
    let batch = Batch::new(sobj, sopt, info.clone());
    let mut eval = FidThrBatchEvaluator::new(batch);

    let (braw, bcomp) = <FidThrBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > as ThrEvaluate<
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        Calls,
        SingleCodomain<FidOutEvaluator>,
        FidOutEvaluator,
        Sp<BaseDom, BaseDom>,
        Stepped<BaseDom, FidOutEvaluator, FnState>,
        Batch<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, BaseDom, EmptyInfo, RSInfo>,
    >>::evaluate(&mut eval, obj.clone(), cod.clone(), stop.clone());

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
    assert_eq!(
        stop.lock().unwrap().calls(),
        0,
        "Number of calls is wrong."
    );

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

    let (vobj,vopt) = bcomp.into_iter().map(
        |(sj,st)|
        {
            let mut obj = sj.sol;
            let mut opt = st.sol;
            obj.discard(&mut opt);
            (obj,opt)
        }
    ).collect();
    let batch = Batch::new(vobj, vopt, info.clone());
    let mut eval = FidThrBatchEvaluator::new(batch);

    <FidThrBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > as ThrEvaluate<
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        Calls,
        SingleCodomain<FidOutEvaluator>,
        FidOutEvaluator,
        Sp<BaseDom, BaseDom>,
        Stepped<BaseDom, FidOutEvaluator, FnState>,
        Batch<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, BaseDom, EmptyInfo, RSInfo>,
    >>::evaluate(&mut eval, obj.clone(), cod.clone(), stop.clone());

    assert_eq!(
        stop.lock().unwrap().calls(),
        20,
        "Number of calls is wrong."
    );

}
