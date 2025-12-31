use tantale_algos::{RSInfo, RandomSearch};
use tantale_core::{
    BaseDom, FidBasePartial, BaseTypeDom, EmptyInfo, Stepped, SId, Searchspace, SingleCodomain, Sp, domain::NoDomain, experiment::{FidBatchEvaluator, MonoEvaluate, FidThrBatchEvaluator, ThrEvaluate}, solution::{Batch, HasId, Lone, SolutionShape}, stop::Calls
};

use super::init_func::sp_evaluator_fid;
use crate::init_func::{FidOutEvaluator, FnState};

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

type BBatch = Batch<SId, EmptyInfo, RSInfo, Lone<FidBasePartial<SId, BaseDom, EmptyInfo>,SId,BaseDom,EmptyInfo>>;

#[test]
fn test_serde_fidbatchevaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);

    let mut rng = rand::rng();
    let sobj= <Sp<BaseDom,NoDomain> as Searchspace<FidBasePartial<SId,BaseDom,EmptyInfo>, SId, EmptyInfo>>::vec_sample_obj(&sp, Some(&mut rng), 20, sinfo.clone());
    let pair = sp.vec_onto_obj(sobj);
    let sobj_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = pair
        .iter()
        .map(|s| (s.get_id(), s.get_sobj().x.clone()))
        .collect();
    let sopt_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = pair
        .iter()
        .map(|s | (s.get_id(), s.get_sopt().x.clone()))
        .collect();
    let batch: BBatch = Batch::new(pair, info.clone());
    let mut eval = FidBatchEvaluator::new(batch);

    let (bcomp,braw, ) = <
        FidBatchEvaluator<
            SId,EmptyInfo,RSInfo,
            Lone<FidBasePartial<SId,BaseDom,EmptyInfo>,SId,BaseDom,EmptyInfo>,
            FnState
        >
        as MonoEvaluate<
            FidBasePartial<SId,BaseDom,EmptyInfo>,
            SId, RandomSearch, Sp<BaseDom,NoDomain>, FidOutEvaluator, Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator,FnState>
        >
    >::evaluate(&mut eval, &obj, &cod, &mut stop);

    let mut hcobj = HashMap::new();
    let mut hsobj: HashMap<SId, Arc<[tantale_core::BaseTypeDom]>> = HashMap::new();
    let mut hcopt = HashMap::new();
    let mut hsopt: HashMap<SId, Arc<[tantale_core::BaseTypeDom]>> = HashMap::new();

    let compiter = (&bcomp).into_iter();

    sobj_bis
        .into_iter()
        .zip(sopt_bis)
        .zip(compiter)
        .for_each(|((sobj, sopt), pair)| {
            hsobj.insert(sobj.0, sobj.1);
            hsopt.insert(sopt.0, sopt.1);
            hcobj.insert(pair.get_sobj().get_id(), pair.get_sobj());
            hcopt.insert(pair.get_sopt().get_id(), pair.get_sopt());
        });

    assert_eq!(
        bcomp.pairs.len(),
        20,
        "Number of shapes is wrong."
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

    
    (&bcomp).into_iter().for_each(|pair| {
        let id = pair.get_id();
        let cobj = hcobj.get(&id).unwrap();
        let copt = hcopt.get(&id).unwrap();
        let sobj = hsobj.get(&id).unwrap();
        let sopt = hsopt.get(&id).unwrap();

        assert!(Arc::ptr_eq(&pair.get_sobj().sol.x, sobj),"Obj Partial do not point to the same solutions.");
        assert!(Arc::ptr_eq(&pair.get_sobj().sol.x, &cobj.sol.x),"Obj Computed do not point to the same solutions.");
        assert!(Arc::ptr_eq(&pair.get_sopt().sol.x, sopt),"Opt Partial do not point to the same solutions.");
        assert!(Arc::ptr_eq(&pair.get_sopt().sol.x, &copt.sol.x),"Opt Computed do not point to the same solutions.");
    });
}

#[test]
fn test_serde_fidthrbatchevaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = Arc::new(SingleCodomain::new(|o: &FidOutEvaluator| o.obj));
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj= <Sp<BaseDom,NoDomain> as Searchspace<FidBasePartial<SId,BaseDom,EmptyInfo>, SId, EmptyInfo>>::vec_sample_obj(&sp, Some(&mut rng), 20, sinfo.clone());
    let pair = sp.vec_onto_obj(sobj);
    let sobj_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = pair
        .iter()
        .map(|s| (s.get_id(), s.get_sobj().x.clone()))
        .collect();
    let sopt_bis: Vec<(SId, Arc<[tantale_core::BaseTypeDom]>)> = pair
        .iter()
        .map(|s | (s.get_id(), s.get_sopt().x.clone()))
        .collect();
    let batch: BBatch = Batch::new(pair, info.clone());
    let mut eval = FidThrBatchEvaluator::new(batch);

    let (bcomp,braw, ) = <
        FidThrBatchEvaluator<
            SId,EmptyInfo,RSInfo,
            Lone<FidBasePartial<SId,BaseDom,EmptyInfo>,SId,BaseDom,EmptyInfo>,
            FnState
        >
        as ThrEvaluate<
            FidBasePartial<SId,BaseDom,EmptyInfo>,
            SId, RandomSearch, Sp<BaseDom,NoDomain>, FidOutEvaluator, Calls,
            Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator,FnState>
        >
    >::evaluate(&mut eval, obj.clone(), cod.clone(), stop.clone());

    let mut hcobj = HashMap::new();
    let mut hsobj: HashMap<SId, Arc<[tantale_core::BaseTypeDom]>> = HashMap::new();
    let mut hcopt = HashMap::new();
    let mut hsopt: HashMap<SId, Arc<[tantale_core::BaseTypeDom]>> = HashMap::new();

    let compiter = (&bcomp).into_iter();

    sobj_bis
        .into_iter()
        .zip(sopt_bis)
        .zip(compiter)
        .for_each(|((sobj, sopt), pair)| {
            hsobj.insert(sobj.0, sobj.1);
            hsopt.insert(sopt.0, sopt.1);
            hcobj.insert(pair.get_sobj().get_id(), pair.get_sobj());
            hcopt.insert(pair.get_sopt().get_id(), pair.get_sopt());
        });

    assert_eq!(
        bcomp.pairs.len(),
        20,
        "Number of shapes is wrong."
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
    assert_eq!(stop.lock().unwrap().calls(), 20, "Number of calls is wrong.");

    
    (&bcomp).into_iter().for_each(|pair| {
        let id = pair.get_id();
        let cobj = hcobj.get(&id).unwrap();
        let copt = hcopt.get(&id).unwrap();
        let sobj = hsobj.get(&id).unwrap();
        let sopt = hsopt.get(&id).unwrap();

        assert!(Arc::ptr_eq(&pair.get_sobj().sol.x, sobj),"Obj Partial do not point to the same solutions.");
        assert!(Arc::ptr_eq(&pair.get_sobj().sol.x, &cobj.sol.x),"Obj Computed do not point to the same solutions.");
        assert!(Arc::ptr_eq(&pair.get_sopt().sol.x, sopt),"Opt Partial do not point to the same solutions.");
        assert!(Arc::ptr_eq(&pair.get_sopt().sol.x, &copt.sol.x),"Opt Computed do not point to the same solutions.");
    });
}