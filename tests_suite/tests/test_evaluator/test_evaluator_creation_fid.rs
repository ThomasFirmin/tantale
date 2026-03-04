use tantale_algos::{BatchRandomSearch, RSInfo, random_search::RandomSearch};
use tantale_core::{
    EmptyInfo, FidelitySol, Mixed, MixedTypeDom, SId, Searchspace, SingleCodomain, Sp, Stepped,
    checkpointer::NoFuncStateCheck,
    domain::NoDomain,
    experiment::{
        FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, OutBatchEvaluate, OutShapeEvaluate,
        ThrEvaluate, basics::IdxMapPool, sequential::seqfidevaluator::FidSeqEvaluator,
    },
    solution::{Batch, HasId, IntoComputed, Lone, SolutionShape},
    stop::Calls,
};

use super::init_func::sp_evaluator_fid;
use crate::init_func::{FidOutEvaluator, FnState};

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

type BBatch =
    Batch<SId, EmptyInfo, RSInfo, Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>>;

#[test]
fn test_fidbatchevaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);

    let mut rng = rand::rng();
    let sobj = <Sp<Mixed, NoDomain> as Searchspace<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
    let pair = sp.vec_onto_obj(sobj);
    let sobj_bis: Vec<(SId, Arc<[tantale_core::MixedTypeDom]>)> = pair
        .iter()
        .map(|s| (s.id(), s.get_sobj().x.clone()))
        .collect();
    let sopt_bis: Vec<(SId, Arc<[tantale_core::MixedTypeDom]>)> = pair
        .iter()
        .map(|s| (s.id(), s.get_sopt().x.clone()))
        .collect();
    let batch: BBatch = Batch::new(pair, info.clone());
    let mut eval = FidBatchEvaluator::new(batch, IdxMapPool::new(None));

    let (bcomp, braw) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);

    let mut hcobj = HashMap::new();
    let mut hsobj: HashMap<SId, Arc<[tantale_core::MixedTypeDom]>> = HashMap::new();
    let mut hcopt = HashMap::new();
    let mut hsopt: HashMap<SId, Arc<[tantale_core::MixedTypeDom]>> = HashMap::new();

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
    assert_eq!(stop.calls(), 0, "Number of calls is wrong.");

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

    let pairs: Vec<_> = bcomp
        .into_iter()
        .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
        .collect();
    let batch = Batch::new(pairs, info.clone());
    eval.update(batch);
    let (bcomp, _) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);
    let pairs: Vec<_> = bcomp
        .into_iter()
        .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
        .collect();
    let batch = Batch::new(pairs, info.clone());
    eval.update(batch);
    let (bcomp, _) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);
    let pairs: Vec<_> = bcomp
        .into_iter()
        .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
        .collect();
    let batch = Batch::new(pairs, info.clone());
    eval.update(batch);
    let (bcomp, _) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);
    let pairs: Vec<_> = bcomp
        .into_iter()
        .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
        .collect();
    let batch = Batch::new(pairs, info.clone());
    eval.update(batch);
    let (_, _) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);

    assert_eq!(
        stop.calls(),
        20,
        "Number of calls is wrong after fully evaluated."
    );
}

#[test]
fn test_fidthrbatchevaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = Arc::new(SingleCodomain::new(|o: &FidOutEvaluator| o.obj));
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj = <Sp<Mixed, NoDomain> as Searchspace<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
    let pair = sp.vec_onto_obj(sobj);
    let sobj_bis: Vec<(SId, Arc<[tantale_core::MixedTypeDom]>)> = pair
        .iter()
        .map(|s| (s.id(), s.get_sobj().x.clone()))
        .collect();
    let sopt_bis: Vec<(SId, Arc<[tantale_core::MixedTypeDom]>)> = pair
        .iter()
        .map(|s| (s.id(), s.get_sopt().x.clone()))
        .collect();
    let batch: BBatch = Batch::new(pair, info.clone());
    let mut eval = FidThrBatchEvaluator::new(batch, IdxMapPool::new(None));

    let (bcomp, braw) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, obj.clone(), cod.clone(), stop.clone());

    let mut hcobj = HashMap::new();
    let mut hsobj: HashMap<SId, Arc<[tantale_core::MixedTypeDom]>> = HashMap::new();
    let mut hcopt = HashMap::new();
    let mut hsopt: HashMap<SId, Arc<[tantale_core::MixedTypeDom]>> = HashMap::new();

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
    assert_eq!(stop.lock().unwrap().calls(), 0, "Number of calls is wrong.");

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

    let pairs: Vec<_> = bcomp
        .into_iter()
        .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
        .collect();
    let batch = Batch::new(pairs, info.clone());
    eval.update(batch);
    let (bcomp, _) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, obj.clone(), cod.clone(), stop.clone());
    let pairs: Vec<_> = bcomp
        .into_iter()
        .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
        .collect();
    let batch = Batch::new(pairs, info.clone());
    eval.update(batch);
    let (bcomp, _) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, obj.clone(), cod.clone(), stop.clone());
    let pairs: Vec<_> = bcomp
        .into_iter()
        .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
        .collect();
    let batch = Batch::new(pairs, info.clone());
    eval.update(batch);
    let (bcomp, _) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, obj.clone(), cod.clone(), stop.clone());
    let pairs: Vec<_> = bcomp
        .into_iter()
        .map(|p| <Lone<_, _, _, _> as IntoComputed>::extract(p).0)
        .collect();
    let batch = Batch::new(pairs, info.clone());
    eval.update(batch);
    let (_, _) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut eval, obj.clone(), cod.clone(), stop.clone());

    assert_eq!(
        stop.lock().unwrap().calls(),
        20,
        "Number of calls is wrong after fully evaluated."
    );
}

#[test]
fn test_seqfidevaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
    let obj = Arc::new(Stepped::new(func));
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);

    let mut rng = rand::rng();
    let pair = <Sp<Mixed, NoDomain> as Searchspace<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        EmptyInfo,
    >>::sample_pair(&sp, &mut rng, sinfo.clone());
    let sobj_bis = (pair.id(), pair.get_sobj().x.clone());
    let sopt_bis = (pair.id(), pair.get_sopt().x.clone());
    let mut eval = FidSeqEvaluator::new(Some(pair), IdxMapPool::new(None));

    let out = <FidSeqEvaluator<
        SId,
        EmptyInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        RandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        Option<
            OutShapeEvaluate<
                SId,
                EmptyInfo,
                Sp<Mixed, NoDomain>,
                FidelitySol<SId, Mixed, EmptyInfo>,
                SingleCodomain<FidOutEvaluator>,
                FidOutEvaluator,
            >,
        >,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);
    let (comp, raw) = out.unwrap();

    assert_eq!(stop.calls(), 0, "Number of calls is wrong.");
    assert!(
        Arc::ptr_eq(&comp.get_sobj().sol.x, &sobj_bis.1),
        "Obj Partial and Computed do not point to the same solutions."
    );
    assert!(
        Arc::ptr_eq(&comp.get_sopt().sol.x, &sopt_bis.1),
        "Opt Partial and Computed do not point to the same solutions."
    );
    assert_eq!(
        comp.id(),
        sobj_bis.0,
        "Obj Id Computed and Partial do not point to the same solutions."
    );
    assert_eq!(
        comp.id(),
        sopt_bis.0,
        "Opt Id Computed and Partial do not point to the same solutions."
    );
    assert_eq!(
        raw.0, sobj_bis.0,
        "Obj Id Raw and Partial do not point to the same solutions."
    );
    assert_eq!(
        raw.0, sopt_bis.0,
        "Opt Id Raw and Partial do not point to the same solutions."
    );
    eval.update(IntoComputed::extract(comp).0);

    let out = <FidSeqEvaluator<
        SId,
        EmptyInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        RandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        Option<
            OutShapeEvaluate<
                SId,
                EmptyInfo,
                Sp<Mixed, NoDomain>,
                FidelitySol<SId, Mixed, EmptyInfo>,
                SingleCodomain<FidOutEvaluator>,
                FidOutEvaluator,
            >,
        >,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);
    let (comp, _) = out.unwrap();
    eval.update(IntoComputed::extract(comp).0);

    let out = <FidSeqEvaluator<
        SId,
        EmptyInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        RandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        Option<
            OutShapeEvaluate<
                SId,
                EmptyInfo,
                Sp<Mixed, NoDomain>,
                FidelitySol<SId, Mixed, EmptyInfo>,
                SingleCodomain<FidOutEvaluator>,
                FidOutEvaluator,
            >,
        >,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);
    let (comp, _) = out.unwrap();
    eval.update(IntoComputed::extract(comp).0);

    let out = <FidSeqEvaluator<
        SId,
        EmptyInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        RandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        Option<
            OutShapeEvaluate<
                SId,
                EmptyInfo,
                Sp<Mixed, NoDomain>,
                FidelitySol<SId, Mixed, EmptyInfo>,
                SingleCodomain<FidOutEvaluator>,
                FidOutEvaluator,
            >,
        >,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);
    let (comp, _) = out.unwrap();
    eval.update(IntoComputed::extract(comp).0);

    let out = <FidSeqEvaluator<
        SId,
        EmptyInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        IdxMapPool<SId, FnState, NoFuncStateCheck>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        RandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        Option<
            OutShapeEvaluate<
                SId,
                EmptyInfo,
                Sp<Mixed, NoDomain>,
                FidelitySol<SId, Mixed, EmptyInfo>,
                SingleCodomain<FidOutEvaluator>,
                FidOutEvaluator,
            >,
        >,
    >>::evaluate(&mut eval, &obj, &cod, &mut stop);
    let (comp, _) = out.unwrap();
    eval.update(IntoComputed::extract(comp).0);

    assert_eq!(stop.calls(), 1, "Number of calls is wrong.");
}
