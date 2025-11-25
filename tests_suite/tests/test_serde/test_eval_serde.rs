use std::sync::Arc;

use rmp_serde;
use tantale::core::stop::Calls;
use tantale_algos::RSInfo;
use tantale_core::{
    experiment::{
        BatchEvaluator, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, ThrBatchEvaluator,
        ThrEvaluate,
    },
    solution::{partial::FidelityPartial, Batch},
    BaseDom, BasePartial, EmptyInfo, FidBasePartial, Objective, SId, Searchspace,
    SingleCodomain, Sp, Stepped,
};

use super::init_func::{sp_evaluator, sp_evaluator_fid, FidOutEvaluator, FnState, OutEvaluator};

#[test]
fn test_serde_batchevaluator() {
    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = SingleCodomain::new(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);

    let mut rng = rand::rng();
    let sobj: Vec<BasePartial<_, _, _>> = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
    let sopt: Vec<BasePartial<_, _, _>> = sp.vec_onto_obj(&sobj);
    let batch = Batch::new(sobj, sopt, info.clone());
    let eval: BatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > = BatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: BatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch.sobj.iter().zip(&neval.batch.sobj).all(|(i, j)| {
            let isol = i.x.clone();
            let jsol = j.x.clone();
            (i.id == j.id) && (isol == jsol)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (_braw, bcomp) = <BatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > as MonoEvaluate<
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        BasePartial<SId, BaseDom, EmptyInfo>,
        Calls,
        SingleCodomain<OutEvaluator>,
        OutEvaluator,
        Sp<BaseDom, BaseDom>,
        Objective<BaseDom, OutEvaluator>,
        Batch<BasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, BaseDom, EmptyInfo, RSInfo>,
    >>::evaluate(&mut neval, &obj, &cod, &mut stop);

    let (vobj, vopt) = bcomp
        .into_iter()
        .map(|(sj, st)| {
            let obj = sj.sol;
            let opt = st.sol;
            (obj, opt)
        })
        .collect();
    let batch = Batch::new(vobj, vopt, info.clone());
    <BatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > as MonoEvaluate<
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        BasePartial<SId, BaseDom, EmptyInfo>,
        Calls,
        SingleCodomain<OutEvaluator>,
        OutEvaluator,
        Sp<BaseDom, BaseDom>,
        Objective<BaseDom, OutEvaluator>,
        Batch<BasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, BaseDom, EmptyInfo, RSInfo>,
    >>::update(&mut neval, batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let nneval: BatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval
            .batch
            .sobj
            .iter()
            .zip(&nneval.batch.sobj)
            .all(|(i, j)| {
                let isol = i.x.clone();
                let jsol = j.x.clone();
                (i.id == j.id) && (isol == jsol)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );
}

#[test]
fn test_serde_thrbatchevaluator() {
    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = Arc::new(SingleCodomain::new(|o: &OutEvaluator| o.obj));
    let obj = Arc::new(Objective::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(std::sync::Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj: Vec<BasePartial<_, _, _>> = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
    let sopt: Vec<BasePartial<_, _, _>> = sp.vec_onto_obj(&sobj);
    let batch = Batch::new(sobj, sopt, info.clone());
    let eval: ThrBatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > = ThrBatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: ThrBatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch
            .lock()
            .unwrap()
            .sobj
            .iter()
            .zip(&neval.batch.lock().unwrap().sobj)
            .all(|(i, j)| {
                let isol = i.x.clone();
                let jsol = j.x.clone();
                (i.id == j.id) && (isol == jsol)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (_braw, bcomp) = <ThrBatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > as ThrEvaluate<
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        BasePartial<SId, BaseDom, EmptyInfo>,
        Calls,
        SingleCodomain<OutEvaluator>,
        OutEvaluator,
        Sp<BaseDom, BaseDom>,
        Objective<BaseDom, OutEvaluator>,
        Batch<BasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, BaseDom, EmptyInfo, RSInfo>,
    >>::evaluate(&mut neval, obj.clone(), cod.clone(), stop.clone());

    let (vobj, vopt) = bcomp
        .into_iter()
        .map(|(sj, st)| {
            let obj = sj.sol;
            let opt = st.sol;
            (obj, opt)
        })
        .collect();
    let batch = Batch::new(vobj, vopt, info.clone());
    <ThrBatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > as ThrEvaluate<
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        BasePartial<SId, BaseDom, EmptyInfo>,
        Calls,
        SingleCodomain<OutEvaluator>,
        OutEvaluator,
        Sp<BaseDom, BaseDom>,
        Objective<BaseDom, OutEvaluator>,
        Batch<BasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, BaseDom, EmptyInfo, RSInfo>,
    >>::update(&mut neval, batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let nneval: ThrBatchEvaluator<
        BasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval
            .batch
            .lock()
            .unwrap()
            .sobj
            .iter()
            .zip(&nneval.batch.lock().unwrap().sobj)
            .all(|(i, j)| {
                let isol = i.x.clone();
                let jsol = j.x.clone();
                (i.id == j.id) && (isol == jsol)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );
}

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
    let sobj: Vec<FidBasePartial<_, _, _>> = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
    let sopt: Vec<FidBasePartial<_, _, _>> = sp.vec_onto_obj(&sobj);
    let batch = Batch::new(sobj, sopt, info.clone());
    let eval: FidBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > = FidBatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: FidBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch.sobj.iter().zip(&neval.batch.sobj).all(|(i, j)| {
            let isol = i.x.clone();
            let jsol = j.x.clone();
            (i.fid == j.fid) && (i.id == j.id) && (isol == jsol)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

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
    >>::evaluate(&mut neval, &obj, &cod, &mut stop);

    assert!(
        braw.into_iter().all(|(_i, o)| {o.fid.0 == 1.0}),
        "Error while serializing and deserializing function states."
    );

    let (vobj, vopt) = bcomp
        .into_iter()
        .map(|(sj, st)| {
            let mut obj = sj.sol;
            let mut opt = st.sol;
            obj.resume(&mut opt, 0.0);
            (obj, opt)
        })
        .collect();
    let batch = Batch::new(vobj, vopt, info.clone());
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
    >>::update(&mut neval, batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let mut nneval: FidBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval
            .batch
            .sobj
            .iter()
            .zip(&nneval.batch.sobj)
            .all(|(i, j)| {
                let isol = i.x.clone();
                let jsol = j.x.clone();
                (i.fid == j.fid) && (i.id == j.id) && (isol == jsol)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (braw, _bcomp) = <FidBatchEvaluator<
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
    >>::evaluate(&mut nneval, &obj, &cod, &mut stop);

    assert!(
        braw.into_iter().all(|(_i, o)| {o.fid.0 == 2.0}),
        "Error while serializing and deserializing function states."
    );
}

#[test]
fn test_serde_thrfidbatchevaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = Arc::new(SingleCodomain::new(|o: &FidOutEvaluator| o.obj));
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(std::sync::Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj: Vec<FidBasePartial<_, _, _>> = sp.vec_sample_obj(Some(&mut rng), 20, sinfo.clone());
    let sopt: Vec<FidBasePartial<_, _, _>> = sp.vec_onto_obj(&sobj);
    let batch = Batch::new(sobj, sopt, info.clone());
    let eval: FidThrBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > = FidThrBatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: FidThrBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch
            .lock()
            .unwrap()
            .sobj
            .iter()
            .zip(&neval.batch.lock().unwrap().sobj)
            .all(|(i, j)| {
                let isol = i.x.clone();
                let jsol = j.x.clone();
                (i.fid == j.fid) && (i.id == j.id) && (isol == jsol)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

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
    >>::evaluate(&mut neval, obj.clone(), cod.clone(), stop.clone());

    assert!(
        braw.into_iter().all(|(_i, o)| {o.fid.0 == 1.0}),
        "Error while serializing and deserializing function states."
    );

    let (vobj, vopt) = bcomp
        .into_iter()
        .map(|(sj, st)| {
            let mut obj = sj.sol;
            let mut opt = st.sol;
            obj.resume(&mut opt, 0.0);
            (obj, opt)
        })
        .collect();
    let batch = Batch::new(vobj, vopt, info.clone());
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
    >>::update(&mut neval, batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let nneval: FidThrBatchEvaluator<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BaseDom,
        BaseDom,
        EmptyInfo,
        RSInfo,
        FnState,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval
            .batch
            .lock()
            .unwrap()
            .sobj
            .iter()
            .zip(&nneval.batch.lock().unwrap().sobj)
            .all(|(i, j)| {
                let isol = i.x.clone();
                let jsol = j.x.clone();
                (i.fid == j.fid) && (i.id == j.id) && (isol == jsol)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (braw, _bcomp) = <FidThrBatchEvaluator<
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
    >>::evaluate(&mut neval, obj.clone(), cod.clone(), stop.clone());

    assert!(
        braw.into_iter().all(|(_i, o)| {o.fid.0 == 2.0}),
        "Error while serializing and deserializing function states."
    );
}
