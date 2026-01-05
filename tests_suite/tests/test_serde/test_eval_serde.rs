use std::sync::{Arc, Mutex};

use rmp_serde;
use tantale::core::stop::Calls;
use tantale_algos::{RSInfo, BatchRandomSearch};
use tantale_core::{
    domain::NoDomain,
    experiment::{
        BatchEvaluator, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, ThrBatchEvaluator,
        ThrEvaluate,
    },
    solution::{Batch, HasId, Lone, SolutionShape},
    BaseDom, BasePartial, BaseTypeDom, EmptyInfo, FidBasePartial, Objective, SId, Searchspace,
    SingleCodomain, Sp, Stepped,
};

use super::init_func::{sp_evaluator, sp_evaluator_fid, FidOutEvaluator, FnState, OutEvaluator};

type BEvaluator = BatchEvaluator<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<BasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
>;
type ThrBEvaluator = ThrBatchEvaluator<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<BasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
>;
type FBEvaluator = FidBatchEvaluator<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
    FnState,
>;
type FThrBEvaluator = FidThrBatchEvaluator<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
    FnState,
>;

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
    let sobj: Vec<BasePartial<_, _, _>> = <Sp<BaseDom, NoDomain> as Searchspace<
        BasePartial<SId, _, _>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(
        &sp, Some(&mut rng), 20, sinfo.clone()
    );
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());
    let eval = BatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: BatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BasePartial<SId, _, _>, SId, _, EmptyInfo>,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch.into_iter().zip(&neval.batch).all(|(i, j)| {
            (i.get_id() == j.get_id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, _braw) = <BatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
    > as MonoEvaluate<
        BasePartial<SId, _, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        OutEvaluator,
        Calls,
        Objective<Arc<[BaseTypeDom]>, OutEvaluator>,
    >>::evaluate(&mut neval, &obj, &cod, &mut stop);

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    <
        BatchEvaluator<
            SId,
            EmptyInfo,
            RSInfo,
            Lone<BasePartial<SId, _,EmptyInfo>,SId,_,EmptyInfo>
        > as 
        MonoEvaluate<
            BasePartial<SId, _,EmptyInfo>,
            SId,
            BatchRandomSearch,
            Sp<BaseDom,NoDomain>,
            OutEvaluator,
            Calls,
            Objective<Arc<[BaseTypeDom]>,OutEvaluator>
        >
        >::update(&mut neval, batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let nneval: BEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval.batch.into_iter().zip(&nneval.batch).all(|(i, j)| {
            (i.get_id() == j.get_id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
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
    let stop = Arc::new(Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj: Vec<BasePartial<_, _, _>> = <Sp<BaseDom, NoDomain> as Searchspace<
        BasePartial<SId, _, _>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(
        &sp, Some(&mut rng), 20, sinfo.clone()
    );
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());
    let eval = ThrBatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: ThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BasePartial<SId, _, _>, SId, _, EmptyInfo>,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&neval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.get_id() == j.get_id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, _braw) = <ThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
    > as ThrEvaluate<
        BasePartial<SId, _, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        OutEvaluator,
        Calls,
        Objective<Arc<[BaseTypeDom]>, OutEvaluator>,
    >>::evaluate(&mut neval, obj.clone(), cod.clone(), stop.clone());

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    <ThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BasePartial<SId, _, EmptyInfo>, SId, _, EmptyInfo>,
    > as ThrEvaluate<
        BasePartial<SId, _, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        OutEvaluator,
        Calls,
        Objective<Arc<[BaseTypeDom]>, OutEvaluator>,
    >>::update(&mut neval, batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let nneval: ThrBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval
            .batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&nneval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.get_id() == j.get_id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );
}

#[test]
fn test_serde_fidbatchevaluator() {
    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);

    let mut rng = rand::rng();
    let sobj: Vec<FidBasePartial<_, _, _>> =
        <Sp<BaseDom, NoDomain> as Searchspace<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            EmptyInfo,
        >>::vec_sample_obj(&sp, Some(&mut rng), 20, sinfo.clone());
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());
    let eval: FidBatchEvaluator<_, _, _, _, FnState> = FidBatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: FBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch.into_iter().zip(&neval.batch).all(|(i, j)| {
            (i.get_id() == j.get_id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, braw) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        FnState,
    > as MonoEvaluate<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
    >>::evaluate(&mut neval, &obj, &cod, &mut stop);

    assert!(
        braw.into_iter().all(|(_i, o)| { o.fid.0 == 1 }),
        "Error while serializing and deserializing function states."
    );

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        FnState,
    > as MonoEvaluate<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
    >>::update(&mut neval, batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let mut nneval: FBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval.batch.into_iter().zip(&nneval.batch).all(|(i, j)| {
            (i.get_id() == j.get_id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (_bcomp, braw) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        FnState,
    > as MonoEvaluate<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
    >>::evaluate(&mut nneval, &obj, &cod, &mut stop);

    assert!(
        braw.into_iter().all(|(_i, o)| { o.fid.0 == 2 }),
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
    let stop = Arc::new(Mutex::new(Calls::new(50)));

    let mut rng = rand::rng();
    let sobj: Vec<FidBasePartial<_, _, _>> =
        <Sp<BaseDom, NoDomain> as Searchspace<
            FidBasePartial<SId, BaseDom, EmptyInfo>,
            SId,
            EmptyInfo,
        >>::vec_sample_obj(&sp, Some(&mut rng), 20, sinfo.clone());
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());
    let eval: FidThrBatchEvaluator<_, _, _, _, FnState> = FidThrBatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: FThrBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&neval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.get_id() == j.get_id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, braw) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        FnState,
    > as ThrEvaluate<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
    >>::evaluate(&mut neval, obj.clone(), cod.clone(), stop.clone());

    assert!(
        braw.into_iter().all(|(_i, o)| { o.fid.0 == 1 }),
        "Error while serializing and deserializing function states."
    );

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        FnState,
    > as ThrEvaluate<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
    >>::update(&mut neval, batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let mut nneval: FThrBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval
            .batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&nneval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.get_id() == j.get_id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (_bcomp, braw) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidBasePartial<SId, BaseDom, EmptyInfo>, SId, BaseDom, EmptyInfo>,
        FnState,
    > as ThrEvaluate<
        FidBasePartial<SId, BaseDom, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<BaseDom, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[BaseTypeDom]>, FidOutEvaluator, FnState>,
    >>::evaluate(&mut nneval, obj.clone(), cod.clone(), stop.clone());

    assert!(
        braw.into_iter().all(|(_i, o)| { o.fid.0 == 2 }),
        "Error while serializing and deserializing function states."
    );
}
