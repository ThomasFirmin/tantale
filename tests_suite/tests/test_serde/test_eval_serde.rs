use std::sync::{Arc, Mutex};

use rmp_serde;
use tantale::algos::{BatchRandomSearch, RSInfo};
use tantale::core::experiment::basics::{LoadPool, Pool};
use tantale::core::stop::Calls;
use tantale::core::{
    BaseSol, Codomain, EmptyInfo, EvalStep, FidelitySol, FolderConfig, MessagePack, Mixed,
    MixedTypeDom, MonoCheckpointer, Objective, SId, Searchspace, SingleCodomain, Sp, Stepped,
    checkpointer::{Checkpointer, FuncStateCheckpointer, messagepack::MPFnStateCheckpointer},
    domain::NoDomain,
    experiment::{
        BatchEvaluator, FidBatchEvaluator, FidThrBatchEvaluator, MonoEvaluate, OutBatchEvaluate,
        ThrBatchEvaluator, ThrEvaluate, basics::IdxMapPool,
    },
    solution::{Batch, HasId, Lone, SolutionShape},
};

use super::init_func::{FidOutEvaluator, FnState, OutEvaluator, sp_evaluator, sp_evaluator_fid};

type BEvaluator = BatchEvaluator<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<BaseSol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
>;
type ThrBEvaluator = ThrBatchEvaluator<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<BaseSol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
>;
type FBEvaluator = FidBatchEvaluator<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
    FnState,
    Pool<MPFnStateCheckpointer, FnState, SId>,
>;
type FThrBEvaluator = FidThrBatchEvaluator<
    SId,
    EmptyInfo,
    RSInfo,
    Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
    FnState,
    Pool<MPFnStateCheckpointer, FnState, SId>,
>;

struct Cleaner {
    path: String,
}
impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn test_serde_batchevaluator() {
    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = SingleCodomain::new(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);
    let mut acc = SingleCodomain::new_accumulator();

    let mut rng = rand::rng();
    let sobj: Vec<BaseSol<_, _, _>> = <Sp<Mixed, NoDomain> as Searchspace<
        BaseSol<SId, _, _>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());
    let eval = BatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: BatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BaseSol<SId, _, _>, SId, _, EmptyInfo>,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch.into_iter().zip(&neval.batch).all(|(i, j)| {
            (i.id() == j.id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, _braw) = <BatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BaseSol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
    > as MonoEvaluate<
        BaseSol<SId, _, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        OutEvaluator,
        Calls,
        Objective<Arc<[MixedTypeDom]>, OutEvaluator>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, BaseSol<SId, _, _>, _, _>,
    >>::evaluate(&mut neval, &obj, &cod, &mut stop, &mut acc);

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    neval.update(batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let nneval: BEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        neval.batch.into_iter().zip(&nneval.batch).all(|(i, j)| {
            (i.id() == j.id())
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
    let acc = Arc::new(Mutex::new(SingleCodomain::new_accumulator()));

    let mut rng = rand::rng();
    let sobj: Vec<BaseSol<_, _, _>> = <Sp<Mixed, NoDomain> as Searchspace<
        BaseSol<SId, _, _>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());
    let eval = ThrBatchEvaluator::new(batch);

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: ThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BaseSol<SId, _, _>, SId, _, EmptyInfo>,
    > = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    assert!(
        eval.batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&neval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.id() == j.id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, _braw) = <ThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<BaseSol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
    > as ThrEvaluate<
        BaseSol<SId, _, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        OutEvaluator,
        Calls,
        Objective<Arc<[MixedTypeDom]>, OutEvaluator>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, BaseSol<SId, _, _>, _, _>,
    >>::evaluate(
        &mut neval,
        obj.clone(),
        cod.clone(),
        stop.clone(),
        acc.clone(),
    );

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    neval.update(batch);

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
                (i.id() == j.id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );
}

#[test]
fn test_serde_fidbatchevaluator() {
    drop(Cleaner {
        path: String::from("tmp_test_serde_fidbatchevaluator"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_serde_fidbatchevaluator"),
    };

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);
    let mut acc = SingleCodomain::new_accumulator();

    let mut rng = rand::rng();
    let sobj: Vec<FidelitySol<_, _, _>> = <Sp<Mixed, NoDomain> as Searchspace<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());

    let config = FolderConfig::new("tmp_test_serde_fidbatchevaluator").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    checkpointer.init();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let eval: FidBatchEvaluator<_, _, _, _, FnState, Pool<MPFnStateCheckpointer, FnState, SId>> =
        FidBatchEvaluator::new(batch, Pool::IdxMap(IdxMapPool::new(Some(fn_check))));

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: FBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    let config = FolderConfig::new("tmp_test_serde_fidbatchevaluator").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let fn_states = fn_check.load_all_func_state();
    let mut pool = IdxMapPool::from_iter(fn_states);
    pool.check = Some(fn_check);
    neval.pool = Pool::IdxMap(pool);
    checkpointer.after_load();

    assert!(
        eval.batch.into_iter().zip(&neval.batch).all(|(i, j)| {
            (i.id() == j.id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, braw) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        Pool<MPFnStateCheckpointer, FnState, SId>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut neval, &obj, &cod, &mut stop, &mut acc);

    assert!(
        braw.into_iter()
            .all(|(_i, o)| { Into::<EvalStep>::into(o.fid).0 == 1 }),
        "Error while serializing and deserializing function states."
    );

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    neval.update(batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let mut nneval: FBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    let config = FolderConfig::new("tmp_test_serde_fidbatchevaluator").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let fn_states = fn_check.load_all_func_state();
    let mut pool = IdxMapPool::from_iter(fn_states);
    pool.check = Some(fn_check);
    nneval.pool = Pool::IdxMap(pool);
    checkpointer.after_load();

    assert!(
        neval.batch.into_iter().zip(&nneval.batch).all(|(i, j)| {
            (i.id() == j.id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (_bcomp, braw) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        Pool<MPFnStateCheckpointer, FnState, SId>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut nneval, &obj, &cod, &mut stop, &mut acc);

    assert!(
        braw.into_iter()
            .all(|(_i, o)| { Into::<EvalStep>::into(o.fid).0 == 2 }),
        "Error while serializing and deserializing function states."
    );
}

#[test]
fn test_serde_thrfidbatchevaluator() {
    drop(Cleaner {
        path: String::from("tmp_test_serde_fidthrbatchevaluator"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_serde_fidthrbatchevaluator"),
    };

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = Arc::new(SingleCodomain::new(|o: &FidOutEvaluator| o.obj));
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(Mutex::new(Calls::new(50)));
    let acc = Arc::new(Mutex::new(SingleCodomain::new_accumulator()));
    let mut rng = rand::rng();
    let sobj: Vec<FidelitySol<_, _, _>> = <Sp<Mixed, NoDomain> as Searchspace<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());

    let config = FolderConfig::new("tmp_test_serde_fidthrbatchevaluator").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    checkpointer.init();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let eval: FidThrBatchEvaluator<_, _, _, _, FnState, Pool<MPFnStateCheckpointer, FnState, SId>> =
        FidThrBatchEvaluator::new(batch, Pool::IdxMap(IdxMapPool::new(Some(fn_check))));

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: FThrBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    let config = FolderConfig::new("tmp_test_serde_fidthrbatchevaluator").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let fn_states = fn_check.load_all_func_state();
    let mut pool = IdxMapPool::from_iter(fn_states);
    pool.check = Some(fn_check);
    neval.pool = Arc::new(Mutex::new(Pool::IdxMap(pool)));
    checkpointer.after_load();

    assert!(
        eval.batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&neval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.id() == j.id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, braw) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        Pool<MPFnStateCheckpointer, FnState, SId>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(
        &mut neval,
        obj.clone(),
        cod.clone(),
        stop.clone(),
        acc.clone(),
    );

    assert!(
        braw.into_iter()
            .all(|(_i, o)| { Into::<EvalStep>::into(o.fid).0 == 1 }),
        "Error while serializing and deserializing function states."
    );

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    neval.update(batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let mut nneval: FThrBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    let config = FolderConfig::new("tmp_test_serde_fidthrbatchevaluator").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let fn_states = fn_check.load_all_func_state();
    let mut pool = IdxMapPool::from_iter(fn_states);
    pool.check = Some(fn_check);
    nneval.pool = Arc::new(Mutex::new(Pool::IdxMap(pool)));
    checkpointer.after_load();

    assert!(
        neval
            .batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&nneval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.id() == j.id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (_bcomp, braw) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        Pool<MPFnStateCheckpointer, FnState, SId>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(
        &mut nneval,
        obj.clone(),
        cod.clone(),
        stop.clone(),
        acc.clone(),
    );

    assert!(
        braw.into_iter()
            .all(|(_i, o)| { Into::<EvalStep>::into(o.fid).0 == 2 }),
        "Error while serializing and deserializing function states."
    );
}

#[test]
fn test_serde_fidbatchevaluator_loadpool() {
    drop(Cleaner {
        path: String::from("tmp_test_serde_fidbatchevaluator_loadpool"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_serde_fidbatchevaluator_loadpool"),
    };

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = SingleCodomain::new(|o: &FidOutEvaluator| o.obj);
    let obj = Stepped::new(func);
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let mut stop = Calls::new(50);
    let mut acc = SingleCodomain::new_accumulator();

    let mut rng = rand::rng();
    let sobj: Vec<FidelitySol<_, _, _>> = <Sp<Mixed, NoDomain> as Searchspace<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());

    let config = FolderConfig::new("tmp_test_serde_fidbatchevaluator_loadpool").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    checkpointer.init();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let eval: FidBatchEvaluator<_, _, _, _, FnState, Pool<MPFnStateCheckpointer, FnState, SId>> =
        FidBatchEvaluator::new(batch, Pool::Load(LoadPool::new(fn_check)));

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: FBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    let config = FolderConfig::new("tmp_test_serde_fidbatchevaluator_loadpool").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let pool = LoadPool::new(fn_check);
    neval.pool = Pool::Load(pool);
    checkpointer.after_load();

    assert!(
        eval.batch.into_iter().zip(&neval.batch).all(|(i, j)| {
            (i.id() == j.id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, braw) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        Pool<MPFnStateCheckpointer, FnState, SId>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut neval, &obj, &cod, &mut stop, &mut acc);

    assert!(
        braw.into_iter()
            .all(|(_i, o)| { Into::<EvalStep>::into(o.fid).0 == 1 }),
        "Error while serializing and deserializing function states."
    );

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    neval.update(batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let mut nneval: FBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    let config = FolderConfig::new("tmp_test_serde_fidbatchevaluator_loadpool").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let fn_states = fn_check.load_all_func_state();
    let mut pool = IdxMapPool::from_iter(fn_states);
    pool.check = Some(fn_check);
    nneval.pool = Pool::IdxMap(pool);
    checkpointer.after_load();

    assert!(
        neval.batch.into_iter().zip(&nneval.batch).all(|(i, j)| {
            (i.id() == j.id())
                && (i.get_sobj().x == j.get_sobj().x)
                && (i.get_sopt().x == j.get_sopt().x)
        }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (_bcomp, braw) = <FidBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        Pool<MPFnStateCheckpointer, FnState, SId>,
    > as MonoEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(&mut nneval, &obj, &cod, &mut stop, &mut acc);

    assert!(
        braw.into_iter()
            .all(|(_i, o)| { Into::<EvalStep>::into(o.fid).0 == 2 }),
        "Error while serializing and deserializing function states."
    );
}

#[test]
fn test_serde_thrfidbatchevaluator_loadpool() {
    drop(Cleaner {
        path: String::from("tmp_test_serde_fidthrbatchevaluator_loadpool"),
    });
    let _cleaner = Cleaner {
        path: String::from("tmp_test_serde_fidthrbatchevaluator_loadpool"),
    };

    let sp = sp_evaluator_fid::get_searchspace();
    let func = sp_evaluator_fid::example;
    let cod = Arc::new(SingleCodomain::new(|o: &FidOutEvaluator| o.obj));
    let obj = Arc::new(Stepped::new(func));
    let info = std::sync::Arc::new(RSInfo { iteration: 0 });
    let sinfo = std::sync::Arc::new(EmptyInfo {});
    let stop = Arc::new(Mutex::new(Calls::new(50)));
    let acc = Arc::new(Mutex::new(SingleCodomain::new_accumulator()));
    let mut rng = rand::rng();
    let sobj: Vec<FidelitySol<_, _, _>> = <Sp<Mixed, NoDomain> as Searchspace<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        EmptyInfo,
    >>::vec_sample_obj(&sp, &mut rng, 20, sinfo.clone());
    let pairs = sp.vec_onto_obj(sobj);
    let batch = Batch::new(pairs, info.clone());

    let config = FolderConfig::new("tmp_test_serde_fidthrbatchevaluator_loadpool").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    checkpointer.init();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let eval: FidThrBatchEvaluator<_, _, _, _, FnState, Pool<MPFnStateCheckpointer, FnState, SId>> =
        FidThrBatchEvaluator::new(batch, Pool::Load(LoadPool::new(fn_check)));

    let eval_ser = rmp_serde::encode::to_vec(&eval).unwrap();
    let mut neval: FThrBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    let config = FolderConfig::new("tmp_test_serde_fidthrbatchevaluator_loadpool").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let pool = LoadPool::new(fn_check);
    neval.pool = Arc::new(Mutex::new(Pool::Load(pool)));
    checkpointer.after_load();

    assert!(
        eval.batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&neval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.id() == j.id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (bcomp, braw) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        Pool<MPFnStateCheckpointer, FnState, SId>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(
        &mut neval,
        obj.clone(),
        cod.clone(),
        stop.clone(),
        acc.clone(),
    );

    assert!(
        braw.into_iter()
            .all(|(_i, o)| { Into::<EvalStep>::into(o.fid).0 == 1 }),
        "Error while serializing and deserializing function states."
    );

    let pairs = bcomp
        .into_iter()
        .map(|p| Lone::new(p.extract_sobj().sol))
        .collect();
    let batch = Batch::new(pairs, info.clone());
    neval.update(batch);

    let eval_ser = rmp_serde::encode::to_vec(&neval).unwrap();
    let mut nneval: FThrBEvaluator = rmp_serde::decode::from_slice(&eval_ser).unwrap();

    let config = FolderConfig::new("tmp_test_serde_fidthrbatchevaluator_loadpool").into();
    let mut checkpointer = MessagePack::new(config).unwrap();
    let fn_check = checkpointer.new_func_state_checkpointer();
    let fn_states = fn_check.load_all_func_state();
    let mut pool = IdxMapPool::from_iter(fn_states);
    pool.check = Some(fn_check);
    nneval.pool = Arc::new(Mutex::new(Pool::IdxMap(pool)));
    checkpointer.after_load();

    assert!(
        neval
            .batch
            .lock()
            .unwrap()
            .pairs
            .iter()
            .zip(&nneval.batch.lock().unwrap().pairs)
            .all(|(i, j)| {
                (i.id() == j.id())
                    && (i.get_sobj().x == j.get_sobj().x)
                    && (i.get_sopt().x == j.get_sopt().x)
            }),
        "Solution mismatch after serializing and deserializing Evaluator."
    );

    let (_bcomp, braw) = <FidThrBatchEvaluator<
        SId,
        EmptyInfo,
        RSInfo,
        Lone<FidelitySol<SId, Mixed, EmptyInfo>, SId, Mixed, EmptyInfo>,
        FnState,
        Pool<MPFnStateCheckpointer, FnState, SId>,
    > as ThrEvaluate<
        FidelitySol<SId, Mixed, EmptyInfo>,
        SId,
        BatchRandomSearch,
        Sp<Mixed, NoDomain>,
        FidOutEvaluator,
        Calls,
        Stepped<Arc<[MixedTypeDom]>, FidOutEvaluator, FnState>,
        OutBatchEvaluate<SId, _, _, Sp<Mixed, NoDomain>, FidelitySol<SId, _, _>, _, _>,
    >>::evaluate(
        &mut nneval,
        obj.clone(),
        cod.clone(),
        stop.clone(),
        acc.clone(),
    );

    assert!(
        braw.into_iter()
            .all(|(_i, o)| { Into::<EvalStep>::into(o.fid).0 == 2 }),
        "Error while serializing and deserializing function states."
    );
}
