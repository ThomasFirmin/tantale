use tantale::algos::{
    tpe,
    Univariate,
    LinearSplit,
    Tpe,
    UniformWeighter,
};
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, Objective, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::{OutEvaluator, sp_evaluator};
use crate::run_checker::run_reader;

#[test]
fn test_tpe_seq_run() {
    let _clean = Cleaner::new("tmp_test_tpe_seqrun");

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let opt = Tpe::new(
        5,
        10,
        Univariate,
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let cod = tpe::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_tpe_seqrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = tpe::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(mono, Tpe<Univariate, UniformWeighter, LinearSplit, _, _, _, _, _>, Evaluated, (sp, cod), obj, (rec, check));

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = tpe::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(mono, Tpe<Univariate, UniformWeighter, LinearSplit, _, _, _, _, _>, Evaluated, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_tpe_seqrun", 100);
}

#[test]
fn test_tpe_seqthr_run() {
    let _clean = Cleaner::new("tmp_test_tpe_seqthrrun");

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let opt = Tpe::new(
        5,
        10,
        Univariate,
        UniformWeighter::default(),
        LinearSplit::new(0.25).unwrap(),
    );
    let cod = tpe::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded((sp, cod), obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_tpe_seqthrrun", 50);

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = tpe::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(threaded, Tpe<Univariate, UniformWeighter, LinearSplit, _, _, _, _, _>, Evaluated, (sp, cod), obj, (rec, check));

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator::get_searchspace();
    let func = sp_evaluator::example;
    let cod = tpe::codomain(|o: &OutEvaluator| o.obj);
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(threaded, Tpe<Univariate, UniformWeighter, LinearSplit, _, _, _, _, _>, Evaluated, (sp, cod), obj, (rec, check));
    run_reader("tmp_test_tpe_seqthrrun", 100);
}