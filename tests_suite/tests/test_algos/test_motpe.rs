use tantale::algos::bayesian::splitter::MOSplit;
use tantale::algos::{Tpe, UniformWeighter, Univariate, tpe};
use tantale::core::{
    CSVRecorder, FolderConfig, MessagePack, Objective, SaverConfig,
    experiment::{Runable, mono, threaded},
    load,
    stop::Evaluated,
};

use crate::cleaner::Cleaner;
use crate::init_func::sp_evaluator_mo;
use crate::run_checker::{run_reader, run_reader_eps};

#[test]
fn test_mo_tpe_seq_run() {
    let _clean = Cleaner::new("tmp_test_mo_tpe_seqrun");

    let sp = sp_evaluator_mo::get_searchspace();
    let func = sp_evaluator_mo::example;
    let opt = Tpe::new(
        5,
        10,
        Univariate,
        UniformWeighter::default(),
        MOSplit::new(0.25).unwrap(),
    );
    let obj = Objective::new(func);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_mo_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = mono(sp, obj, opt, stop, (rec, check));
    exp.run();

    run_reader("tmp_test_mo_tpe_seqrun", 50);

    let sp = sp_evaluator_mo::get_searchspace();
    let func = sp_evaluator_mo::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_mo_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        mono,
        tpe!(Univariate, UniformWeighter, MOSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert_eq!(expstop.calls(), 50, "Number of calls is wrong");
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_mo::get_searchspace();
    let func = sp_evaluator_mo::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_mo_tpe_seqrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(
        mono,
        tpe!(Univariate, UniformWeighter, MOSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );
    run_reader("tmp_test_mo_tpe_seqrun", 100);
}

#[test]
fn test_mo_tpe_seqthr_run() {
    let _clean = Cleaner::new("tmp_test_mo_tpe_seqthrrun");

    let sp = sp_evaluator_mo::get_searchspace();
    let func = sp_evaluator_mo::example;
    let opt = Tpe::new(
        5,
        10,
        Univariate,
        UniformWeighter::default(),
        MOSplit::new(0.25).unwrap(),
    );
    let obj = Objective::new(func);
    let stop = Evaluated::new(50);
    let config = FolderConfig::new("tmp_test_mo_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config);

    let exp = threaded(sp, obj, opt, stop, (rec, check));
    exp.run();

    run_reader_eps("tmp_test_mo_tpe_seqthrrun", 50, num_cpus::get() * 4);

    let sp = sp_evaluator_mo::get_searchspace();
    let func = sp_evaluator_mo::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_mo_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let mut exp = load!(
        threaded,
        tpe!(Univariate, UniformWeighter, MOSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );

    let expstop: &mut Evaluated = exp.get_mut_stop();
    assert!(
        expstop.calls() >= 50 && expstop.calls() <= 50 + num_cpus::get(),
        "Number of calls is wrong"
    );
    expstop.add(50);

    exp.run();

    let sp = sp_evaluator_mo::get_searchspace();
    let func = sp_evaluator_mo::example;
    let obj = Objective::new(func);

    let config = FolderConfig::new("tmp_test_mo_tpe_seqthrrun").init();
    let rec = CSVRecorder::new(config.clone(), true, true, true, true);
    let check = MessagePack::new(config).unwrap();

    let _exp = load!(
        threaded,
        tpe!(Univariate, UniformWeighter, MOSplit),
        Evaluated,
        sp,
        obj,
        (rec, check)
    );
    run_reader_eps("tmp_test_mo_tpe_seqthrrun", 100, num_cpus::get() * 4);
}
