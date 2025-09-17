use super::init_sp::sp_m_equal_allmsamp;

use csv::StringRecord;
use tantale::core::{Codomain, Computed, Optimizer, Partial, SId, Searchspace, SingleCodomain};
use tantale_algos::RSInfo;
use tantale_algos::RandomSearch;
use tantale_core::Objective;
use tantale_core::experiment::sequential::seqevaluator::Evaluator;
use tantale_core::saver::CSVSaver;
use tantale_core::saver::CSVWritable;
use tantale_core::saver::Saver;
use tantale_core::stop::{Calls, Stop};
use tantale_core::EmptyInfo;
use tantale_core::LinkedOutcome;
use tantale_core::Solution;
use tantale_core::Sp;
use tantale_core::{OptInfo, SolInfo};

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
mod infos {
    use serde::{Deserialize, Serialize};
    use tantale_macros::Outcome;
    #[derive(Outcome, Debug, Serialize, Deserialize)]
    pub struct OutExample {
        pub fid2: usize,
        pub con3: i64,
        pub con4: f64,
        pub con5: f64,
        pub mul6: f64,
        pub mul7: f64,
        pub mul8: f64,
        pub mul9: f64,
        pub tvec: Vec<f64>,
    }

    pub fn get_out(fid: usize, a: i64) -> OutExample {
        OutExample {
            fid2: fid,
            con3: a,
            con4: 4.4,
            con5: 5.5,
            mul6: 6.6,
            mul7: 7.7,
            mul8: 8.8,
            mul9: 9.9,
            tvec: Vec::from([1.1, 2.2, 3.3]),
        }
    }
}

use infos::OutExample;

type EvalType<Info,SInfo> = Evaluator<SId,sp_m_equal_allmsamp::_TantaleMixedObj,sp_m_equal_allmsamp::_TantaleMixedObj,Info,SInfo>;

pub fn run_saver<'a, Scp, Op, St, Sv>(
    hash_obj: &mut HashMap<
        usize,
        Arc<Partial<SId, sp_m_equal_allmsamp::_TantaleMixedObj, Op::SInfo>>,
    >,
    hash_opt: &mut HashMap<
        usize,
        Arc<Partial<SId, sp_m_equal_allmsamp::_TantaleMixedObj, Op::SInfo>>,
    >,
    hash_out: &mut HashMap<
        usize,
        Arc<LinkedOutcome<OutExample, SId, sp_m_equal_allmsamp::_TantaleMixedObj, Op::SInfo>>,
    >,
    hash_cod: &mut HashMap<
        usize,
        Arc<
            Computed<
                SId,
                sp_m_equal_allmsamp::_TantaleMixedObj,
                SingleCodomain<OutExample>,
                OutExample,
                Op::SInfo,
            >,
        >,
    >,
    hash_inf: &mut HashMap<usize, Arc<Op::Info>>,
    sp: Arc<Scp>,
    cod: Arc<SingleCodomain<OutExample>>,
    opt: &'a mut Op,
    stop: &mut St,
    saver: &mut Sv,
    size: usize,
) -> (St, &'a Op::State, Op, EvalType<Op::Info,Op::SInfo>)
where
    Scp: Searchspace<
        SId,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        Op::SInfo,
    >,
    Op: Optimizer<
        SId,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        SingleCodomain<OutExample>,
        OutExample,
        Scp,
    >,
    St: Stop + Send + Sync,
    Sv: Saver<
        SId,
        St,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        SingleCodomain<OutExample>,
        OutExample,
        Scp,
        Op,
        EvalType<Op::Info,Op::SInfo>,
        Objective<sp_m_equal_allmsamp::_TantaleMixedObj,SingleCodomain<OutExample>,OutExample>
    >,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
{
    let (sobj, sopt, infos) = opt.first_step(sp.clone());
    let eval: EvalType<Op::Info,Op::SInfo> = Evaluator::new(sobj.clone(), sopt.clone(),infos.clone());
    let (cobj, copt, vinfos): (Vec<_>, Vec<_>, Vec<_>) = sobj
        .iter()
        .zip(sopt.iter())
        .map(|(a, b)| {
            let id = a.get_id().id;
            let aelem = a.get_x()[0].clone();
            let aelem = match aelem {
                sp_m_equal_allmsamp::_TantaleMixedObjTypeDom::Int(ae) => ae,
                _ => panic!("Should be a Int."),
            };
            let outcome = Arc::new(infos::get_out(id, aelem));
            let mut codel = cod.get_elem(&outcome);
            codel.value = id as f64;
            let r =
                sp.computed::<SingleCodomain<_>, OutExample>(a.clone(), b.clone(), Arc::new(codel));
            let linked = LinkedOutcome::new(outcome.clone(), a.clone());
            hash_obj.insert(id, a.clone());
            hash_opt.insert(id, b.clone());
            hash_out.insert(id, Arc::new(LinkedOutcome::new(outcome.clone(), a.clone())));
            hash_cod.insert(id, r.0.clone());
            hash_inf.insert(id, infos.clone());
            (r.0, r.1, linked)
        })
        .collect();

    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);
    stop.update(tantale_core::stop::ExpStep::Distribution);

    let (cobj, copt, vinfos) = (Arc::new(cobj), Arc::new(copt), vinfos);
    saver.save_partial(
        cobj.clone(),
        copt.clone(),
        sp.clone(),
        cod.clone(),
        infos.clone(),
    );
    saver.save_codom(cobj.clone(), sp.clone(), cod.clone());
    saver.save_out(vinfos, sp.clone());
    saver.save_state(sp.clone(), opt.get_state(), stop, &eval);

    let (sobj, sopt, infos) = opt.step((cobj.clone(), copt.clone()), sp.clone());
    let eval: EvalType<Op::Info,Op::SInfo> = Evaluator::new(sobj.clone(), sopt.clone(), infos.clone());
    let computed: (Vec<_>, Vec<_>, Vec<_>) = sobj
        .iter()
        .zip(sopt.iter())
        .map(|(a, b)| {
            let id = a.get_id().id;
            let aelem = a.get_x()[0].clone();
            let aelem = match aelem {
                sp_m_equal_allmsamp::_TantaleMixedObjTypeDom::Int(ae) => ae,
                _ => panic!("Should be a Int."),
            };
            let outcome = Arc::new(infos::get_out(id, aelem));
            let cod = SingleCodomain::new(|x: &OutExample| x.mul6);
            let mut codel = cod.get_elem(&outcome);
            codel.value = id as f64;
            let r =
                sp.computed::<SingleCodomain<_>, OutExample>(a.clone(), b.clone(), Arc::new(codel));
            let linked = LinkedOutcome::new(outcome.clone(), a.clone());
            hash_obj.insert(id, a.clone());
            hash_opt.insert(id, b.clone());
            hash_out.insert(id, Arc::new(LinkedOutcome::new(outcome.clone(), a.clone())));
            hash_cod.insert(id, r.0.clone());
            hash_inf.insert(id, infos.clone());
            (r.0, r.1, linked)
        })
        .collect();
    
    let cobj = Arc::new(computed.0);
    let copt = Arc::new(computed.1);
    saver.save_partial(
        cobj.clone(),
        copt.clone(),
        sp.clone(),
        cod.clone(),
        infos.clone(),
    );
    saver.save_codom(cobj.clone(), sp.clone(), cod.clone());
    saver.save_out(computed.2, sp.clone());
    saver.save_state(sp.clone(), opt.get_state(), stop, &eval);

    run_reader(
        "tmp_test", hash_obj, hash_opt, hash_out, hash_cod, hash_inf, size,
    );
    let nstop = saver.load_stop(&sp, &cod).unwrap();
    let nopt = saver.load_optimizer(&sp, &cod).unwrap();
    let neval = saver.load_evaluate(&sp, &cod).unwrap();

    (nstop, opt.get_state(), nopt, neval)
}

pub fn run_reader<SInfo, Info>(
    path: &str,
    hash_obj: &HashMap<usize, Arc<Partial<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>>>,
    hash_opt: &HashMap<usize, Arc<Partial<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>>>,
    hash_out: &HashMap<
        usize,
        Arc<LinkedOutcome<OutExample, SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>>,
    >,
    hash_cod: &HashMap<
        usize,
        Arc<
            Computed<
                SId,
                sp_m_equal_allmsamp::_TantaleMixedObj,
                SingleCodomain<OutExample>,
                OutExample,
                SInfo,
            >,
        >,
    >,
    hash_inf: &HashMap<usize, Arc<Info>>,
    size: usize,
) where
    SInfo: SolInfo + CSVWritable<(), ()>,
    Info: OptInfo + CSVWritable<(), ()>,
{
    let cod = Arc::new(SingleCodomain::new(|x: &OutExample| x.mul6));

    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("evaluations"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");

    let mut size_obj: usize = 0;
    let mut size_opt: usize = 0;
    let mut size_out: usize = 0;
    let mut size_cod: usize = 0;

    // Check `Obj`
    let mut rdr = csv::Reader::from_path(path_obj).unwrap();
    for line in rdr.records() {
        let record = line.unwrap();
        size_obj += 1;
        let id: usize = record[0].parse().unwrap();
        let content = hash_obj.get(&id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content.get_id().id)]);
        let x_str: Vec<String> = content.get_x().iter().map(|x| format!("{}", x)).collect();
        str_content.extend(x_str);
        let info_str = hash_inf.get(&id).unwrap();
        str_content.extend(info_str.write(&()));
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match."
        );
    }

    // Check `Opt`
    let mut rdr = csv::Reader::from_path(path_opt).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        size_opt += 1;

        let id: usize = record[0].parse().unwrap();
        let content = hash_opt.get(&id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content.get_id().id)]);
        let x_str: Vec<String> = content.get_x().iter().map(|x| format!("{}", x)).collect();
        str_content.extend(x_str);
        let info_str = hash_inf.get(&id).unwrap().write(&());
        str_content.extend(info_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match."
        );
    }

    // Check `Codom`
    let mut rdr = csv::Reader::from_path(path_cod).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        size_cod += 1;

        let id: usize = record[0].parse().unwrap();
        let content = hash_cod.get(&id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", content.get_id().id)]);
        let cod_str: Vec<String> = cod.write(&content.get_y());
        str_content.extend(cod_str);

        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            record[0], record[1],
            "Wrong ID associated with the codomain"
        );
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match."
        );
    }

    // Check `Outcome`
    let mut rdr = csv::Reader::from_path(path_out).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        size_out += 1;

        let id: usize = record[0].parse().unwrap();
        let content = hash_out.get(&id).unwrap();
        let con: i64 = record[2].parse().unwrap();
        let true_con = match content.sol.get_x()[0] {
            sp_m_equal_allmsamp::_TantaleMixedObjTypeDom::Int(f) => f,
            _ => panic!("Wrong type for con2"),
        };
        let mut str_content: Vec<String> = Vec::from([format!("{}", content.sol.get_id().id)]);
        let out_str: Vec<String> = content.out.write(&());
        str_content.extend(out_str);

        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            record[0], record[1],
            "Wrong ID associated with the codomain"
        );
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match."
        );
        assert_eq!(
            con, true_con,
            "Record mismatch between 2nd element of Obj solutino and con2 Outcome."
        );
    }

    assert_eq!(
        size_obj, size,
        "Some solutions are missing within recorded obj save."
    );
    assert_eq!(
        size_opt, size,
        "Some solutions are missing within recorded opt save."
    );
    assert_eq!(
        size_cod, size,
        "Some solutions are missing within recorded cod save."
    );
    assert_eq!(
        size_out, size,
        "Some solutions are missing within recorded out save."
    );
}

fn test_csv_func() {
    let sp = Arc::new(sp_m_equal_allmsamp::get_searchspace());
    let cod = Arc::new(SingleCodomain::new(|x: &OutExample| x.mul6));

    let mut rs = RandomSearch::new(3);
    let mut stop = Calls::new(100);
    let mut saver = CSVSaver::new("tmp_test", true, true, true, 1);
    <CSVSaver as Saver<
        SId,
        Calls,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        SingleCodomain<OutExample>,
        OutExample,
        tantale_core::Sp<
            sp_m_equal_allmsamp::_TantaleMixedObj,
            sp_m_equal_allmsamp::_TantaleMixedObj,
        >,
        RandomSearch,
        EvalType<RSInfo,EmptyInfo>,
        Objective<sp_m_equal_allmsamp::_TantaleMixedObj,SingleCodomain<OutExample>,OutExample>,
    >>::init(&mut saver, sp.clone().as_ref(), cod.clone().as_ref());

    let mut hash_obj: HashMap<
        usize,
        Arc<Partial<SId, sp_m_equal_allmsamp::_TantaleMixedObj, EmptyInfo>>,
    > = HashMap::new();
    let mut hash_opt: HashMap<
        usize,
        Arc<Partial<SId, sp_m_equal_allmsamp::_TantaleMixedObj, EmptyInfo>>,
    > = HashMap::new();
    let mut hash_outcome: HashMap<usize, Arc<LinkedOutcome<OutExample, _, _, _>>> = HashMap::new();
    let mut hash_codom: HashMap<
        usize,
        Arc<Computed<_, _, SingleCodomain<OutExample>, OutExample, _>>,
    > = HashMap::new();

    let mut hash_info: HashMap<usize, Arc<RSInfo>> = HashMap::new();

    let (mut nstop, rstate, mut nopt,_):(_,_,_,EvalType<RSInfo,EmptyInfo>) = run_saver(
        &mut hash_obj,
        &mut hash_opt,
        &mut hash_outcome,
        &mut hash_codom,
        &mut hash_info,
        sp.clone(),
        cod.clone(),
        &mut rs,
        &mut stop,
        &mut saver,
        6,
    );
    let nstate = <RandomSearch as Optimizer<
        SId,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        SingleCodomain<OutExample>,
        _,
        Sp<sp_m_equal_allmsamp::_TantaleMixedObj, sp_m_equal_allmsamp::_TantaleMixedObj>,
    >>::get_state(&mut nopt);

    assert_eq!(
        stop.0, nstop.0,
        "States of threshold in Calls are different after loading."
    );
    assert_eq!(
        stop.1, nstop.1,
        "States of calls in Calls are different after loading."
    );
    assert_eq!(stop.0, 6, "Wrong number of Calls before loading");
    assert_eq!(nstop.0, 6, "Wrong number of Calls after loading");

    assert_eq!(
        rstate.batch, nstate.batch,
        "Batch fields for RSState are different after loading."
    );
    assert_eq!(
        rstate.iteration, nstate.iteration,
        "Iteration fields for RSState are different after loading."
    );

    let (nnstop, rstate, mut nopt, _) = run_saver(
        &mut hash_obj,
        &mut hash_opt,
        &mut hash_outcome,
        &mut hash_codom,
        &mut hash_info,
        sp.clone(),
        cod.clone(),
        &mut rs,
        &mut nstop,
        &mut saver,
        12,
    );
    let nstate = <RandomSearch as Optimizer<
        SId,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        SingleCodomain<OutExample>,
        _,
        Sp<sp_m_equal_allmsamp::_TantaleMixedObj, sp_m_equal_allmsamp::_TantaleMixedObj>,
    >>::get_state(&mut nopt);

    assert_eq!(
        nstop.0, nnstop.0,
        "States of threshold in Calls are different after loading."
    );
    assert_eq!(
        nstop.1, nnstop.1,
        "States of calls in Calls are different after loading."
    );
    assert_eq!(nstop.0, 12, "Wrong number of Calls before loading");
    assert_eq!(nnstop.0, 12, "Wrong number of Calls after loading");

    assert_eq!(
        rstate.batch, nstate.batch,
        "Batch fields for RSState are different after loading."
    );
    assert_eq!(
        rstate.iteration, nstate.iteration,
        "Iteration fields for RSState are different after loading."
    );
    drop(Cleaner {});
}

struct Cleaner;

impl Drop for Cleaner {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all("tmp_test");
    }
}

#[test]
fn test_csv() {
    drop(Cleaner {});
    test_csv_func();
}
