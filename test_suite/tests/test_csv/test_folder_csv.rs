use super::init_sp::sp_m_equal_allmsamp;

use csv::StringRecord;
use tantale::core::{Codomain, Computed, Optimizer, PartialSol, SId, Searchspace, SingleCodomain};
use tantale_algos::random_search::RSState;
use tantale_algos::RandomSearch;
use tantale_core::optimizer::OptState;
use tantale_core::saver::CSVSaver;
use tantale_core::saver::CSVWritable;
use tantale_core::saver::Saver;
use tantale_core::stop::{Calls, Stop};
use tantale_core::LinkedOutcome;
use tantale_core::Solution;
use tantale_core::{OptInfo, SolInfo};

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

mod infos {
    use tantale_macros::Outcome;
    #[derive(Outcome)]
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

pub fn run_saver<Scp, Op, St, Sv, Info, SInfo, State>(
    sp: Scp,
    opt: &mut Op,
    stop: &St,
    mut saver: Sv,
) -> (Sv,St,State)
where
    Scp: Searchspace<
        SId,
        PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
        PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        SInfo,
    >,
    Op: Optimizer<
        SId,
        PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
        PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        SInfo,
        SingleCodomain<OutExample>,
        OutExample,
        Scp,
        Info,
        State,
    >,
    St: Stop,
    Sv: Saver<
        SId,
        St,
        PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
        PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        sp_m_equal_allmsamp::_TantaleMixedObj,
        SInfo,
        SingleCodomain<OutExample>,
        OutExample,
        Scp,
        Info,
        State,
    >,
    Info: OptInfo + CSVWritable<()>,
    SInfo: SolInfo + CSVWritable<()>,
    State: OptState,
{
    saver.init();

    let mut hash_obj: HashMap<
        usize,
        Arc<PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>>,
    > = HashMap::new();
    let mut hash_opt: HashMap<
        usize,
        Arc<PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>>,
    > = HashMap::new();
    let mut hash_outcome: HashMap<
        usize,
        Arc<
            LinkedOutcome<
                OutExample,
                PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
                _,
                _,
                _,
            >,
        >,
    > = HashMap::new();
    let mut hash_codom: HashMap<
        usize,
        Arc<
            Computed<
                _,
                PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
                _,
                SingleCodomain<OutExample>,
                OutExample,
                _,
            >,
        >,
    > = HashMap::new();

    let mut hash_info: HashMap<usize,Arc<Info>> = HashMap::new();

    let sp = Arc::new(sp);
    let stop = Arc::new(stop);

    let (sobj, sopt, infos) = opt.first_step(sp.clone());
    let infos = Arc::new(infos);
    let (cobj,copt,vinfos): (Vec<_>, Vec<_>, Vec<_>) = sobj
        .iter()
        .zip(sopt.iter())
        .map(|(a, b)| {
            let id = a.get_id().id;
            let aelem = a.get_x()[0];
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
            let linked = Arc::new(LinkedOutcome::new(outcome.clone(), a.clone()));
            hash_obj.insert(id, a.clone());
            hash_opt.insert(id, b.clone());
            hash_outcome.insert(id, linked.clone());
            hash_codom.insert(id, r.0.clone());
            hash_info.insert(id,infos.clone());
            (r.0, r.1, linked)
        })
        .collect();

    let (cobj,copt,vinfos) = (Arc::new(cobj),Arc::new(copt),Arc::new(vinfos));
    saver.save_partial(sobj.clone(), sopt.clone(), sp.clone(), infos.clone());
    saver.save_codom(cobj.clone());
    saver.save_out(vinfos.clone());
    saver.save_state(sp.clone(), opt.get_state(), &stop);

    let (sobj, sopt, infos) = opt.step((cobj.clone(),copt.clone()),sp.clone());
    let infos = Arc::new(infos);
    let computed: (Vec<_>, Vec<_>, Vec<_>) = sobj
        .iter()
        .zip(sopt.iter())
        .map(|(a, b)| {
            let id = a.get_id().id;
            let aelem = a.get_x()[0];
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
            let linked = Arc::new(LinkedOutcome::new(outcome.clone(), a.clone()));
            hash_obj.insert(id, a.clone());
            hash_opt.insert(id, b.clone());
            hash_outcome.insert(id, linked.clone());
            hash_codom.insert(id, r.0.clone());
            hash_info.insert(id,infos.clone());
            (r.0, r.1, linked)
        })
        .collect();

    saver.save_partial(sobj.clone(), sopt.clone(), sp.clone(), infos.clone());
    saver.save_codom(Arc::new(computed.0));
    saver.save_out(Arc::new(computed.2));
    saver.save_state(sp.clone(), opt.get_state(), &stop);

    run_reader("tmp_test", hash_obj, hash_opt, hash_outcome, hash_codom, hash_info);
    saver.clean();
    Sv::load("tmp_test").unwrap()
}
pub fn run_reader<SInfo,Info>(
    path: &str,
    hash_obj: HashMap<usize, Arc<PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>>>,
    hash_opt: HashMap<usize, Arc<PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>>>,
    hash_out: HashMap<
        usize,
        Arc<
            LinkedOutcome<
                OutExample,
                PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
                SId,
                sp_m_equal_allmsamp::_TantaleMixedObj,
                SInfo,
            >,
        >,
    >,
    hash_cod: HashMap<
        usize,
        Arc<
            Computed<
                SId,
                PartialSol<SId, sp_m_equal_allmsamp::_TantaleMixedObj, SInfo>,
                sp_m_equal_allmsamp::_TantaleMixedObj,
                SingleCodomain<OutExample>,
                OutExample,
                SInfo,
            >,
        >,
    >,
    hash_inf: HashMap<usize,Arc<Info>>
) where
    SInfo: SolInfo + CSVWritable<()>,
    Info:OptInfo + CSVWritable<()>,
{
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("evaluations"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    println!("{:?}",path_obj);

    let mut size_obj: usize = 0;
    let mut size_opt: usize = 0;
    let mut size_out: usize = 0;
    let mut size_cod: usize = 0;

    // Check `Obj`
    let mut rdr = csv::Reader::from_path(path_obj).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
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
        let cod_str: Vec<String> = content.get_y().write(&());
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
        let con:i64 = record[2].parse().unwrap();
        let true_con = match content.sol.get_x()[0]{
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
        assert_eq!(con, true_con, "Record mismatch between 2nd element of Obj solutino and con2 Outcome.");
    }

    assert!(
        (size_obj == 6 && size_opt == 6 && size_out == 6 && size_cod == 6),
        "Some solutions are missing within recorded infos."
    )
}

#[test]
fn test_csv() {
    let sp = sp_m_equal_allmsamp::get_searchspace();
    let mut rs = RandomSearch::new(3);
    let stop = Calls::new(100);
    let saver = CSVSaver::new("tmp_test", true, true, true, 1);

    let (nsaver, nstop, nstate) = run_saver(sp, &mut rs, &stop, saver);
    assert_eq!(stop.0,nstop.0,"States of threshold in Calls are different after loading.");
    assert_eq!(stop.1,nstop.1,"States of calls in Calls are different after loading.");

    let rstate : &RSState = rs.get_state();
    assert_eq!(rstate.batch, nstate.batch, "Batch fields for RSState are different after loading.");
    assert_eq!(rstate.iteration, nstate.iteration, "Iteration fields for RSState are different after loading.");

}
