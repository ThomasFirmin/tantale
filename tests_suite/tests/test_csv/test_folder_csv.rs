use super::init_sp::sp_m_equal_allmsamp::get_searchspace;

use csv::StringRecord;
use tantale::algos::BatchRandomSearch;
use tantale::algos::random_search;
use tantale::core::domain::onto::{LinkObj, LinkOpt, LinkTyObj, LinkTyOpt};
use tantale::core::objective::FuncWrapper;
use tantale::core::objective::Step;
use tantale::core::optimizer::opt::CompBatch;
use tantale::core::recorder::csv::{InfoCSVWrite, ScpCSVWrite, SolCSVWrite};
use tantale::core::searchspace::CompShape;
use tantale::core::solution::shape::{SolObj, SolOpt};
use tantale::core::solution::{
    HasId, HasSolInfo, HasUncomputed, HasY, IntoComputed, SolutionShape, Uncomputed,
};
use tantale::core::{BaseSol, BatchRecorder, Computed, EmptyInfo, NoDomain, Objective};
use tantale::core::{
    Codomain, FolderConfig, Mixed, MixedTypeDom, SId, Searchspace, Solution, Sp,
    optimizer::opt::BatchOptimizer,
    recorder::csv::{CSVRecorder, CSVWritable},
    solution::{Batch, HasInfo, OutBatch},
    stop::{Calls, Stop},
};

use std::{path::Path, sync::Arc};

mod infos {
    use serde::{Deserialize, Serialize};
    use tantale::macros::{CSVWritable, Outcome};

    #[derive(Outcome, Debug, Serialize, Deserialize, CSVWritable)]
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

pub fn run_recorder<Scp, Op, St, Rec, Fn, PSol>(
    sp: &Scp,
    cod: &Op::Cod,
    opt: &mut Op,
    stop: &mut St,
    recorder: &mut Rec,
    size: usize,
) where
    PSol: Uncomputed<SId, LinkOpt<Scp>, Op::SInfo, Raw = Arc<[LinkTyOpt<Scp>]>>,
    PSol::Twin<LinkObj<Scp>>: Uncomputed<SId, LinkObj<Scp>, Op::SInfo, Raw = Arc<[LinkTyObj<Scp>]>>,
    Scp: Searchspace<PSol, SId, Op::SInfo, Obj = Mixed>
        + SolCSVWrite<PSol, SId, Op::SInfo>
        + ScpCSVWrite<PSol, SId, Op::SInfo, Op::Cod, OutExample>
        + Send
        + Sync,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, OutExample>: SolutionShape<
            SId,
            Op::SInfo,
            SolObj = Computed<
                PSol::Twin<LinkObj<Scp>>,
                SId,
                LinkObj<Scp>,
                Op::Cod,
                OutExample,
                Op::SInfo,
            >,
            SolOpt = Computed<PSol, SId, LinkOpt<Scp>, Op::Cod, OutExample, Op::SInfo>,
        > + HasY<Op::Cod, OutExample>
        + InfoCSVWrite<SId, Op::SInfo>
        + HasY<Op::Cod, OutExample>
        + Send
        + Sync,
    SolObj<CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, OutExample>, SId, Op::SInfo>:
        HasUncomputed<SId, Scp::Obj, Op::SInfo, Uncomputed = SolObj<Scp::SolShape, SId, Op::SInfo>>,
    SolOpt<CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, OutExample>, SId, Op::SInfo>:
        HasUncomputed<SId, Scp::Opt, Op::SInfo, Uncomputed = SolOpt<Scp::SolShape, SId, Op::SInfo>>,
    Op: BatchOptimizer<PSol, SId, LinkOpt<Scp>, OutExample, Scp, Fn>,
    St: Stop + Send + Sync,
    Rec: BatchRecorder<PSol, SId, OutExample, Scp, Op, Fn>,
    Fn: FuncWrapper<Arc<[LinkTyObj<Scp>]>>,
    Op::Cod: CSVWritable<Op::Cod, <Op::Cod as Codomain<OutExample>>::TypeCodom> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
{
    let batch = opt.first_step(sp);
    let mut obatch = OutBatch::empty(batch.info());
    let mut cbatch = Batch::empty(batch.info());
    batch.into_iter().for_each(|pair| {
        let id = pair.id();
        let aelem = pair.get_sobj().get_x()[0].clone();
        let aelem = match aelem {
            MixedTypeDom::Int(ae) => ae,
            _ => panic!("Should be a Int."),
        };
        let outcome = infos::get_out(id.id, aelem);
        let codel = Arc::new(cod.get_elem(&outcome));
        let pair = pair.into_computed(codel);
        cbatch.add(pair);
        obatch.add((id, outcome));
    });
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));

    recorder.save(&cbatch, &obatch, sp, cod);

    let batch = opt.first_step(sp);
    let mut tobatch = OutBatch::empty(batch.info());
    let mut tcbatch = Batch::empty(batch.info());
    batch.into_iter().for_each(|pair| {
        let id = pair.id();
        let aelem = pair.get_sobj().get_x()[0].clone();
        let aelem = match aelem {
            MixedTypeDom::Int(ae) => ae,
            _ => panic!("Should be a Int."),
        };
        let outcome = infos::get_out(id.id, aelem);
        let codel = Arc::new(cod.get_elem(&outcome));
        let pair = pair.into_computed(codel);
        tcbatch.add(pair);
        tobatch.add((id, outcome));
    });
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));
    stop.update(tantale::core::stop::ExpStep::Distribution(Step::Evaluated));

    recorder.save(&tcbatch, &tobatch, sp, cod);
    run_reader::<Scp, Op, St, CSVRecorder, Fn, PSol>(
        "tmp_test", cbatch, obatch, tcbatch, tobatch, cod, size,
    );
}

pub fn run_reader<Scp, Op, St, Rec, Fn, PSol>(
    path: &str,
    cbatch: CompBatch<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, OutExample>,
    obatch: OutBatch<SId, Op::Info, OutExample>,
    tcbatch: CompBatch<SId, Op::SInfo, Op::Info, Scp, PSol, Op::Cod, OutExample>,
    tobatch: OutBatch<SId, Op::Info, OutExample>,
    cod: &Op::Cod,
    size: usize,
) where
    PSol: Uncomputed<SId, LinkOpt<Scp>, Op::SInfo, Raw = Arc<[LinkTyOpt<Scp>]>>,
    PSol::Twin<LinkObj<Scp>>: Uncomputed<SId, LinkObj<Scp>, Op::SInfo, Raw = Arc<[LinkTyObj<Scp>]>>,
    Scp: Searchspace<PSol, SId, Op::SInfo, Obj = Mixed>
        + SolCSVWrite<PSol, SId, Op::SInfo>
        + ScpCSVWrite<PSol, SId, Op::SInfo, Op::Cod, OutExample>
        + Send
        + Sync,
    CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, OutExample>: SolutionShape<
            SId,
            Op::SInfo,
            SolObj = Computed<
                PSol::Twin<LinkObj<Scp>>,
                SId,
                LinkObj<Scp>,
                Op::Cod,
                OutExample,
                Op::SInfo,
            >,
            SolOpt = Computed<PSol, SId, LinkOpt<Scp>, Op::Cod, OutExample, Op::SInfo>,
        > + HasY<Op::Cod, OutExample>
        + InfoCSVWrite<SId, Op::SInfo>
        + HasY<Op::Cod, OutExample>
        + Send
        + Sync,
    SolObj<CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, OutExample>, SId, Op::SInfo>:
        HasUncomputed<SId, Scp::Obj, Op::SInfo, Uncomputed = SolObj<Scp::SolShape, SId, Op::SInfo>>,
    SolOpt<CompShape<Scp, PSol, SId, Op::SInfo, Op::Cod, OutExample>, SId, Op::SInfo>:
        HasUncomputed<SId, Scp::Opt, Op::SInfo, Uncomputed = SolOpt<Scp::SolShape, SId, Op::SInfo>>,
    Op: BatchOptimizer<PSol, SId, LinkOpt<Scp>, OutExample, Scp, Fn>,
    St: Stop + Send + Sync,
    Rec: BatchRecorder<PSol, SId, OutExample, Scp, Op, Fn>,
    Fn: FuncWrapper<Arc<[LinkTyObj<Scp>]>>,
    Op::Cod: CSVWritable<Op::Cod, <Op::Cod as Codomain<OutExample>>::TypeCodom> + Send + Sync,
    Op::Info: CSVWritable<(), ()> + Send + Sync,
    Op::SInfo: CSVWritable<(), ()> + Send + Sync,
{
    let true_path = Path::new(path);
    let eval_path = true_path.join(Path::new("recorder"));
    let path_obj = eval_path.join("obj.csv");
    let path_opt = eval_path.join("opt.csv");
    let path_out = eval_path.join("out.csv");
    let path_cod = eval_path.join("cod.csv");
    let path_info = eval_path.join("info.csv");

    let mut size_all: usize = 0;
    let mut size_out: usize = 0;

    // Check `Obj`, `Opt`, `Codom`
    let mut rdr_obj = csv::Reader::from_path(path_obj).unwrap();
    let mut rdr_opt = csv::Reader::from_path(path_opt).unwrap();
    let mut rdr_cod = csv::Reader::from_path(path_cod).unwrap();
    let mut rdr_info = csv::Reader::from_path(path_info).unwrap();

    let records = rdr_obj
        .records()
        .zip(rdr_opt.records())
        .zip(rdr_cod.records())
        .zip(rdr_info.records());

    let info = cbatch.info.clone();
    let tinfo = tcbatch.info.clone();
    let size_comp = cbatch.size();
    let size_tcomp = tcbatch.size();
    let size_cout = obatch.size();
    let size_tcout = tobatch.size();

    for (((line_obj, line_opt), line_cod), line_info) in records {
        let (computed_batch, computed_info) = if size_all < cbatch.size() {
            (&cbatch, info.clone())
        } else {
            (&tcbatch, tinfo.clone())
        };
        size_all += 1;

        let record = line_obj.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let pair = iter_cbatch.find(|p| p.id().id == id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.id().id)]);
        let x_str: Vec<String> = pair
            .get_sobj()
            .get_x()
            .iter()
            .map(|x| format!("{}", x))
            .collect();
        str_content.extend(x_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Obj."
        );

        let record = line_opt.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let pair = iter_cbatch.find(|p| p.id().id == id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.get_sopt().id().id)]);
        let x_str: Vec<String> = pair
            .get_sopt()
            .get_x()
            .iter()
            .map(|x| format!("{}", x))
            .collect();
        str_content.extend(x_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Opt."
        );

        let record = line_cod.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let pair = iter_cbatch.find(|p| p.id().id == id).unwrap();
        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.get_sobj().id().id)]);
        let cod_str: Vec<String> = cod.write(&pair.get_sobj().y());
        str_content.extend(cod_str);
        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Codomain."
        );

        let record = line_info.unwrap();
        let id: usize = record[0].parse().unwrap();
        let mut iter_cbatch = computed_batch.into_iter();
        let pair = iter_cbatch.find(|p| p.id().id == id).unwrap();

        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.get_sobj().id().id)]);
        let sinfo_str = pair.get_sobj().sinfo().write(&());
        str_content.extend(sinfo_str);
        let info_str = computed_info.write(&());
        str_content.extend(info_str);

        let record_str: Vec<String> = record.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            str_content, record_str,
            "True baseline and CSV record do not match for Info."
        );
    }

    // Check `Outcome`
    let mut rdr = csv::Reader::from_path(path_out).unwrap();
    let mut record = StringRecord::new();
    while rdr.read_record(&mut record).unwrap() {
        let computed_batch = if size_out < cbatch.size() {
            &cbatch
        } else {
            &tcbatch
        };
        let output_batch = if size_out < obatch.size() {
            &obatch
        } else {
            &tobatch
        };

        size_out += 1;

        let id: usize = record[0].parse().unwrap();
        let con: i64 = record[2].parse().unwrap();
        let mut iter_obatch = output_batch.into_iter();
        let mut iter_cbatch = computed_batch.into_iter();
        let content = iter_obatch
            .find(|(sid, out)| (sid.id == id) && (out.fid2 == id))
            .unwrap();
        let pair = iter_cbatch.find(|p| p.id().id == id).unwrap();
        let true_con = match pair.get_sobj().sol.get_x()[0] {
            MixedTypeDom::Int(f) => f,
            _ => panic!("Wrong type for con2"),
        };
        let mut str_content: Vec<String> = Vec::from([format!("{}", pair.get_sobj().sol.id().id)]);
        let out_str: Vec<String> = content.1.write(&());
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

    assert_eq!(size_all, size, "Some solutions are missing.");
    assert_eq!(
        size_all,
        size_comp + size_tcomp,
        "Some solutions are missing within computed batch."
    );
    assert_eq!(
        size_all, size_out,
        "Some solutions are missing between out and all."
    );
    assert_eq!(
        size_out,
        size_cout + size_tcout,
        "Some solutions are missing within recorded out batch."
    );
}

fn test_csv_func() {
    let sp = get_searchspace();
    let cod = random_search::codomain(|x: &OutExample| x.mul6);

    let mut rs = BatchRandomSearch::new(3);
    let mut stop = Calls::new(100);
    let config = Arc::new(FolderConfig::new("tmp_test"));
    let mut recorder = CSVRecorder::new(config, true, true, true, true).unwrap();
    <CSVRecorder as BatchRecorder<
        BaseSol<SId, Mixed, EmptyInfo>,
        SId,
        OutExample,
        Sp<Mixed, NoDomain>,
        BatchRandomSearch,
        Objective<Arc<[MixedTypeDom]>, OutExample>,
    >>::init(&mut recorder, &sp, &cod);

    run_recorder(&sp, &cod, &mut rs, &mut stop, &mut recorder, 6);
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
    drop(Cleaner {});
}
