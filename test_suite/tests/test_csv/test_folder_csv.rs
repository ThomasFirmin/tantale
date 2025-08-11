use super::init_sp::sp_m_equal_allmsamp;

use tantale::core::{EmptyInfo, Searchspace, PartialSol, SId,Optimizer, Computed, Codomain, SingleCodomain};
use tantale_algos::random_search::RSState;
use tantale_core::optimizer::OptState;
use tantale_core::saver::CSVSaver;
use tantale_core::saver::Saver;
use tantale_algos::RandomSearch;
use tantale_core::ArcVecArc;
use tantale_core::{Domain,OptInfo,Outcome,SolInfo};
use tantale_core::stop::{Stop,Calls};

use std::fmt::{Display,Debug};
use std::sync::Arc;

mod infos {
    use tantale_core::{saver::CSVWritable, OptInfo, SolInfo};
    use tantale_macros::Outcome;
    #[derive(Outcome)]
    pub struct OutExample {
        pub fid2: f64,
        pub con3: f64,
        pub con4: f64,
        pub con5: f64,
        pub mul6: f64,
        pub mul7: f64,
        pub mul8: f64,
        pub mul9: f64,
        pub tvec: Vec<f64>,
    }

    pub fn get_out() -> OutExample {
        OutExample {
            fid2: 2.2,
            con3: 3.3,
            con4: 4.4,
            con5: 5.5,
            mul6: 6.6,
            mul7: 7.7,
            mul8: 8.8,
            mul9: 9.9,
            tvec: Vec::from([1.1, 2.2, 3.3]),
        }
    }

    pub struct SomeSInfo{
        pub sinfo1 : f64,
        pub sinfo2 : f64,
        pub sinfo3 : i32,
    }

    impl SolInfo for SomeSInfo{}
    impl CSVWritable<()> for SomeSInfo{
        fn header(&self) -> Vec<String> {
            Vec::from([String::from("sinfo1"),String::from("sinfo2"),String::from("sinfo3")])
        }
    
        fn write(&self, _comp: &()) -> Vec<String> {
            Vec::from([self.sinfo1.to_string(),self.sinfo3.to_string(),self.sinfo3.to_string()])
        }
    }

    pub fn get_sinfo() -> SomeSInfo{
        SomeSInfo { sinfo1: 1.1, sinfo2: 2.2, sinfo3: -3 }
    }

    pub struct SomeOInfo{
        pub info1 : f64,
        pub info2 : u32,
    }

    impl OptInfo for SomeOInfo{}
    impl CSVWritable<()> for SomeOInfo{
        fn header(&self) -> Vec<String> {
            Vec::from([String::from("info1"),String::from("info2")])
        }
    
        fn write(&self, _comp: &()) -> Vec<String> {
            Vec::from([self.info1.to_string(),self.info2.to_string()])
        }
    }

    pub fn get_oinfo() -> SomeOInfo{
        SomeOInfo { info1: 1.1, info2: 2 }
    }

}


pub fn run_saver<Scp, Op, St, Sv, Obj, Opt, Out, Cod, Info, SInfo, State>(
    mut sp: Scp,
    mut opt: Op,
    mut stop: St,
    mut saver: Sv,
)
where
    Scp: Searchspace<SId, PartialSol<SId, Obj, SInfo>, PartialSol<SId, Opt, SInfo>, Obj, Opt, SInfo>,
    Op: Optimizer<SId, PartialSol<SId, Obj, SInfo>, PartialSol<SId, Opt, SInfo>, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    St: Stop,
    Sv: Saver<SId, St, PartialSol<SId, Obj, SInfo>, PartialSol<SId, Opt, SInfo>, Obj, Opt, SInfo, Cod, Out, Scp, Info, State>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Out: Outcome,
    Cod: Codomain<Out>,
    Info: OptInfo,
    SInfo: SolInfo,
    State:OptState,
{
    use infos::OutExample;

    let sp = Arc::new(sp);

    let outcome = infos::get_out();
    let cod = SingleCodomain::new(|x:&OutExample| x.mul6);
    let codel = Arc::new(cod.get_elem(&outcome));

    let info = Arc::new(infos::get_oinfo());
    let sinfo = infos::get_sinfo();
    let mut rng = rand::rng();

    let samples = opt.first_step(sp.clone());

    let computed: (Vec<_>, Vec<_>) = samples.0.clone().into_iter().zip(samples.1.clone().into_iter()).map(
        |(a,b)|
        sp.computed(a.clone(), b.clone(), codel.clone())
    ).unzip();

    let samples_obj:Arc<Vec<Arc<PartialSol<SId,_,_>>>>  = sp.vec_sample_obj(
        Some(&mut rng), 
        10,
        Arc::new(sinfo)
    );
    let samples_opt = sp.vec_onto_opt(samples_obj.clone());

    saver.save_partial(samples_obj.clone(), samples_opt.clone(), sp.clone() , info.clone());
}


#[test]
fn test_csv(){
    let mut sp = sp_m_equal_allmsamp::get_searchspace();
    let mut rs = RandomSearch::new(3);
    let mut stop = Calls::new(100);
    let mut saver = CSVSaver::new("./test", true, true, true, 0);

    run_saver(sp,rs,stop,saver);
}
