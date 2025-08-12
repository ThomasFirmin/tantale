use super::init_sp::sp_m_equal_allmsamp;

use tantale::core::{Searchspace, PartialSol, SId,Optimizer, Codomain, SingleCodomain};
use tantale_core::optimizer::OptState;
use tantale_core::saver::CSVSaver;
use tantale_core::saver::Saver;
use tantale_algos::RandomSearch;
use tantale_core::LinkedOutcome;
use tantale_core::{Domain,OptInfo,SolInfo};
use tantale_core::stop::{Stop,Calls};

use std::fmt::{Display,Debug};
use std::sync::Arc;

mod infos {
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

}

use infos::OutExample;

pub fn run_saver<Scp, Op, St, Sv, Obj, Opt, Info, SInfo, State>(
    sp: Scp,
    mut opt: Op,
    stop: St,
    mut saver: Sv,
)
where
    Scp: Searchspace<SId, PartialSol<SId, Obj, SInfo>, PartialSol<SId, Opt, SInfo>, Obj, Opt, SInfo>,
    Op: Optimizer<SId, PartialSol<SId, Obj, SInfo>, PartialSol<SId, Opt, SInfo>, Obj, Opt, SInfo, SingleCodomain<OutExample>, OutExample, Scp, Info, State>,
    St: Stop,
    Sv: Saver<SId, St, PartialSol<SId, Obj, SInfo>, PartialSol<SId, Opt, SInfo>, Obj, Opt, SInfo, SingleCodomain<OutExample>, OutExample, Scp, Info, State>,
    Obj: Domain + Clone + Display + Debug,
    Opt: Domain + Clone + Display + Debug,
    Info: OptInfo,
    SInfo: SolInfo,
    State:OptState,
{
    saver.init();

    let sp = Arc::new(sp);
    let stop = Arc::new(stop);

    let outcome = Arc::new(infos::get_out());
    let cod  = SingleCodomain::new(|x:&OutExample| x.mul6);
    let codel = Arc::new(cod.get_elem(&outcome));

    let (sobj,sopt,infos) = opt.first_step(sp.clone());
    let infos = Arc::new(infos);
    let computed: (Vec<_>, Vec<_>, Vec<_>) = sobj.iter().zip(sopt.iter()).map(
        |(a,b)|
        {
            let r = sp.computed::<SingleCodomain<_>,OutExample>(a.clone(), b.clone(), codel.clone());

            let linked = LinkedOutcome::new(outcome.clone(), a.clone());
            (r.0,r.1,linked)
        }
    ).collect();

    saver.save_partial(sobj.clone(), sopt.clone(), sp.clone() , infos.clone());
    saver.save_codom(Arc::new(computed.0));
    saver.save_out(computed.2);
    saver.save_state(sp.clone(), opt.get_state(), &stop);


    
    saver.clean();
}


#[test]
fn test_csv(){
    let sp = sp_m_equal_allmsamp::get_searchspace();
    let rs = RandomSearch::new(3);
    let stop = Calls::new(100);
    let saver = CSVSaver::new("tmp_test", true, true, true, 1);

    run_saver(sp,rs,stop,saver);
}
