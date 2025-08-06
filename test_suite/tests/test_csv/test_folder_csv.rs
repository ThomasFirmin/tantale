use super::init_sp::*;
use super::init_cod::*;
use tantale::core::saver::CSVLeftRight;
use tantale::core::{EmptyInfo, Searchspace, Solution, PartialSol, SId};

use std::sync::Arc;
use paste::paste;

mod outcome {
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

    pub fn get_struct() -> OutExample {
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

#[test]
fn test_csv(){
    let sp = sp_m_equal_allmsamp::get_searchspace();
    let outcome = outcome::get_struct();
    let cod = get_elemfidelconst();
}
