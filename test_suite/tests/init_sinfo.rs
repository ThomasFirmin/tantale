use tantale::core::SolInfo;

pub struct  TestSInfo{
    pub info : f64,
}

impl SolInfo for TestSInfo{}

pub fn get_sinfo()->TestSInfo{
    TestSInfo { info: 42.0}
}