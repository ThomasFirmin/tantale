use serde::{Deserialize, Serialize};
use tantale::{core::SolInfo, macros::{Recordable, Schema}};

#[derive(Schema, Recordable, Debug, Serialize, Deserialize)]
pub struct TestSInfo {
    pub info: f64,
}
impl SolInfo for TestSInfo {}

pub fn get_sinfo() -> TestSInfo {
    TestSInfo { info: 42.0 }
}
