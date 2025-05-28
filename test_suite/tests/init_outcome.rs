use tantale::core::{Outcome, HashOut};

pub fn get_hash() -> HashOut {
    HashOut::from([
        ("obj1", 1.0),
        ("fid2", 2.0),
        ("con3", 3.0),
        ("con4", 4.0),
        ("con5", 5.0),
        ("mul6", 6.0),
        ("mul7", 7.0),
        ("mul8", 8.0),
        ("mul9", 9.0),
        ("more", 10.0),
        ("info", 11.0),
    ]
    )
}

pub struct  OutExample{
    pub obj1 : f64,
    pub fid2 : f64,
    pub con3 : f64,
    pub con4 : f64,
    pub con5 : f64,
    pub mul6 : f64,
    pub mul7 : f64,
    pub mul8 : f64,
    pub mul9 : f64,
    pub more : f64,
    pub info : f64,
}

impl Outcome for OutExample{}

pub fn get_struct()->OutExample{
    OutExample{
        obj1: 1.0,
        fid2: 2.0,
        con3: 3.0,
        con4: 4.0,
        con5: 5.0,
        mul6: 6.0,
        mul7: 7.0,
        mul8: 8.0,
        mul9: 9.0,
        more: 10.0,
        info: 11.0,
    }
}