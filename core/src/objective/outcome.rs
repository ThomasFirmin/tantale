use std::collections::HashMap;

pub trait Outcome{}

pub type HashOut = HashMap<&'static str, f64>;
impl Outcome for HashOut{}