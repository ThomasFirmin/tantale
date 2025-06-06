//! An [`Outcome`](tantale::core::Outcome) describes the output of
//! the function to be maximized. This output may contain the values
//! to be optimized, constraints, fidilities, and other information
//! linked to the evaluation (e.g. computation time).


use std::collections::HashMap;

pub trait Outcome{}

pub type HashOut = HashMap<&'static str, f64>;
impl Outcome for HashOut{}