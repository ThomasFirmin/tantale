use std::collections::HashMap;

pub trait Outcome{}

pub type HashOut<T> = HashMap<&str, T>;
impl <T> Outcome for HashOut<T>{}