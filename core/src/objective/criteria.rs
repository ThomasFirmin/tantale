use std::collections::HashMap;

/// [`Criteria`] is a trait defining what to extract from the [`HashMap`] output of the
/// [objective function](tantale::core::objective;;obj::Objective::compute), and how
/// to process this extracted output.
/// By default we always consider a **maximization** problem whithin the [`Optimizer`].
pub trait Criteria{
    fn extract(&self, out : &HashMap<&'static str,f64>) -> f64;
}

/// A [`Criteria`] describing the extracted value as something to be maximized.
/// $$f(x) = x$$
/// 
/// # Note
/// Simply copy the extracted value to the output. Indeed, by default an [`Optimizer`] solves
/// a **maximization** problem.
pub struct Max{
    pub key : &'static str,
}

impl Max{
    pub fn new(key:&'static str) -> Self{
        Max { key }
    }
}

impl Criteria for Max {
    fn extract(&self, out : &HashMap<&'static str,f64>) -> f64 {
        *out.get(self.key).unwrap()
    }
}

/// A [`Criteria`] describing the extracted value as something to be minimized.
/// $$f(x) = -x$$
pub struct Min{
    pub key : &'static str,
}

impl Min{
    pub fn new(key:&'static str) -> Self{
        Min { key }
    }
}

impl Criteria for Min {
    fn extract(&self, out : &HashMap<&'static str,f64>) -> f64 {
        -*out.get(self.key).unwrap()
    }
}

/// A [`Criteria`] applying to the extracted value a function `lambda`.
/// $$f(x) = \lambda (x)$$
/// 
/// # Note
/// The output value will be maximized. Indeed, by default an [`Optimizer`] solves
/// a **maximization** problem.
pub struct Lambda{
    pub key : &'static str,
    pub lambda : fn(f64) -> f64,
}

impl Lambda{
    pub fn new(key:&'static str, lambda : fn(f64) -> f64) -> Self{
        Lambda { key, lambda }
    }
}

impl Criteria for Lambda {
    fn extract(&self, out : &HashMap<&'static str,f64>) -> f64 {
        (self.lambda)(*out.get(self.key).unwrap())
    }
}