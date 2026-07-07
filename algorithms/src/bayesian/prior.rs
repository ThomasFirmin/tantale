use serde::{Deserialize, Serialize};


/// A Beta prior distribution with parameters `alpha` and `beta`.
/// The Beta distribution is defined on the interval [0, 1] and is commonly used
/// as a prior distribution for probabilities in Bayesian statistics.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct BetaPrior(pub f64, pub f64, pub f64);

impl BetaPrior {
    /// Creates a new `BetaPrior` with the given `alpha` and `beta` parameters.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The alpha parameter of the Beta distribution (must be > 0).
    /// * `beta` - The beta parameter of the Beta distribution (must be > 0).
    ///
    /// # Panics
    ///
    /// This function will panic if either `alpha` or `beta` is less than or equal to 0.
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0, "Alpha must be greater than 0");
        assert!(beta > 0.0, "Beta must be greater than 0");
        BetaPrior(alpha, beta, alpha + beta)
    }

    /// Returns the `alpha` parameter of the Beta distribution.
    pub fn alpha(&self) -> f64 {
        self.0
    }

    /// Returns the `beta` parameter of the Beta distribution.
    pub fn beta(&self) -> f64 {
        self.1
    }

    /// Returns the sum of the `alpha` and `beta` parameters of the Beta distribution.
    pub fn sum(&self) -> f64 {
        self.2
    }
}
