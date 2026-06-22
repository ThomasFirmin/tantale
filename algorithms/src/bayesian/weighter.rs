use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
/// A struct to hold the weights for a set of points, as well as the prior weight.
pub struct PointWeights {
    pub weights: Vec<f64>,
    pub prior_weight: f64,
}

impl PointWeights {
    /// Computes the total weight by summing the individual weights and adding the prior weight.
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum::<f64>() + self.prior_weight
    }
}

#[derive(Serialize, Deserialize, Debug)]
/// A struct to hold the weights for the good and bad sets, as well as the prior weights for both sets.
pub struct TPEWeights {
    pub good: PointWeights,
    pub bad: PointWeights,
}

/// The [`Weighter`] trait defines a method for weighting the good and bad sets within the TPE algorithm.
pub trait Weighter<T>
where
    Self: Serialize + for<'de> Deserialize<'de>,
    T: Serialize + for<'de> Deserialize<'de>,
{
    /// Computes the weights for the good and bad sets.
    ///
    /// # Parameters
    /// - `good`: A slice of points in the good set.
    /// - `bad`: A slice of points in the bad set.
    ///
    /// # Returns
    /// A [`TPEWeights`] struct containing the weights for the good and bad sets, as well as the prior weights for both sets.
    fn weight(&self, good: &[&T], bad: &[&T]) -> TPEWeights;
}

/// A simple uniform [`Weighter`].
/// Each point $s$ in the $\\texttt{good}$ and $\\texttt{bad}$ sets is assigned the same weight.
/// The weight is inversely proportional to the size of the respective set.
///
/// $$
/// w_s = \\begin{cases}
/// \\frac{1}{|\\texttt{good} + \\texttt{prior}|} & \\text{if } s \\in \\texttt{good} \\\\
/// \\frac{1}{|\\texttt{bad} + \\texttt{prior}|} & \\text{if } s \\in \\texttt{bad}
/// \end{cases}\enspace\\text{,}
/// $$
/// there is a $+1$ for the prior weight.
/// The prior weight for each set is computed as:
/// $$
/// w_0 = \\frac{\\texttt{prior}}{|\\texttt{set}| + \\texttt{prior}}
/// $$
#[derive(Serialize, Deserialize, Debug)]
pub struct UniformWeighter(f64);

impl UniformWeighter {
    /// Creates a new [`UniformWeighter`] with a default prior weight of 1.0.
    pub fn new(prior: f64) -> Self {
        UniformWeighter(prior)
    }
}

/// The default implementation of [`UniformWeighter`] uses a prior weight of 1.0.
impl Default for UniformWeighter {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl<T> Weighter<T> for UniformWeighter
where
    T: Serialize + for<'de> Deserialize<'de>,
{
    fn weight(&self, good: &[&T], bad: &[&T]) -> TPEWeights {
        let n_good = good.len() as f64;
        let n_bad = bad.len() as f64;

        let normalize_cst_good = n_good + self.0;
        let normalize_cst_bad = n_bad + self.0;

        let good_weight = 1.0 / normalize_cst_good;
        let bad_weight = 1.0 / normalize_cst_bad;

        let good_prior_weight = self.0 / normalize_cst_good;
        let bad_prior_weight = self.0 / normalize_cst_bad;

        TPEWeights {
            good: PointWeights {
                weights: vec![good_weight; good.len()],
                prior_weight: good_prior_weight,
            },
            bad: PointWeights {
                weights: vec![bad_weight; bad.len()],
                prior_weight: bad_prior_weight,
            },
        }
    }
}

// /// An Expected Improvement (EI) [`Weighter`] that assigns weights based on the difference between the quantile of the good set and the values in both sets.
// ///
// /// For the $\\texttt{good} \subset \\mathcal{D}$ set, the weight for each point $s$ is computed as:
// /// $$
// /// w_s = \\frac{y^\\gamma - y_s}{\\sum_{s' \\in \\texttt{good}} \\left( 1 + \\frac{1}{|\\texttt{good}|} \\right)\\left(y^\\gamma - y(s')\\right)}
// /// $$
// /// where $\\gamma$ is the quantile parameter and $y^\\gamma$ is the value at that quantile within $\\mathcal{D}$, and $y(s)$ is the value of point $s$.
// ///
// /// For the bad set, the weight for each point is uniform and inversely proportional to the size of the bad set, with an additional prior weight:
// /// $$
// /// w_s = \\frac{1}{|\\texttt{bad}| + 1}
// /// $$
// #[derive(Serialize, Deserialize, Debug)]
// pub struct EIWeighter(f64);

// impl EIWeighter {
//     /// Creates a new [`EIWeighter`] with a default prior weight of 1.0.
//     pub fn new() -> Self {
//         EIWeighter(1.0)
//     }
// }

// impl Default for EIWeighter {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl<Cod,Out> Weighter<Cod,Out> for EIWeighter
// where
//     Cod: Single<Out, TypeCodom = f64>,
//     Out: Outcome,
// {
//     fn weight<T: HasY<Cod,Out>>(&self, good: &[T], bad: &[T]) -> TPEWeights
//     {
//         let quantile = good.last().unwrap().y();
//         let mut good_sum = 0.0;
//         let good_diff: Vec<f64> = good.iter().map(|s|
//         {
//             let diff = *quantile - *s.y();
//             good_sum += diff;
//             diff
//         }).collect();

//         let good_mean = good_sum / (good.len() as f64);
//         let good_weights: Vec<f64> = good_diff.iter().map(|d| d / (good_sum + good_mean)).collect();

//         let bad_weight = 1.0 / (bad.len() + 1) as f64;
//         let bad_weights = vec![bad_weight ; bad.len()];
//         TPEWeights {
//             good: PointWeights {
//                 weights: good_weights,
//                 prior_weight: good_mean / (good_sum + good_mean),
//             },
//             bad: PointWeights {
//                 weights: bad_weights,
//                 prior_weight: bad_weight,
//             },
//         }
//     }

//     fn with_prior(mut self, prior: f64) -> Self {
//         self.0 = prior;
//         self
//     }
// }

// /// An old decay [`Weighter`] described in [Watanabe 2025](https://arxiv.org/pdf/2304.11127).
// ///
// /// $$
// ///  w_s = \\begin{cases}
// /// \\frac{1}{|\\texttt{good}| + 1} & \\text{if } s \\in \\texttt{good} \\\\
// /// \\tau(i) + \\frac{1 - \\tau(i)}{|\\texttt{bad}| + 1} & \\text{if } s \\in \\texttt{bad}
// /// \\end{cases}
// /// $$
// ///
// /// where $i$ is the index of the point in the sorted bad set, and $\tau(i)$ is a decay function defined as:
// ///
// /// $$
// ///  \tau(i) := \\texttt{i} / (\\texttt{length} - \\texttt{decay\_length})
// /// $$
// #[derive(Serialize, Deserialize, Debug)]
// pub struct OldDecayWeighter(f64, usize);

// impl OldDecayWeighter{
//     /// Creates a new [`OldDecayWeighter`] with a default decay length of 25.
//     pub fn new() -> Self {
//         OldDecayWeighter(1.0, 25)
//     }
//     /// Creates a new [`OldDecayWeighter`] with a specified decay length.
//     pub fn with_decay_length(mut self, length: usize) -> Self {
//         self.1 = length;
//         self
//     }
//     /// Computes the decay rate for a given index `i` and total length of the set.
//     pub fn decay_rate(&self, i:usize, length:usize) -> f64 {
//         i  as f64 / (length - self.1) as f64
//     }
// }

// impl Default for OldDecayWeighter {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl<Cod,Out> Weighter<Cod,Out> for OldDecayWeighter
// where
//     Cod: Codomain<Out>,
//     Out: Outcome,
// {
//     fn weight<T: HasY<Cod,Out>>(&self, good: &[T], bad: &[T]) -> TPEWeights
//     {
//         let good_weight = 1.0 / (good.len() + 1) as f64;
//         let good_weights = vec![good_weight ; good.len()];
//         if bad.len() < self.1 {
//             let bad_weight = 1.0 / (bad.len() + 1) as f64;
//             return TPEWeights {
//                 good: PointWeights {
//                     weights: good_weights,
//                     prior_weight: good_weight,
//                 },
//                 bad: PointWeights {
//                     weights: vec![bad_weight ; bad.len()],
//                     prior_weight: bad_weight,
//                 },
//             }
//         }
//         let mut sum :f64 = 0.0;
//         let bad_weight_prime: Vec<f64> = (0..bad.len()).map(
//             |i|{
//                 if i > bad.len() - self.1{
//                     sum += 1.0;
//                     1.0
//                 } else {
//                     let decay = self.decay_rate(i, bad.len());
//                     let w = decay + (1.0-decay)/(bad.len() + 1) as f64;
//                     sum += w;
//                     w
//                 }
//             }
//         ).collect();
//         let bad_weights: Vec<f64> = bad_weight_prime.iter().map(|w| w/sum).collect();
//         TPEWeights {
//             good_weights,
//             bad_weights,
//             good_prior_weight: good_weight,
//             bad_prior_weight: self.0 / sum }
//     }

//     fn with_prior(mut self, prior: f64) -> Self {
//         self.0 = prior;
//         self
//     }
// }
