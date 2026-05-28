use serde::{Deserialize, Serialize};

use crate::{bayesian::error::SplitError, utils::OrdArchive};

/// The [`Splitter`] trait defines a method for splitting a vector into two parts within the TPE algorithm.
pub trait Splitter
where
    Self: Sized + Serialize + for<'a> Deserialize<'a>,
{
    /// Splits the given archive of points into two parts: the "good" set and the "bad" set.
    /// The "good" set contains the best points, while the "bad" set contains the worst points.
    /// The split is typically based on a quantile of the archive, which determines the fraction of points that belong to the "good" set.
    ///
    /// # Parameters
    /// - `archive`: A reference to a [`OrdArchive`] containing the observed points, sorted in ascending order by their objective values.
    ///
    /// # Returns
    /// A tuple containing two slices: the first slice corresponds to the "good" set (the best points), and the second slice corresponds to the "bad" set (the worst points).
    fn split<'a, T: Ord + Serialize + for<'de> Deserialize<'de>>(
        &self,
        archive: &'a OrdArchive<T>,
    ) -> (&'a [T], &'a [T]);
}

/// A simple linear [`Splitter`].
/// The quantile is determined by a fixed parameter $\beta$, where the best set contains the top $\beta$ fraction of the values.
///
/// # Condition
/// $\beta \in (0,1)$.
///
/// # Behavior
///
/// A small $\beta$ will result in a smaller best set, resulting in a more explorative optimization.
/// See [Watanabe](https://arxiv.org/pdf/2304.11127) for more details.
#[derive(Serialize, Deserialize, Debug)]
pub struct LinearSplit(pub f64);
impl LinearSplit {
    pub fn new(beta: f64) -> Result<Self, SplitError> {
        if beta <= 0.0 || beta >= 1.0 {
            return Err(SplitError("Beta must be between 0 and 1".into()));
        }
        Ok(LinearSplit(beta))
    }
}
impl Splitter for LinearSplit {
    fn split<'a, T: Ord + Serialize + for<'de> Deserialize<'de>>(
        &self,
        archive: &'a OrdArchive<T>,
    ) -> (&'a [T], &'a [T]) {
        let quantile = (archive.size() as f64 * (1.0 - self.0)).ceil() as usize;
        (&archive.points[quantile..], &archive.points[..quantile])
    }
}

/// A square root [`Splitter`].
///
/// The quantile is determined by a parameter $\beta$,
/// where the best set contains the top $\beta / \sqrt{n}$ fraction of the values, with $n$ being the number of observations.
///
/// # Condition
/// $\beta \in (0,\sqrt{N})$.
///
/// # Behavior
///
/// Compared to the [`LinearSplit`] function, the [`SqrtSplit`] is more exploratibe by decreasing the quantile as the number of observations increases.
///
/// A small $\beta$ will result in a smaller best set, resulting in a more explorative optimization.
/// See [Watanabe](https://arxiv.org/pdf/2304.11127) for more details.
#[derive(Serialize, Deserialize, Debug)]
pub struct SqrtSplit(pub f64);

impl SqrtSplit {
    pub fn new(beta: f64) -> Result<Self, SplitError> {
        if beta <= 0.0 {
            return Err(SplitError("Beta must be positive".into()));
        }
        Ok(SqrtSplit(beta))
    }
}
impl Splitter for SqrtSplit {
    fn split<'a, T: Ord + Serialize + for<'de> Deserialize<'de>>(
        &self,
        archive: &'a OrdArchive<T>,
    ) -> (&'a [T], &'a [T]) {
        let size = archive.size() as f64;
        let quantile = (size - (self.0 / size.sqrt())).ceil();
        if quantile < 0.0 {
            return (&[], &archive.points[..]);
        }
        let quantile = quantile as usize;
        (&archive.points[quantile..], &archive.points[..quantile])
    }
}
