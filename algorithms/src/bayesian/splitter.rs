use std::ops::Neg;

use crate::bayesian::error::SplitError;
use tantale_core::{Dominate, NdArrayDominate, NonDominatedSorting, Orderable, OrderedArchive};

use ndarray::{Array2, Axis, Zip, concatenate, s};
use serde::{Deserialize, Serialize};
use wfg_rs::wfg::{inclusive_wfg, reference_point, wfg};

/// The [`Splitter`] trait defines a method for splitting a vector into two parts within the TPE algorithm.
/// The element to be split must implement the [`Orderable`] trait, which allows for comparison and ordering of elements.
pub trait Splitter<T>: Serialize + for<'a> Deserialize<'a>
where
    T: Orderable + Serialize + for<'a> Deserialize<'a>,
{
    /// Splits the given archive of points into two parts: the "good" set and the "bad" set.
    /// The "good" set contains the best points, while the "bad" set contains the worst points.
    /// The split is typically based on a quantile of the archive, which determines the fraction of points that belong to the "good" set.
    ///
    /// # Parameters
    /// - `archive`: A reference to a [`OrderedArchive`] containing the observed points, sorted in ascending order by their objective values.
    ///
    /// # Returns
    /// A tuple containing two slices: the first slice corresponds to the "good" set (the best points), and the second slice corresponds to the "bad" set (the worst points).
    fn split<'a>(&self, archive: &'a OrderedArchive<T>) -> (Vec<&'a T>, Vec<&'a T>);
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
impl<T> Splitter<T> for LinearSplit
where
    T: PartialOrd + Orderable + Serialize + for<'a> Deserialize<'a>,
{
    fn split<'a>(&self, archive: &'a OrderedArchive<T>) -> (Vec<&'a T>, Vec<&'a T>) {
        let quantile = (archive.size() as f64 * (1.0 - self.0)).ceil() as usize;
        let (bad, good) = archive.points.split_at(quantile);
        (good.iter().collect(), bad.iter().collect())
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
impl<T> Splitter<T> for SqrtSplit
where
    T: PartialOrd + Orderable + Serialize + for<'a> Deserialize<'a>,
{
    fn split<'a>(&self, archive: &'a OrderedArchive<T>) -> (Vec<&'a T>, Vec<&'a T>) {
        let size = archive.size() as f64;
        let quantile = (size - (self.0 / size.sqrt())).ceil();
        if quantile < 0.0 {
            return (Vec::new(), archive.points.iter().collect());
        }
        let quantile = quantile as usize;
        let (bad, good) = archive.points.split_at(quantile);
        (good.iter().collect(), bad.iter().collect())
    }
}

pub fn greedy_hss<'a, T>(front: &mut Vec<&'a T>, n: usize) -> Vec<&'a T>
where
    T: Dominate,
{
    if front.len() == n {
        return std::mem::take(front);
    }
    if front.len() < n {
        panic!("Not enough points in the front to extract {} points", n);
    }

    // Create a 2D array from the front points / negate because WFG consider a minimization problem
    let mut nd_front = front.as_slice().dom_array().neg();
    // Get the reference point for hypervolume calculation
    let ref_point = reference_point(nd_front.view());
    // Initialize an empty array to store the extracted points
    let mut extracted = Array2::<f64>::zeros((n, nd_front.ncols()));
    // Initialize a vector to store the selected points
    let mut selected = Vec::new();

    // Calculate the inclusive hypervolume contributions of the points in the front
    let inclusive = inclusive_wfg(nd_front.view(), ref_point.view());

    let mut best_contrib = inclusive
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|m| (m.0, *m.1));

    let mut i = 0;
    while let Some((idx, _)) = best_contrib
        && i < n
    {
        extracted.row_mut(i).assign(&nd_front.row(idx));
        let last = nd_front.nrows() - 1;
        if idx != last {
            let (mut a, b) =
                nd_front.multi_slice_mut((s![idx..idx + 1, ..], s![last..last + 1, ..]));
            a.assign(&b);
        }
        nd_front.slice_axis_inplace(Axis(0), ndarray::Slice::from(..last));

        let hv_indic_extracted = wfg(extracted.slice(s![..i, ..]), ref_point.view(), true, false);

        let nd_worses = Zip::from(&nd_front)
            .and_broadcast(extracted.row(i))
            .map_collect(|&x, &y| x.max(y));

        best_contrib = nd_worses
            .rows()
            .into_iter()
            .enumerate()
            .map(|(idx, row)| {
                let union = concatenate(
                    Axis(0),
                    &[extracted.slice(s![..i, ..]), row.insert_axis(Axis(0))],
                )
                .unwrap();
                let hv = wfg(union.view(), ref_point.view(), false, false) - hv_indic_extracted;
                (idx, hv)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));

        selected.push(front.remove(idx));
        i += 1;
    }
    selected
}

/// A hypervolume-based [`Splitter`].
/// The quantile is determined by a parameter $\beta$,
/// where the best set contains the top $\beta$ fraction of the values.
///
/// The algorithm fills selected points using points of non-domination rank ($\texttt{rank}(1), \dots ,\texttt{rank}(J)$) until the
/// $\lvert D_\text{good} \rvert + \lvert D_{\texttt{rank}(j)} \rvert \leq \lceil \beta \lvert D \rvert \rceil$.
/// The sets of $\texttt{rank}(j)$ are obtained via [`NonDominatedSorting`].
/// Then if the set of $\texttt{rank}(j)$ is not sufficient to fill the remaining solutions, a greedy approach on set of rank $\texttt{rank}(j+1)$
/// is used to select the points that contribute the most to the hypervolume of the front, ensuring that the best set is diverse and representative of the Pareto front.
///
/// # Note
///
/// See [Ozaki et al.](https://www.jair.org/index.php/jair/article/view/13188/26784) for more details.
#[derive(Serialize, Deserialize, Debug)]
pub struct MOSplit(pub f64);

impl MOSplit {
    pub fn new(beta: f64) -> Result<Self, SplitError> {
        if beta <= 0.0 || beta >= 1.0 {
            return Err(SplitError("Beta must be between 0 and 1".into()));
        }
        Ok(MOSplit(beta))
    }
}

impl<T> Splitter<T> for MOSplit
where
    T: Dominate + Serialize + for<'a> Deserialize<'a>,
{
    fn split<'a>(&self, archive: &'a OrderedArchive<T>) -> (Vec<&'a T>, Vec<&'a T>) {
        let quantile = (archive.size() as f64 * self.0).ceil() as usize;

        let mut fronts = archive.lex_non_dominated_sort();

        let mut good = Vec::with_capacity(quantile);
        let mut bad = Vec::with_capacity(archive.size() - quantile);

        let mut i = 0;
        while i < fronts.len() && good.len() + fronts[i].len() < quantile {
            good.extend_from_slice(&fronts[i]);
            i += 1;
        }

        if i < fronts.len() && good.len() < quantile {
            let remaining = quantile - good.len();

            let extracted = greedy_hss(&mut fronts[i], remaining);

            good.extend(extracted);
            bad.extend_from_slice(&fronts[i]);
            i += 1;
        }

        // Add the remaining fronts to the bad set
        for f in &fronts[i..] {
            bad.extend_from_slice(f);
        }

        (good, bad)
    }
}
