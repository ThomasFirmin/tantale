//! This module defines the Pareto domination relationship between two [`Dominate`] elements.

use crate::{OrderedArchive, utils::orderable::{Orderable, arg_lexsort, lexsort}};

use itertools::Itertools;
use ndarray::{Array2, Zip};
use serde::{Deserialize, Serialize};

/// The [`pareto_mask`] function computes a boolean mask indicating which elements in the input slice are non-dominated.
pub fn pareto_mask<T: Dominate>(values: &[T]) -> Vec<bool> {
    let n = values.len();
    let mut mask = vec![true; n];
    for i in 0..n {
        if !mask[i] {
            continue;  // skip already-dominated points
        }
        for j in (i + 1)..n {
            if !mask[j] {
                continue;  // skip already-dominated points
            }
            if values[i].dominates(&values[j]) {
                mask[j] = false;
            } else if values[j].dominates(&values[i]) {
                mask[i] = false;
                break;
            }
        }
    }
    mask
}

/// The [`pareto_mask_lexsorted`] function computes a boolean mask indicating which elements in the input slice 
/// are non-dominated, assuming that the input is sorted in lexicographical order of objectives.
pub fn pareto_mask_lexsorted<T: Dominate>(values: &[T]) -> Vec<bool> {
    let n = values.len();
    let d = values[0].len_objectives();
    let mut mask = vec![true; n];
    // Track best seen so far per objective (we're maximizing)
    let mut best = vec![f64::NEG_INFINITY; d];
    
    // Scan in reverse: last point (best on obj[0]) sets the bar
    for i in (0..n).rev() {
        // A point is dominated if any previously seen point beats it on all objectives
        // With lexsort, we only need to track the running max of other objectives
        if values[i].iter_obj().zip(best.iter()).any(|(v, &b)| v < b) {
            mask[i] = false;
        } else {
            // Update best
            for (k, b) in best.iter_mut().enumerate().take(d) {
                *b = (*b).max(values[i].get_objective_by_index(k));
            }
        }
    }
    mask
}

/// Implements the binary search used for non-dominated sorting, described in [Zhang et al. (2014)](http://www.soft-computing.de/ENS.pdf).
///
/// # Parameters
/// - `target`: The [`Dominate`] object to be sorted.
/// - `fronts`: The set of non-dominated fronts
///
/// # Returns
/// The index of the front where the target should be sorted. If the target is dominated by all fronts, returns `fronts.len() + 1`.
///
/// # Pseudo-code
///
/// **Front Binary Search**
/// ---
/// **Inputs**
/// 1. &emsp; $F$ &emsp;&emsp; *The set of non-dominated fronts*
/// 2. &emsp; $p$ &emsp;&emsp; *The point to be sorted*
/// 3. &emsp;
/// 4. &emsp; $s \gets \lvert F \rvert$ &emsp;&emsp; *The number of fronts*
/// 5. &emsp; $k_{min} \gets 0$ &emsp;&emsp; *The minimum index of the front*
/// 6. &emsp;&emsp; $k_{max} \gets s$ &emsp;&emsp; *The maximum index of the front*
/// 7. &emsp; $k \gets \lceil k_{max}/2 \rceil$ &emsp;&emsp; *The index of the front to be compared*
/// 8. &emsp;
/// 9. &emsp; **loop**
/// 10. &emsp;&emsp; **if** $\exists q \in F_k\,;\, q \succ p$ **then**
/// 11. &emsp;&emsp;&emsp; $k_{min} \gets k$
/// 12. &emsp;&emsp;&emsp; **if** $k_{max} = k_{min} + 1$ **then**
/// 13. &emsp;&emsp;&emsp;&emsp; **return** $k_{max}$
/// 14. &emsp;&emsp;&emsp; **else if** $k_{min} = s$ **then**
/// 15. &emsp;&emsp;&emsp;&emsp; **return** $s + 1$
/// 16. &emsp;&emsp;&emsp; **else**
/// 17. &emsp;&emsp;&emsp;&emsp; $k \gets \lceil (k_{min} + k_{max})/2 \rceil$
/// 18. &emsp;&emsp; **else if** $k = k_{min} + 1$ **then**
/// 19. &emsp;&emsp;&emsp; **return** $k$
/// 20. &emsp;&emsp; **else**
/// 21. &emsp;&emsp;&emsp; $k_{max} \gets k$
/// 22. &emsp;&emsp;&emsp; $k \gets \lceil (k_{min} + k_{max})/2 \rceil$
/// ---
pub fn front_binary_search<T: Dominate, F: DominateView<T>>(target: &T, fronts: &[Vec<F>]) -> usize {
    let n_fronts = fronts.len();
    let mut k_min = 0;
    let mut k_max = n_fronts;
    let mut k = k_max.div_ceil(2);

    loop {
        let current_front = &fronts[k - 1];
        let one_dominates = current_front.iter().any(|x| x.view().dominates(target));

        if one_dominates {
            k_min = k;
            if k_max == k_min + 1 && k_max < n_fronts {
                return k_max;
            } else if k_min == n_fronts {
                return n_fronts + 1;
            } else {
                k = (k_min + k_max).div_ceil(2);
            }
        } else if k == k_min + 1 {
            return k;
        } else {
            k_max = k;
            k = (k_min + k_max).div_ceil(2);
        }
    }
}

/// Similar to [`front_binary_search`] but works on the indices of the input objects instead of the objects themselves.
pub fn arg_front_binary_search<T: Dominate>(
    target: &T,
    values: &[T],
    fronts: &[Vec<usize>],
) -> usize {
    let n_fronts = fronts.len();
    let mut k_min = 0;
    let mut k_max = n_fronts;
    let mut k = k_max.div_ceil(2);

    loop {
        let current_front = &fronts[k - 1];
        let one_dominates = current_front.iter().any(|x| values[*x].dominates(target));

        if one_dominates {
            k_min = k;
            if k_max == k_min + 1 && k_max < n_fronts {
                return k_max;
            } else if k_min == n_fronts {
                return n_fronts + 1;
            } else {
                k = (k_min + k_max).div_ceil(2);
            }
        } else if k == k_min + 1 {
            return k;
        } else {
            k_max = k;
            k = (k_min + k_max).div_ceil(2);
        }
    }
}

/// Defines the Pareto domination relationship between two elements of a [`Multi`](crate::Multi)-objective [`Codomain`](crate::Codomain).
///
/// Since Tantale maximizes, `self` dominates `other` if:
/// - `self` is at least as good as `other` on every objective ($\forall i, y_i(self) \geq y_i(other)$), and
/// - `self` is strictly better than `other` on at least one objective ($\exists i, y_i(self) > y_i(other)$).
///
/// For [`Constrained`](crate::Constrained) variants, constraint feasibility takes priority:
/// - A solution with strictly lower total constraint violation always dominates the other,
///   regardless of objective values.
/// - When both solutions share the same total violation (including the fully feasible case),
///   standard Pareto dominance on objectives applies.
///
/// # Note
/// The [`Cost`](crate::Cost) dimension is intentionally excluded from dominance: it is used by
/// cost-aware algorithms (e.g. for budget allocation), not as a Pareto objective.
pub trait Dominate: Orderable {
    /// Returns `true` if `self` Pareto-dominates `other`.
    fn dominates(&self, other: &Self) -> bool;

    /// Returns a slice of the objective values of this [`Codomain`](crate::Codomain).
    fn get_objectives(&self) -> &[f64];

    /// Returns a boxed slice of the objective values of this [`Codomain`](crate::Codomain).
    fn clone_objective(&self) -> Vec<f64> {
        self.get_objectives().to_vec()
    }

    /// Returns the value of the objective at the specified index.
    fn get_objective_by_index(&self, idx: usize) -> f64;

    /// Returns the number of objectives in this [`Codomain`](crate::Codomain).
    fn len_objectives(&self) -> usize;

    /// Returns an iterator over the objective values of this [`Codomain`](crate::Codomain).
    fn iter_obj(&self) -> impl Iterator<Item = f64> {
        (0..self.len_objectives()).map(|idx| self.get_objective_by_index(idx))
    }
}

impl Dominate for [f64]
{
    fn dominates(&self, other: &Self) -> bool {
        let mut strictly_better = false;
        for (a, b) in self.iter().zip(other.iter()) {
            if a < b {
                return false; // Not at least as good
            } else if a > b {
                strictly_better = true; // Found a strictly better objective
            }
        }
        strictly_better
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        self[idx]
    }

    fn len_objectives(&self) -> usize {
        self.len()
    }
    
    fn get_objectives(&self) -> &[f64] {
        self
    }
}


impl<T: Dominate> Dominate for &T {
    fn dominates(&self, other: &Self) -> bool {
        (**self).dominates(*other)
    }

    fn get_objectives(&self) -> &[f64] {
        (**self).get_objectives()
    }

    fn get_objective_by_index(&self, idx: usize) -> f64 {
        (**self).get_objective_by_index(idx)
    }

    fn len_objectives(&self) -> usize {
        (**self).len_objectives()
    }
}

/// A helper trait for types that can be viewed as a [`Dominate`] object, 
/// allowing for flexible handling of references and owned values.
pub trait DominateView<T:Dominate> {
    fn view(&self) -> &T;
}

impl<T: Dominate> DominateView<T> for T {
    fn view(&self) -> &T {
        self
    }
}

impl<T: Dominate> DominateView<T> for &T {
    fn view(&self) -> &T {
        self
    }
}

/// A helper trait for converting  objects made of [`Dominate`] into an [`ndarray::Array2<f64>`] representation.
pub trait NdArrayDominate {
    /// Returns a 2D array representation of the object, 
    /// where each row corresponds to an individual and each column corresponds to an objective.
    fn dom_array(&self) -> Array2<f64>;
}

impl<T:Dominate> NdArrayDominate for [T] {
    fn dom_array(&self) -> Array2<f64> {
        let n_rows = self.len();
        let n_cols = self[0].len_objectives();
        let mut array = Array2::<f64>::uninit((n_rows, n_cols));
        Zip::from(array.rows_mut())
        .and(self)
        .for_each(
            |mut row, d|
            {
                Zip::from(&mut row)
                .and(d.get_objectives())
                .for_each(
                    |r, o|
                    {
                        r.write(*o);
                    }
                );
            }
        );
        unsafe {array.assume_init()}
    }
}

/// Implements the Pareto front extraction algorithm, 
/// which identifies the non-dominated solutions from a given set of solutions.
/// 
/// # Example
///
/// Consider the following set of solutions, where each solution is represented as a point in a 2D objective space:
///
/// ```text
///  5.0 |  1  .  .  .  .  .  .  .  .  .  .
///  4.5 |  .  .  .  .  2  .  .  .  .  .  .
///  4.0 |  .  6  .  .  .  .  3  .  .  .  .
///  3.5 | 11  .  .  .  7  .  .  .  .  .  .
///  3.0 |  .  . 12  .  .  .  8  .  4  .  .
///  2.5 |  .  .  .  .  .  .  .  .  .  .  .
///  2.0 |  .  .  .  . 13  .  .  .  9  .  .
///  1.5 |  .  .  .  .  .  .  .  .  .  .  .
///  1.0 |  .  .  .  .  . 14  .  .  .  .  5
///  0.5 |  .  .  .  .  .  .  .  .  .  .  .
///  0.0 |  .  .  .  .  .  . 15  .  .  . 10
///      +---------------------------------
///       0.0   1.0   2.0   3.0   4.0   5.0
/// ```
///
/// The fronts are as follows:
///
/// ```text
/// Pareto front:
///     [5, 4, 3, 2, 1]
/// ```
pub trait ParetoFront<T:Dominate> {
        // Returns the non-dominated solutions from the input set of solutions.
        fn pareto(&self, lexsorted: bool) -> Vec<&T>;
        // Returns a boolean mask indicating which elements in the input set of solutions are non-dominated.
        fn is_pareto(&self, lexsorted: bool) -> Vec<bool>;
        /// Returns a clone of the non-dominated solutions from the input set of solutions.
        fn pareto_clone(&self, lexsorted: bool) -> Vec<T> where T: Clone;
        // Return the non-dominated indexes of solutions from the input set of solutions.
        fn pareto_arg(&self, lexsorted: bool) -> Vec<usize>;
        /// Returns a tuple of the dominated solutions and the non-dominated solutions from the input set of solutions.
        fn pareto_split(&self, lexsorted: bool) -> (Vec<&T>, Vec<&T>);
        /// Return a tuple of the dominated indexes and the non-dominated indexes of solutions from the input set of solutions.
        fn pareto_argsplit(&self, lexsorted: bool) -> (Vec<usize>, Vec<usize>);
}

impl <T, U> ParetoFront<T> for U
where
    T: Dominate,
    U: ?Sized + AsRef<[T]>,
{
    fn pareto(&self, lexsorted: bool) -> Vec<&T> {
        let mask = if lexsorted {pareto_mask_lexsorted(self.as_ref())} else {pareto_mask(self.as_ref())};
        self.as_ref().iter().zip(mask).filter_map(|(x, m)| if m { Some(x) } else { None }).collect()
    }

    fn is_pareto(&self, lexsorted: bool) -> Vec<bool> {
        if lexsorted {pareto_mask_lexsorted(self.as_ref())} else {pareto_mask(self.as_ref())}
    }

    fn pareto_arg(&self, lexsorted: bool) -> Vec<usize> {
        let mask = if lexsorted {pareto_mask_lexsorted(self.as_ref())} else {pareto_mask(self.as_ref())};
        self.as_ref().iter().zip(mask).enumerate().filter_map(|(i, (_, m))| if m { Some(i) } else { None }).collect()
    }
    
    fn pareto_split(&self, lexsorted: bool) -> (Vec<&T>, Vec<&T>) {
        let mask = if lexsorted {pareto_mask_lexsorted(self.as_ref())} else {pareto_mask(self.as_ref())};
        self.as_ref().iter().zip(mask).partition_map(|(x, m)| if m { itertools::Either::Right(x) } else { itertools::Either::Left(x) })
    }
    
    fn pareto_argsplit(&self, lexsorted: bool) -> (Vec<usize>, Vec<usize>) {
        let mask = if lexsorted {pareto_mask_lexsorted(self.as_ref())} else {pareto_mask(self.as_ref())};
        (0..self.as_ref().len()).zip(mask).partition_map(|(x, m)| if m { itertools::Either::Right(x) } else { itertools::Either::Left(x) })
    }

    fn pareto_clone(&self, lexsorted: bool) -> Vec<T> 
    where T: Clone 
    {
        let mask = if lexsorted {pareto_mask_lexsorted(self.as_ref())} else {pareto_mask(self.as_ref())};
        self.as_ref()
            .iter()
            .zip(mask.iter())
            .filter(|&(_v, &keep)| keep)
            .map(|(v, &_keep)| v.clone())
            .collect()
    }
}

/// Similar to [`ParetoFront`] but consumes the input set of solutions instead of borrowing it.
pub trait IntoParetoFront<T: Dominate> {
        /// Returns the non-dominated solutions from the input set of solutions.
        fn into_pareto_front(self) -> Vec<T>;
        /// Returns the dominated and non-dominated solutions from the input set of solutions.
        fn into_pareto_split(self) -> (Vec<T>, Vec<T>);
}

impl <T:Dominate> IntoParetoFront<T> for Vec<T> {
    fn into_pareto_front(self) -> Vec<T> {
        let mask = pareto_mask(&self);
        self.into_iter().zip(mask).filter_map(|(x, m)| if m { Some(x) } else { None }).collect()
    }

    fn into_pareto_split(self) -> (Vec<T>, Vec<T>) {    
        let mask = pareto_mask(&self);
        self.into_iter().zip(mask).partition_map(|(x, m)| if m { itertools::Either::Right(x) } else { itertools::Either::Left(x) })
    }
}

/// Implements the binary search non-dominated sorting algorithm, described in [Zhang et al. (2012)](http://www.soft-computing.de/ENS.pdf).
///
/// # Parameters
/// - `self`: An iterable of [`Dominate`] objects to be sorted.
///
/// # Returns
///
/// Returns a vector of non-dominated fronts, where each front is a vector of references to the original objects.
/// The first front contains the non-dominated objects, the second front contains the objects dominated only by those in the first front, and so on.
///
/// # Pseudo-code
///
/// **Efficient Non-Dominated Sorting**
/// ---
/// **Inputs**
/// 1. &emsp; $\mathcal{B}$ &emsp;&emsp; *An iterable of [`Dominate`] object.*
/// 2. &emsp; $F \gets \emptyset$ &emsp;&emsp; *The set of non-dominated fronts, initially empty. $F_1,F_2,\dots$*
/// 3. &emsp;
/// 4. &emsp; **Sort** $\mathcal{B}$ according to first objectives, break ties with seconds, break ties with thirds, etc.
/// 5. &emsp;
/// 6. &emsp; **for all** $p \in \mathcal{B}$ **do**
/// 7. &emsp;&emsp; $i \gets $[FrontBinarySearch](front_binary_search)$(p,F)$
/// 8. &emsp;&emsp; **if** $i > |F|$ **then**
/// 9. &emsp;&emsp;&emsp; $F \gets F \cup \\{p\\}$
/// 10. &emsp;&emsp; **else**
/// 11. &emsp;&emsp;&emsp; $F_i \gets F_i \cup \\{p\\}$
/// 12. &emsp; **return** $F$
/// ---
///
/// # Example
///
/// Consider the following set of solutions, where each solution is represented as a point in a 2D objective space:
///
/// ```text
///  5.0 |  1  .  .  .  .  .  .  .  .  .  .
///  4.5 |  .  .  .  .  2  .  .  .  .  .  .
///  4.0 |  .  6  .  .  .  .  3  .  .  .  .
///  3.5 | 11  .  .  .  7  .  .  .  .  .  .
///  3.0 |  .  . 12  .  .  .  8  .  4  .  .
///  2.5 |  .  .  .  .  .  .  .  .  .  .  .
///  2.0 |  .  .  .  . 13  .  .  .  9  .  .
///  1.5 |  .  .  .  .  .  .  .  .  .  .  .
///  1.0 |  .  .  .  .  . 14  .  .  .  .  5
///  0.5 |  .  .  .  .  .  .  .  .  .  .  .
///  0.0 |  .  .  .  .  .  . 15  .  .  . 10
///      +---------------------------------
///       0.0   1.0   2.0   3.0   4.0   5.0
/// ```
///
/// The fronts are as follows:
///
/// ```text
/// Front 1:
///     [5, 4, 3, 2, 1]
/// Front 2:
///     [10, 9, 8, 7, 6]
/// Front 3:
///     [15, 14, 13, 12, 11]
/// ```
pub trait NonDominatedSorting<T>
where
    T: Dominate,
{   
    /// Sorts the input objects into non-dominated fronts using a binary search approach.
    fn non_dominated_sort(&mut self) -> Vec<Vec<&T>>;
    /// Sorts the input objects into non-dominated fronts using a binary search approach.
    /// Assuming that self is lexicographically sorted.
    fn lex_non_dominated_sort(&self) -> Vec<Vec<&T>>;
    /// Returns the indices of the input objects sorted according to their non-dominated fronts.
    fn non_dominated_argsort(&self) -> Vec<Vec<usize>>;
}

impl<T: Dominate> NonDominatedSorting<T> for [T] {
    fn non_dominated_sort(&mut self) -> Vec<Vec<&T>> {
        lexsort(self);
        self.lex_non_dominated_sort()
    }
    fn lex_non_dominated_sort(&self) -> Vec<Vec<&T>> {
        let last_idx = self.len() - 1;
        let mut fronts = vec![vec![&self[last_idx]]];
        for item in self[..last_idx].iter().rev() {
            let idx_front = front_binary_search(item, &fronts);
            if idx_front > fronts.len() {
                fronts.push(vec![item]);
            } else {
                fronts[idx_front - 1].push(item);
            }
        }
        fronts
    }
    fn non_dominated_argsort(&self) -> Vec<Vec<usize>> {
        let last_idx = self.len() - 1;
        let indices = arg_lexsort(self);
        let mut fronts = vec![vec![indices[last_idx]]];
        for idx in indices[..last_idx].iter().rev() {
            let idx_front = arg_front_binary_search(&self[*idx], self, &fronts);
            if idx_front > fronts.len() {
                fronts.push(vec![*idx]);
            } else {
                fronts[idx_front - 1].push(*idx);
            }
        }
        fronts
    }
}

pub trait IntoNonDominatedSorting<T>
where
    Self: Sized,
    T: Dominate,
{
    fn into_non_dominated_sort(self) -> Vec<Vec<T>>;
}

impl<T: Dominate> IntoNonDominatedSorting<T> for Vec<T> {
    fn into_non_dominated_sort(mut self) -> Vec<Vec<T>> {
        lexsort(&mut self);
        let mut fronts = vec![vec![self.pop().unwrap()]];
        while let Some(item) = self.pop() {
            let idx_front = front_binary_search(&item, &fronts);
            if idx_front > fronts.len() {
                fronts.push(vec![item]);
            } else {
                fronts[idx_front - 1].push(item);
            }
        }
        fronts
    }
}

impl<T> NonDominatedSorting<T> for OrderedArchive<T> 
where
    T: Dominate + Serialize + for<'de> Deserialize<'de>,
{
    fn non_dominated_sort(&mut self) -> Vec<Vec<&T>> {
        self.points.non_dominated_sort()
    }

    fn lex_non_dominated_sort(&self) -> Vec<Vec<&T>> {
        self.points.lex_non_dominated_sort()
    }

    fn non_dominated_argsort(&self) -> Vec<Vec<usize>> {
        self.points.non_dominated_argsort()
    }
}