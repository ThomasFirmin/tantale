use tantale_core::Dominate;

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// [`FrontItem`] is a trait that abstracts over the items contained in the fronts of the non-dominated sorting algorithm.
pub trait FrontItem<T:Dominate> {
    fn as_dominate(&self) -> &T;
}

impl<T: Dominate> FrontItem<T> for T {
    fn as_dominate(&self) -> &T {
        self
    }
}

impl<T: Dominate> FrontItem<T> for &T {
    fn as_dominate(&self) -> &T {
        self
    }
}

/// The [`pareto_mask`] function computes a boolean mask indicating which elements in the input slice are dominated by at least one other element.
fn pareto_mask<T: Dominate>(values: &[T]) -> Vec<bool> {
    let n = values.len();
    let mut mask = vec![true; n];
    for i in 0..n {
        if mask[i] {
            continue;
        }

        for j in (i + 1)..n {
            if mask[j] {
                continue;
            }

            if values[i].dominates(&values[j]) {
                mask[j] = true;
            } else if values[j].dominates(&values[i]) {
                mask[i] = true;
                break;
            }
        }
    }
    mask
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
        fn pareto_front(&self) -> Vec<&T>;
        // Return the non-dominated indexes of solutions from the input set of solutions.
        fn pareto_argfront(&self) -> Vec<usize>;
        /// Returns a tuple of the dominated solutions and the non-dominated solutions from the input set of solutions.
        fn pareto_split(&self) -> (Vec<&T>, Vec<&T>);
        /// Return a tuple of the dominated indexes and the non-dominated indexes of solutions from the input set of solutions.
        fn pareto_argsplit(&self) -> (Vec<usize>, Vec<usize>);
}

impl <T, U> ParetoFront<T> for U
where
    T: Dominate,
    U: AsRef<[T]>,
{
    fn pareto_front(&self) -> Vec<&T> {
        let mask = pareto_mask(self.as_ref());
        self.as_ref().iter().zip(mask).filter_map(|(x, m)| if m { None } else { Some(x) }).collect()
    }

    fn pareto_argfront(&self) -> Vec<usize> {
        let mask = pareto_mask(self.as_ref());
        self.as_ref().iter().zip(mask).enumerate().filter_map(|(i, (_, m))| if m { None } else { Some(i) }).collect()
    }
    
    fn pareto_split(&self) -> (Vec<&T>, Vec<&T>) {
        let mask = pareto_mask(self.as_ref());
        self.as_ref().iter().zip(mask).partition_map(|(x, m)| if m { itertools::Either::Left(x) } else { itertools::Either::Right(x) })
    }
    
    fn pareto_argsplit(&self) -> (Vec<usize>, Vec<usize>) {
        let mask = pareto_mask(self.as_ref());
        (0..self.as_ref().len()).zip(mask).partition_map(|(x, m)| if m { itertools::Either::Left(x) } else { itertools::Either::Right(x) })
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
        self.into_iter().zip(mask).filter_map(|(x, m)| if m { None } else { Some(x) }).collect()
    }

    fn into_pareto_split(self) -> (Vec<T>, Vec<T>) {    
        let mask = pareto_mask(&self);
        self.into_iter().zip(mask).partition_map(|(x, m)| if m { itertools::Either::Left(x) } else { itertools::Either::Right(x) })
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
    /// Returns the indices of the input objects sorted according to their non-dominated fronts.
    fn non_dominated_argsort(&self) -> Vec<Vec<usize>>;
}

pub fn sort_objectives<T: Dominate>(input: &mut [T]) {
    input.sort_by(|a, b| {
        let length: usize = a.get_max_objectives();
        let mut i = 0;
        let mut ord = a
            .get_objective_by_index(0)
            .total_cmp(&b.get_objective_by_index(0))
            .reverse();
        while let Ordering::Equal = ord {
            i += 1;
            if i < length {
                ord = a
                    .get_objective_by_index(i)
                    .total_cmp(&b.get_objective_by_index(i))
                    .reverse();
            } else {
                break;
            }
        }
        ord
    });
}

pub fn argsort_objectives<T: Dominate>(input: &[T]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..input.len()).collect();
        indices.sort_by(|&a, &b| {
            let length: usize = input[a].get_max_objectives();
            let mut i = 0;
            let mut ord = input[a]
                .get_objective_by_index(0)
                .total_cmp(&input[b].get_objective_by_index(0))
                .reverse();
            while let Ordering::Equal = ord {
                i += 1;
                if i < length {
                    ord = input[a]
                        .get_objective_by_index(i)
                        .total_cmp(&input[b].get_objective_by_index(i))
                        .reverse();
                } else {
                    break;
                }
            }
            ord
        });
        indices
    }

impl<T: Dominate> NonDominatedSorting<T> for [T] {
    fn non_dominated_sort(&mut self) -> Vec<Vec<&T>> {
        sort_objectives(self);
        let mut fronts = vec![vec![&self[0]]];
        for item in self[1..].iter() {
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
        let indices = argsort_objectives(self);
        let mut fronts = vec![vec![indices[0]]];
        for idx in indices[1..].iter() {
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
        sort_objectives(&mut self);
        self.reverse();
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
pub fn front_binary_search<T: Dominate, F: FrontItem<T>>(target: &T, fronts: &[Vec<F>]) -> usize {
    let n_fronts = fronts.len();
    let mut k_min = 0;
    let mut k_max = n_fronts;
    let mut k = k_max.div_ceil(2);

    loop {
        let current_front = &fronts[k - 1];
        let one_dominates = current_front.iter().any(|x| x.as_dominate().dominates(target));

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

/// Defines a trait for candidate selection in multi-objective optimization algorithms.
/// This trait can be implemented by any struct that provides a method for selecting candidates implementing [`Dominate`].
///
/// # See also
/// - [`NSGA2Selector`]: An implementation of the NSGA-II crowding distance selection operator.
pub trait CandidateSelector: std::fmt::Debug + Serialize + for<'a> Deserialize<'a> {
    fn select_candidates<'a, T: Dominate>(&self, values: &'a mut [T], size: usize) -> Vec<&'a T>;
    fn arg_select_candidates<T: Dominate>(&self, values: &[T], size: usize) -> Vec<usize>;
}

/// Implements the NSGA-II crowding distance selection operator, described in [Deb et al. (2002)](https://ieeexplore.ieee.org/document/996017).
///
/// # Parameters
/// - `values`: An iterable of [`Dominate`] objects from [`NonDominatedSorting`].
/// - `size`: The number of candidates to select.
///
/// # Returns
/// A vector of distances associated with each candidate, where higher distances indicate more desirable candidates.
/// Candidates with infinite distance are considered the most desirable, as they are non-dominated and on the boundary of the objective space.
///
/// # Panic
/// This function will panic if the number of values is less than the size of candidates to select
///
/// # Pseudo-code
///
/// **Crowding Distance**
/// ---
/// **Inputs**
/// 1. &emsp; $\mathcal{F} = ((f^1_1,\ldots f^1_m), \ldots, (f^n_1,\ldots f^n_m))$ &emsp;&emsp; *A front from [`NonDominatedSorting`]*
/// 2. &emsp; $m$ &emsp;&emsp; *The number of objectives*
/// 3. &emsp; $n$ &emsp;&emsp; *The number of individuals in the front*
/// 4.
/// 5. &emsp; $d \gets 0\cdot\mathbf{1}_n$ &emsp;&emsp; *The crowding distance vector, initialized to zero*
/// 6. &emsp; $i \gets \{1,2,\ldots,n\}$ &emsp;&emsp; *The set of indices of individuals in the front*
/// 6. &emsp; **for** $o \gets 1$ **to** $m$ **do**
/// 7. &emsp;&emsp; $\text{Sort}(i, f_o)$ &emsp;&emsp; *Sort the indices according to the $o$-th objective*
/// 8. &emsp;&emsp; $d_1 \gets d_n \gets \infty$ &emsp;&emsp; *Boundary points are assigned infinite distance*
/// 9. &emsp; **for** $j \gets 2$ **to** $n-1$ **do**
/// 10. &emsp;&emsp; **if** $f_{o}^{max} =\not f_{o}^{min}$ **then**
/// 11. &emsp;&emsp;&emsp; $a \gets i_{j+1}$ &emsp;&emsp; *The index of the next sorted individual*
/// 12. &emsp;&emsp;&emsp; $b \gets i_j$ &emsp;&emsp; *The index of the current sorted individual*
/// 13. &emsp;&emsp;&emsp; $c \gets i_{j-1}$ &emsp;&emsp; *The index of the previous sorted individual*
/// 14. &emsp;&emsp;&emsp; $d_b \gets d_b + \left(f_{o}^{a} - f_{o}^{c}\right) / \left(f_{o}^{max} - f_{o}^{min}\right)$
/// 15. &emsp; **return** $d$
/// ---
pub fn crowding_distance<T: Dominate>(values: &[&T]) -> Vec<f64> {
    let n = values.len();
    let m = values[0].get_max_objectives();

    let mut distances = vec![0.0; n];
    let mut sorted_indices: Vec<usize> = (0..n).collect();

    for i in 0..m {
        sorted_indices.sort_by(|&a, &b| {
            values[a]
                .get_objective_by_index(i)
                .total_cmp(&values[b].get_objective_by_index(i))
        });
        distances[sorted_indices[0]] = f64::INFINITY;
        distances[sorted_indices[n - 1]] = f64::INFINITY;

        let min_value = values[sorted_indices[0]].get_objective_by_index(i);
        let max_value = values[sorted_indices[n - 1]].get_objective_by_index(i);
        for j in 1..(n - 1) {
            if max_value != min_value {
                distances[sorted_indices[j]] += (values[sorted_indices[j + 1]]
                    .get_objective_by_index(i)
                    - values[sorted_indices[j - 1]].get_objective_by_index(i))
                    / (max_value - min_value);
            }
        }
    }
    distances
}

/// Implements the NSGA-II selection operator, which selects candidates based on their non-dominated sorting and crowding distance.
/// From the paper [Deb et al. (2002)](https://ieeexplore.ieee.org/document/996017).
#[derive(Serialize, Deserialize, Debug)]
pub struct NSGA2Selector;

/// The NSGA-II selection operator selects candidates based on their non-dominated sorting and crowding distance.
/// The selection process is as follows:
/// 1. Perform [non dominated sort](NonDominatedSorting::non_dominated_sort) on the input values to obtain the fronts.
/// 2. Iterate through the fronts and add candidates to the selection until the desired size is reached.
/// 3. If the current front cannot be fully added to the selection, compute the [crowding distance](crowding_distance)
///    for the candidates in the current front and select the remaining candidates
///
/// # Pseudo-code
///
/// **NSGA-II Selection**
/// ---
/// **Inputs**
/// 1. &emsp; $\mathcal{B}$ &emsp;&emsp; *An iterable of [`Dominate`] object.*
/// 2. &emsp; $N$ &emsp;&emsp; *The number of candidates to select.*
/// 3.
/// 4. &emsp; $F \gets $[non_dominated_sort](NonDominatedSorting::non_dominated_sort)$(\mathcal{B})$ &emsp;&emsp; *The set of non-dominated fronts*
/// 5. &emsp; $C \gets \emptyset$ &emsp;&emsp; *The set of selected candidates, initially empty*
/// 6. &emsp; $i \gets 1$ &emsp;&emsp; *The index of the current front*
/// 7. &emsp; **while** $|C| + |F_i| \leq N$ **do**
/// 8. &emsp;&emsp; $C \gets C \cup F_i$ &emsp;&emsp; *Add the current front to the selection*
/// 9. &emsp;&emsp; $i \gets i + 1$
/// 10. &emsp; **if** $|C| < N$ **then**
/// 11. &emsp;&emsp; $r \gets N - |C|$ &emsp;&emsp; *The number of remaining candidates to select*
/// 12. &emsp;&emsp; $d \gets $[crowding_distance]$(F_i)$ &emsp;&emsp; *Compute the crowding distance for the current front*
/// 13. &emsp;&emsp; $C \gets C \cup \text{Top}_r(F_i, d)$ &emsp;&emsp; *Select the top $r$ candidates from the current front based on maximum crowding distance*
/// 14. &emsp; **return** $C$
/// ---
impl CandidateSelector for NSGA2Selector {
    fn select_candidates<'a, T: Dominate>(&self, values: &'a mut [T], size: usize) -> Vec<&'a T> {
        if values.len() == size {
            values.iter().collect()
        } else if values.len() < size {
            panic!(
                "The number of values must be greater than or equal to the number of candidates to select."
            );
        } else {
            let fronts = values.non_dominated_sort();
            let mut candidates = Vec::new();
            for front in fronts {
                if candidates.len() + front.len() <= size {
                    candidates.extend(front);
                } else {
                    let remaining = size - candidates.len(); // Number of candidates remaining to select
                    // Compute crowding distances for the current front
                    let dist = crowding_distance(&front);
                    // Associated front elements and distances
                    let mut front_with_distances: Vec<_> =
                        front.into_iter().zip(dist.into_iter()).collect();
                    // Get remaining best candidates based on crowding distance | no need to sort
                    front_with_distances
                        .select_nth_unstable_by(remaining, |a, b| a.1.total_cmp(&b.1).reverse());
                    // Unzip the selected candidates and their distances
                    let (front, _) = front_with_distances
                        .into_iter()
                        .unzip::<&T, f64, Vec<&T>, Vec<f64>>();
                    // Take the top `remaining` candidates from the current front and add them to the final list of candidates
                    candidates.extend(front.into_iter().take(remaining));
                    break;
                }
            }
            candidates
        }
    }

    fn arg_select_candidates<T: Dominate>(&self, values: &[T], size: usize) -> Vec<usize> {
        if values.len() == size {
            (0..values.len()).collect()
        } else if values.len() < size {
            panic!(
                "The number of values must be greater than or equal to the number of candidates to select."
            );
        } else {
            let fronts = values.non_dominated_argsort();
            let mut candidates = Vec::new();
            for front in fronts {
                if candidates.len() + front.len() <= size {
                    candidates.extend(front);
                } else {
                    let remaining = size - candidates.len(); // Number of candidates remaining to select
                    // Compute crowding distances for the current front
                    let sorted = front.iter().map(|&idx| &values[idx]).collect::<Vec<_>>();
                    let dist = crowding_distance(&sorted);
                    // Associated front elements and distances
                    let mut front_with_distances: Vec<_> =
                        front.into_iter().zip(dist.into_iter()).collect();
                    // Get remaining best candidates based on crowding distance | no need to sort
                    front_with_distances
                        .select_nth_unstable_by(remaining, |a, b| a.1.total_cmp(&b.1).reverse());
                    // Unzip the selected candidates and their distances
                    let (front, _) = front_with_distances
                        .into_iter()
                        .unzip::<usize, f64, Vec<usize>, Vec<f64>>();
                    // Take the top `remaining` candidates from the current front and add them to the final list of candidates
                    candidates.extend(front.into_iter().take(remaining));
                    break;
                }
            }
            candidates
        }
    }
}
