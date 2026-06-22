use tantale_core::{Dominate, NonDominatedSorting};

use serde::{Deserialize, Serialize};

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
    let m = values[0].len_objectives();

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
