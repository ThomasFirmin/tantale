use tantale_core::Dominate;
use std::cmp::Ordering;

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
pub trait NonDominatedSorting<T> 
where
    T: Dominate
{
    fn non_dominated_sort(&mut self) -> Vec<Vec<&T>>;
}

impl<T:Dominate> NonDominatedSorting<T> for [T]
{
    fn non_dominated_sort(&mut self) -> Vec<Vec<&T>>
    {
        self.sort_by(|a,b|
            {
                let length: usize = a.get_max_objectives();
                let mut i = 0;
                let mut ord = a.get_objective_by_index(0).total_cmp(&b.get_objective_by_index(0)).reverse();
                while let Ordering::Equal = ord {
                    i += 1;
                    if i < length {
                        ord = a.get_objective_by_index(i).total_cmp(&b.get_objective_by_index(i)).reverse();
                    } else {
                        break
                    }
                }
                ord
            }
        );
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
}

/// Implements the binary search used for non-dominated sorting, described in [Zhang et al. (2021)](http://www.soft-computing.de/ENS.pdf).
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
pub fn front_binary_search<T: Dominate>(target: &T, fronts: &[Vec<&T>]) -> usize {

    let n_fronts = fronts.len();
    let mut k_min = 0;
    let mut k_max = n_fronts;
    let mut k = k_max.div_ceil(2);

    loop {
        let current_front = &fronts[k-1];
        let one_dominates = current_front.iter().any(|x| x.dominates(target));

        if one_dominates {
            k_min = k;
            if k_max == k_min + 1 && k_max < n_fronts {
                return k_max;
            } else if k_min == n_fronts {
                return n_fronts + 1;
            } else {
                k = (k_min + k_max).div_ceil(2);
            }
        }
        else if k == k_min + 1 {
            return k;
        } else {
            k_max = k;
            k = (k_min + k_max).div_ceil(2);
        }
    }
}

pub trait 