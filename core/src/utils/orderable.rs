//! Utilities for ordering and sorting objects based on their objectives, including lexicographic sorting and comparison traits.
//! The [`Orderable`] trait provides a unified interface for comparing objects, 
//! allowing for flexible sorting and ordering based on custom criteria. 

use crate::Dominate;

use std::cmp::Ordering;
use serde::{Deserialize, Serialize};


/// Lexicographically sorts the input objects in descending order of their objectives.
pub fn lexsort<T: Dominate>(input: &mut [T]) {
    input.sort_by(|a, b| a.ord_cmp(b).unwrap());
}

/// Lexicographically sorts the input objects in descending order of their objectives and returns the indices of the sorted objects.
pub fn arg_lexsort<T: Dominate>(input: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..input.len()).collect();
    indices.sort_by(|&a, &b| {
        let length: usize = input[a].len_objectives();
        let mut i = 0;
        let mut ord = input[a]
            .get_objective_by_index(0)
            .total_cmp(&input[b].get_objective_by_index(0));
        while let Ordering::Equal = ord {
            i += 1;
            if i < length {
                ord = input[a]
                    .get_objective_by_index(i)
                    .total_cmp(&input[b].get_objective_by_index(i));
            } else {
                break;
            }
        }
        ord
    });
    indices
}

/// A trait for types that can be ordered based on a custom comparison function.
/// 
/// # Example
/// ```rust
/// use tantale_core::utils::Orderable;
/// let a = vec![1, 2, 3];
/// let b = vec![1, 2, 4];
/// // Lexicographic order
/// assert!(a.ord_lt(&b));
/// assert_eq!(a.ord_cmp(&b), Some(std::cmp::Ordering::Less));
/// ```
pub trait Orderable
{
    fn ord_cmp(&self, other: &Self) -> Option<Ordering>;

    /// Returns `true` if `self` is lexicographically less than `other`.
    fn ord_lt(&self, other: &Self) -> bool{
        self.ord_cmp(other).is_some_and(Ordering::is_lt)
    }

    /// Returns `true` if `self` is lexicographically less than or equal to `other`.
    fn ord_le(&self, other: &Self) -> bool{
        self.ord_cmp(other).is_some_and(Ordering::is_le)
    }

    /// Returns `true` if `self` is lexicographically greater than `other`.
    fn ord_gt(&self, other: &Self) -> bool{
        self.ord_cmp(other).is_some_and(Ordering::is_gt)
    }

    /// Returns `true` if `self` is lexicographically greater than or equal to `other`.
    fn ord_ge(&self, other: &Self) -> bool{
        self.ord_cmp(other).is_some_and(Ordering::is_ge)
    }
}

impl<T: PartialOrd> Orderable for [T]{
    /// Compares two slices lexicographically.
    /// Returns `Some(Ordering)` if the slices have the same length, otherwise returns `None`.
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.len() != other.len() {
            return None;
        }
        for i in 0..self.len() {
            let x = &self[i];
            let y = &other[i];
            match x.partial_cmp(y) {
                Some(Ordering::Less) => return Some(Ordering::Less),
                Some(Ordering::Greater) => return Some(Ordering::Greater),
                Some(Ordering::Equal) => continue,
                None => return None,
            }
        }
        Some(Ordering::Equal)
    }
}

impl<T: PartialOrd> Orderable for &[T]{
    /// Compares two slices lexicographically.
    /// Returns `Some(Ordering)` if the slices have the same length, otherwise returns `None`.
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.len() != other.len() {
            return None;
        }
        for i in 0..self.len() {
            let x = &self[i];
            let y = &other[i];
            match x.partial_cmp(y) {
                Some(Ordering::Less) => return Some(Ordering::Less),
                Some(Ordering::Greater) => return Some(Ordering::Greater),
                Some(Ordering::Equal) => continue,
                None => return None,
            }
        }
        Some(Ordering::Equal)
    }
}

impl<T: Orderable> Orderable for &T {
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).ord_cmp(*other)
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
pub struct OrdView<T: PartialOrd>(pub T);

impl<T: PartialOrd> Orderable for OrdView<T> {
    fn ord_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: PartialOrd> PartialEq<T> for OrdView<T> {
    fn eq(&self, other: &T) -> bool {
        self.0.eq(other)
    }
}

pub trait IntoOrdView {
    type Output: PartialOrd;
    fn into_ord_view(self) -> OrdView<Self::Output>;
}

impl <T: PartialOrd> IntoOrdView for T {
    type Output = T;
    fn into_ord_view(self) -> OrdView<Self::Output> {
        OrdView(self)
    }
}