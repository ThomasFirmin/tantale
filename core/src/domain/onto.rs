//! [`Onto`] traits for [`Domains`](Domain) is used to map a point sampled from an input [`Domain`]
//! onto an output [`Domain`]. For example, mapping $0.5 \in [0.0, 1.0]$ to $[0.0,100.0]$, $f(0.5)=50.0$.
//! Or mapping an [`Int`](crate::domain::Int) sample onto a [`Cat`](crate::domain::Cat) value, $1 \in [0,2]$ to `["relu", "tanh", "sigmoid"]`,
//! $f(1) = \texttt{"tanh"}$.
//!
//! # Examples
//!
//! ```
//! use tantale::core::{Bounded, Domain, DomainBounded};
//! let dom : Bounded<u8> = Bounded::new(0, 255);
//!
//! let mut rng = rand::rng();
//! let sample = dom.sample(&mut rng);
//! assert!(dom.is_in(&sample));
//! assert_eq!(dom.lower(), 0);
//! assert_eq!(dom.upper(), 255);
//! assert_eq!(dom.mid(), 127);
//! assert_eq!(dom.width(), 255);
//! ```

use crate::{Domain, errors::OntoError};

pub trait Linked {
    type Obj: Domain;
    type Opt: Domain;
}

pub type LinkObj<T> = <T as Linked>::Obj;
pub type LinkOpt<T> = <T as Linked>::Opt;
pub type LinkTyObj<T> = <<T as Linked>::Obj as Domain>::TypeDom;
pub type LinkTyOpt<T> = <<T as Linked>::Opt as Domain>::TypeDom;

/// [`Onto`] is a surjective function to map a point from an element of `Self` [`Item`](Onto::Item), to an element of `Target` [`TargetItem`](Onto::TargetItem)
/// It is mostly used to map [`TypeDom`](Domain::TypeDom) to another [`TypeDom`](Domain::TypeDom), using target [`Domain`].
/// See [`OntoDom`] for more information.
pub trait Onto<Target> {
    type TargetItem;
    type Item;
    /// # Parameters
    ///
    /// * `item` : `&`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Domain`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`OntoError`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, item: &Self::Item, target: &Target) -> Result<Self::TargetItem, OntoError>;
}

/// A trait defining a [`Domain`] for which it's [`TypeDom`](Domain::TypeDom) can be mmaped onto another [`TypeDom`](Domain::TypeDom)
/// from another [`Domain`] `B`.
pub trait OntoDom<B>: Domain
where
    Self: Onto<B, TargetItem = B::TypeDom, Item = Self::TypeDom>,
    B: Domain,
{
}
