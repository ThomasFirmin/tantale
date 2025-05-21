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

use crate::domain::derrors::DomainError;
use crate::domain::Domain;

pub type OntoOutput<Tgt> = Result<<Tgt as Domain>::TypeDom, DomainError>;

pub trait Onto<Tgt: Domain>: Domain {
    /// [`Onto`] is a surjective function to map a point from an input [`Domain`] to an output [`Domain`].
    /// If [`Self`] is equal to the targetted domain, then the input `item` should be cloned.
    /// By default if the input and targetted domain are the same (same pointer), returns a clone of `item`.
    ///
    /// # Parameters
    ///
    /// * `item` : `&<`[`Self`]` as `[`Domain`]``>::`[`TypeDom`](Domain::TypeDom) - A borrowed point from the [`Self`] domain to map to the `target` [`Domain`].
    /// * `target` : `&`[`Domain`] - A borrowed targetted [`Domain`].
    ///
    /// # Errors
    ///
    /// * Returns a [`DomainError::OoB`]
    ///     * if input `item` to be mapped is not into [`Self`] domain.
    ///     * if resulting mapped `item` is not into the `target` domain.
    ///
    fn onto(&self, _item: &Self::TypeDom, _target: &Tgt) -> OntoOutput<Tgt>;
}
