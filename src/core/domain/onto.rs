use crate::core::domain::Domain;
use crate::core::domain::DomainError;
pub trait Onto<Out: Domain>: Domain {
    /// [`Onto`] is a surjective function to map a point from an input [`Domain`] to an output [`Domain`].
    /// If [`Self`] is equal to the targetted domain, then the input `item` should be cloned.
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
    fn onto(&self, _item: &Self::TypeDom, _target: &Out) -> Result<Out::TypeDom, DomainError>;
}