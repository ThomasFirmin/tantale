//! A [`NoDomain`] is a degenerate domain that cannot be sampled from. 
//! It is used as a placeholder when the optimizer-facing [`Domain`](crate::Domain) is the same as the objective-facing [`Domain`](crate::Domain).

use crate::domain::PreDomain;

/// A unsamplable empty-domain.
#[derive(Debug)]
pub struct NoDomain;
impl PreDomain for NoDomain {}

impl NoDomain {
    pub fn new() -> Self {
        NoDomain
    }
}

impl Default for NoDomain {
    fn default() -> Self {
        Self::new()
    }
}
