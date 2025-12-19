use crate::domain::PreDomain;

pub struct NoDomain;
impl PreDomain for NoDomain{}

impl NoDomain {
    pub fn new() -> Self{
        NoDomain
    }
}

impl Default for NoDomain {
    fn default() -> Self {
        Self::new()
    }
}