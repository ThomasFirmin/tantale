
#[cfg(doc)]
use crate::core::solution::Solution;

/// # Elements
/// This crate describes a single element from a [`Solution`].
///
use crate::core::domain::{Domain,Bounded,Bool,Cat};
use crate::core::Variable;
use crate::core::domain::errors_domain::{DomainError};

use std::rc::Rc;
use rand::distr::uniform::SampleUniform;
use rand::rngs::ThreadRng;
use num::{Num,NumCast};
use std::fmt::{Display,Debug};

pub struct Elements;