pub mod obj;
pub use obj::{Objective, SimpleObjective};

pub mod criteria;
pub use criteria::{Criteria, Max, Min, Lambda};

pub mod codomain;
pub use codomain::Codomain;

pub mod outcome;
pub use outcome::{Outcome, HashOut};