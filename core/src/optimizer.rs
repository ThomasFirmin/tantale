pub mod opt;
pub use opt::{ArcVecArc, EmptyInfo, OptInfo, OptState, Optimizer,PBType,CBType,OBType};

pub mod batchtype;
pub use batchtype::{BatchType,Batch,Single,CompBatch,CompSingle};
pub mod parallel;