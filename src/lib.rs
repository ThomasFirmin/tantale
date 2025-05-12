//! # Tantale
//!
//! Tantale is a library dedicated to AutoML containing utilitaries to build search spaces, objective functions, algorithms, and parallelization. It is a core library with very few algorithms.
//!

#[doc(inline)]
pub use macros::Mixed;
#[doc(inline)]
pub use tantale_core as core;
#[doc(inline)]
pub use tantale_macros as macros;

// // Re-exports
// #[doc(hidden)]
// pub use quote::ToTokens;
// #[doc(hidden)]
// pub use syn;
