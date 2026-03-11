# Tantale macros

Procedural macros for the Tantale AutoML library.

This crate provides two kinds of macros:
- **Declarative macros** — `objective!` and `hpo!` for defining search spaces and objective functions.
- **Derive macros** — `Outcome`, `FuncState`, `OptState`, `OptInfo`, `SolInfo`, `CSVWritable` for implementing Tantale traits on user-defined types.

Import this crate via the top-level `tantale` re-export:
```rust,ignore
use tantale::macros::{objective, hpo, Outcome, CSVWritable, FuncState};
```

## License

**CeCILL-C** — see [cecill.info](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html).
