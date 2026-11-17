# Tantale macros

`tantale_macros` provides procedural macros for declaring Tantale outcomes,
objective functions, and search spaces. The crate-level Rust documentation in
[`src/lib.rs`](src/lib.rs) is the complete reference, including syntax,
generated items, feature requirements, and examples.

Most applications should use the re-exports from the `tantale` facade crate:

```toml
[dependencies]
tantale = "0.3.2"
```

```rust,ignore
use tantale::macros::{Outcome, Recordable, Schema, objective};
```

Main macros:

- `Outcome` derives a result codomain from field attributes such as
  `#[maximize]`, `#[minimize]`, `#[constraint]`, `#[cost]`, and `#[step]`.
- `objective!` declares an objective and its search space in one function.
- `hpo!` declares a search space independently of an objective.
- `Schema` and `Recordable` derive column names and string values from
  supported named-field structs.
- `pyhpo!` generates a Python-compatible search-space module when the `py`
  feature is enabled.

The `spikes` feature adds the `#[samples]` and `#[spiking]` attributes to
`Outcome`. The `basic_bounds` and `fidelity_bounds` attribute macros are
primarily intended for generic Tantale algorithm implementations.

Outcome structs also need `Schema`, `Recordable`, `Serialize`, and
`Deserialize` implementations. The common pattern is to derive all five
together. `objective!` and `hpo!` generate names in their invocation module;
place multiple declarations in distinct modules to avoid name collisions.

## License

CeCILL-C. See
[cecill.info](https://cecill.info/licences/Licence_CeCILL-C_V1-en.html).
