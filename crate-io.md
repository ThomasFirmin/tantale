# Tantale crate quick start

Add Tantale to a Rust 2024 project:

```console
cargo add tantale
```

The minimum supported Rust version is 1.91.1.

Or add the dependency directly to `Cargo.toml`:

```toml
[dependencies]
tantale = "0.3.2"
```

Enable optional integrations as needed:

```toml
[dependencies]
tantale = { version = "0.3.2", features = ["mpi", "py", "bayes"] }
```

The `mpi` feature requires an MPI implementation. The `py` feature enables the
Python bridge; Python examples and requirements are covered in the
[Python objective integration guide](src/python_integration.md).

Start with the [Quick start tutorial](src/quick_start.md) for a complete
objective, search space, sampler, stopping criterion, and experiment. The
repository README also contains a self-contained Rust example.
