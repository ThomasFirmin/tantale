# Tantale facade documentation

The facade crate re-exports the `tantale_core`, `tantale_algos`, and
`tantale_macros` crates as `tantale::core`, `tantale::algos`, and
`tantale::macros`. The optional Python bridge is available as
`tantale::python` with the `py` feature.

Start with the [Quick start](quick_start.md), then see the
[random search](rs_example.md), [grid search](gs_example.md), and
[multi-objective MO-ASHA](mo_example.md) tutorials. The advanced guides explain
the current sampler interface: [sequential samplers](create_seq_optimizer.md)
and [batched samplers](create_batch_optimizer.md). Python-backed objectives
are described in the [Python integration guide](python_integration.md).
[Burn](tantale_x_burn.md) and [PyTorch](tantale_x_pytorch.md) tutorials
contain model-training examples.

The API-level documentation for domains, outcomes, experiment runners, and
algorithm traits is in the re-exported `core` and `algos` modules on docs.rs.
