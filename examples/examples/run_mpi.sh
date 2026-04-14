#!/usr/bin/env bash
set -euo pipefail

BINARY_DIR="$(cd "$(dirname "$0")/../../target/debug/examples" && pwd)"
NP=4

examples=(
    mpi_test
    mpi_quick_start
    mpi_seq_run
    mpi_seq_run_fid
    mpi_seq_run_fid_loadpool
    mpi_seq_evaluator
    mpi_seq_evaluator_fid
    mpi_batch_run
    mpi_batch_run_fid
    mpi_batch_run_fid_loadpool
    mpi_batch_evaluator
    mpi_batch_evaluator_fid
    mpi_mo_example
    mpi_test_pytantale_objective_async
    mpi_test_pytantale_objective_batch
    mpi_test_pytantale_stepped_async
    mpi_test_pytantale_stepped_batch
)

for example in "${examples[@]}"; do
    binary="$BINARY_DIR/$example"
    if [[ ! -f "$binary" ]]; then
        echo "ERROR: binary not found: $binary" >&2
        exit 1
    fi
    echo "========================================"
    echo "Running: mpiexec -n $NP $binary"
    echo "========================================"
    if ! mpiexec -n "$NP" "$binary"; then
        echo "" >&2
        echo "ERROR: '$example' exited with a non-zero status." >&2
        exit 1
    fi
    echo ""
done

echo "All MPI examples completed successfully."
