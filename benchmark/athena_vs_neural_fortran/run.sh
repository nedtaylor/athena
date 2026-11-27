#!/bin/sh
# Build and run the matched Athena and neural-fortran benchmarks.
set -e
cd "$(dirname "$0")"
MNIST_DIR=${MNIST_DIR:-../../../MNIST-txt}
FLAGS='--profile release --flag "-O3 -fopenmp" --link-flag "-fopenmp"'

eval fpm run benchmark_simple_athena $FLAGS
eval fpm run benchmark_simple_neural_fortran $FLAGS
eval fpm run benchmark_sine_athena $FLAGS
eval fpm run benchmark_sine_neural_fortran $FLAGS

if [ -f "$MNIST_DIR/MNIST_train.txt" ]; then
   echo "MNIST data: $MNIST_DIR"
   eval fpm run benchmark_athena $FLAGS -- "'$MNIST_DIR'"
   eval fpm run benchmark_neural_fortran $FLAGS -- "'$MNIST_DIR'"
else
   echo "MNIST data not found at $MNIST_DIR/MNIST_train.txt; skipping MNIST benchmarks"
fi
