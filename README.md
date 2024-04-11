# MSc Thesis Code

## Usage

```
Neuroevolution framework for testing algorithms on benchmark problems.

Usage: main [OPTIONS] <ALGORITHM> <PROBLEM>

Arguments:
  <ALGORITHM>  The algorithm to test [possible values: oneplusonena, bna, neat]
  <PROBLEM>    the benchmark problem [possible values: half, quarter, two-quarters, xor]

Options:
  -r, --resolution <RESOLUTION>  Resolution, when applicable [default: 1000]
  -i, --iterations <ITERATIONS>  Number of iterations [default: 1000]
  -c, --continuous               Use the continuous version of the algorithm, when applicable
  -e, --es                       Optimize using cma-es
  -n, --neurons <NEURONS>        Number of neurons, when applicable [default: 1]
  -g, --gui                      Display visualization
  -h, --help                     Print help
  -V, --version                  Print version
```
