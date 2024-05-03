use std::time::Instant;
use rayon::prelude::*;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::neuroevolution_algorithm::*;
use neuroevolution::constants::*;
use neuroevolution::benchmarks::*;

fn main() {
    let (fitness_mean, elapsed_mean) = (0..N_RUNS).into_par_iter().map(|_| {
        let network = DiscreteNetwork::new(RESOLUTION, 1, 2);
        let mut algorithm = Algorithm::DiscreteOneplusoneNA(network);
        let problem = Benchmark::new(Problem::Half);

        let start = Instant::now();
        algorithm.optimize(&problem, N_ITERATIONS);
        let elapsed = start.elapsed().as_secs_f64();

        (problem.evaluate(&algorithm), elapsed)
    }).reduce(|| (0., 0.), |a, b| (a.0 + b.0, a.1 + b.1));

    println!("Mean fitness: {:.2}", fitness_mean / N_RUNS as f64);
    println!("Mean elapsed time: {:.3} s", elapsed_mean / N_RUNS as f64);
}
