use rayon::prelude::*;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::neuroevolution_algorithm::*;
use neuroevolution::constants::*;
use neuroevolution::benchmarks::*;

fn main() {
    let fitness_mean = (0..N_RUNS).into_par_iter().map(|_| {
        let network = DiscreteNetwork::new(RESOLUTION, 1, 2);
        let mut algorithm = Algorithm::DiscreteOneplusoneNA(network);
        let problem = Benchmark::new(Problem::Half);
        algorithm.optimize(&problem, N_ITERATIONS);
        problem.evaluate(&algorithm)
    }).sum::<f64>() / N_RUNS as f64;

    println!("Mean fitness: {:.2}", fitness_mean);
}
