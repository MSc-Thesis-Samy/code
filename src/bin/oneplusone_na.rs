use neuroevolution::benchmarks::*;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::traits::NeuroevolutionAlgorithm;
use neuroevolution::constants::*;

fn main() {
    let mut network = Network::new(2, 2);
    network.optimize(half, N_ITERATIONS);
    println!("half fitness: {:.2}", half(&network));
    print!("{:.2}", network);
    network.optimize(quarter, N_ITERATIONS);
    println!("quarter fitness: {:.2}", quarter(&network));
    print!("{:.2}", network);
    network.optimize(two_quarters, N_ITERATIONS);
    println!("two_quarters fitness: {:.2}", two_quarters(&network));
    print!("{:.2}", network);
    network.optimize(square, N_ITERATIONS);
    println!("square fitness: {:.2}", square(&network));
    print!("{:.2}", network);

    let mut network = Network::new(4, 3);
    network.optimize(cube, N_ITERATIONS);
    println!("cube fitness: {:.2}", cube(&network));
    print!("{:.2}", network);

    let mut network = DiscreteNetwork::new(RESOLUTION, 1, 2);
    network.optimize(half, N_ITERATIONS);
    println!("half fitness: {:.2}", half(&network));
    print!("{:.2}", network);
    network.optimize(quarter, N_ITERATIONS);
    println!("quarter fitness: {:.2}", quarter(&network));
    print!("{:.2}", network);
    network.optimize(two_quarters, N_ITERATIONS);
    println!("two_quarters fitness: {:.2}", two_quarters(&network));
    print!("{:.2}", network);

    let mut network = Network::new(1, 2);
    network.optimize_cmaes(half);
    println!("half fitness: {:.2}", half(&network));
    print!("{:.2}", network);
}
