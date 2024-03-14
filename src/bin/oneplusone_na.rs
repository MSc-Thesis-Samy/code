use neuroevolution::benchmarks::*;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::neuroevolution_algorithm::*;
use neuroevolution::constants::*;

fn main() {
    let mut network = Network::new(2, 2);
    network.optimize(half, N_ITERATIONS);
    println!("half fitness: {:.2}", half(&Algorithm::ContinuousOneplusoneNA(&mut network)));
    print!("{:.2}", network);
    network.optimize(quarter, N_ITERATIONS);
    println!("quarter fitness: {:.2}", quarter(&Algorithm::ContinuousOneplusoneNA(&mut network)));
    print!("{:.2}", network);
    network.optimize(two_quarters, N_ITERATIONS);
    println!("two_quarters fitness: {:.2}", two_quarters(&Algorithm::ContinuousOneplusoneNA(&mut network)));
    print!("{:.2}", network);
    network.optimize(square, N_ITERATIONS);
    println!("square fitness: {:.2}", square(&Algorithm::ContinuousOneplusoneNA(&mut network)));
    print!("{:.2}", network);

    let mut network = Network::new(4, 3);
    network.optimize(cube, N_ITERATIONS);
    println!("cube fitness: {:.2}", cube(&Algorithm::ContinuousOneplusoneNA(&mut network)));
    print!("{:.2}", network);

    let mut network = DiscreteNetwork::new(RESOLUTION, 1, 2);
    network.optimize(half, N_ITERATIONS);
    println!("half fitness: {:.2}", half(&Algorithm::DiscreteOneplusoneNA(&mut network)));
    print!("{:.2}", network);
    network.optimize(quarter, N_ITERATIONS);
    println!("quarter fitness: {:.2}", quarter(&Algorithm::DiscreteOneplusoneNA(&mut network)));
    print!("{:.2}", network);
    network.optimize(two_quarters, N_ITERATIONS);
    println!("two_quarters fitness: {:.2}", two_quarters(&Algorithm::DiscreteOneplusoneNA(&mut network)));
    print!("{:.2}", network);

    let mut network = Network::new(1, 2);
    network.optimize_cmaes(half);
    println!("half fitness: {:.2}", half(&Algorithm::ContinuousOneplusoneNA(&mut network)));
    print!("{:.2}", network);
}
