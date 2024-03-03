use neuroevolution::benchmarks::*;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::traits::NeuroevolutionAlgorithm;

fn main() {
    let mut network = Network::new(1, 2);

    network.optimize(half, 1000);
    println!("half Fitness: {}", half(&network));
    print!("{}", network);

    // network.optimize(quarter, 1000);
    // println!("quarter Fitness: {}", quarter(&network));
    // print!("{}", network);
    //
    // network.optimize(two_quarters, 1000);
    // println!("two_quarters Fitness: {}", two_quarters(&network));
    // print!("{}", network);
    //
    // network.optimize(square, 1000);
    // println!("square Fitness: {}", square(&network));
    // print!("{}", network);
    //
    // let mut network = Network::<4, 3>::new();
    // network.optimize(cube, 1000);
    // println!("cube Fitness: {}", cube(&network));
    // print!("{}", network);

    let mut network = DiscreteNetwork::new(100, 1, 2);

    network.optimize(half, 1000);
    println!("half Fitness: {}", half(&network));
    print!("{}", network);

    let mut network = Network::new(1, 2);
    network.optimize_cmaes(half);
    println!("half Fitness: {}", half(&network));
    print!("{}", network);
}
