use neuroevolution::oneplusone_na::*;
use neuroevolution::network::Network;
use neuroevolution::discrete_network::DiscreteNetwork;
use neuroevolution::network_trait::NetworkTrait;

fn main() {
    let mut network = Network::<1, 2>::new();

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

    let mut network = DiscreteNetwork::<1, 2>::new(1000);

    network.optimize(half, 1000);
    println!("half Fitness: {}", half(&network));
    print!("{}", network);
}
