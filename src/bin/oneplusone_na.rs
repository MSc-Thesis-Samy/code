use neuroevolution::one_plus_one_na::*;
use neuroevolution::network::Network;

fn main() {
    let mut network = Network::<2, 2>::new();

    network.optimize(half, 3000);
    println!("half Fitness: {}", half(&network));
    print!("{}", network);

    network.optimize(quarter, 3000);
    println!("quarter Fitness: {}", quarter(&network));
    print!("{}", network);

    network.optimize(two_quarters, 3000);
    println!("two_quarters Fitness: {}", two_quarters(&network));
    print!("{}", network);

    network.optimize(square, 3000);
    println!("square Fitness: {}", square(&network));
    print!("{}", network);

    let mut network = Network::<4, 3>::new();
    network.optimize(cube, 3000);
    println!("cube Fitness: {}", cube(&network));
    print!("{}", network);
}
