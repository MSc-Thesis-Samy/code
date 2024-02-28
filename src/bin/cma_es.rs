use cmaes::fmax;
use neuroevolution::cma_es::*;
use neuroevolution::network::*;

fn main() {
    const N: usize = 4;

    let solution = fmax(half::<N>, vec![0.; N * 2], 1.);
    let network: Network<N,2> = Network::get_network(&solution.point);
    let fitness = solution.value;
    println!("half Fitness: {}", fitness);
    println!("{}", network);

    let solution = fmax(quarter::<N>, vec![0.; N * 2], 1.);
    let network: Network<N,2> = Network::get_network(&solution.point);
    let fitness = solution.value;
    println!("quarter Fitness: {}", fitness);
    println!("{}", network);

    let solution = fmax(two_quarters::<N>, vec![0.; N * 2], 1.);
    let network: Network<N,2> = Network::get_network(&solution.point);
    let fitness = solution.value;
    println!("two_quarters Fitness: {}", fitness);
    println!("{}", network);

    let solution = fmax(square::<N>, vec![0.; N * 2], 1.);
    let network: Network<N,2> = Network::get_network(&solution.point);
    let fitness = solution.value;
    println!("square Fitness: {}", fitness);
    println!("{}", network);

    let solution = fmax(cube::<N>, vec![0.; N * 3], 1.);
    let network: Network<N,3> = Network::get_network(&solution.point);
    let fitness = solution.value;
    println!("cube Fitness: {}", fitness);
    println!("{}", network);
}
