use neuroevolution::one_plus_one_na::*;

fn evaluation_function<const N: usize>(network: &Network<N>) -> f64 {
    1.
}

fn main() {
    let mut network = Network::<1>::new();
    network.optimize(evaluation_function, 10);
    println!("{:?}", network);
}
