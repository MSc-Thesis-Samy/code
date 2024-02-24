use std::f64::consts::PI;
use neuroevolution::network::Network;
use neuroevolution::vneuron::*;
use neuroevolution::bna::*;

fn main() {
    // let mut vneuron = VNeuron::new(2);
    // vneuron.optimize(half, 1000);
    // println!("half fitness: {}", half(&vneuron));
    // println!("Half: {}", vneuron);

    let angle = PI / 2.;
    let bias = 0.5;
    let bend = 2. * PI / 3.;
    let network = VNeuron::from_params(2, bias, vec![angle], bend);
    println!("half fitness: {:.2}", half(&network));
    println!("Half: {}", network);
}
