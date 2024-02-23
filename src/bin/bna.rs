use std::f64::consts::PI;
use neuroevolution::vneuron::*;

fn main() {
    let input = vec![1., PI / 8.];
    let dim = 2;
    let bend = PI / 4.;
    let angles = vec![PI / 4.];
    let bias = 0.;
    let vneuron = VNeuron::from_params(dim, bias, angles, bend);
    println!("{:?}", vneuron.evaluate(&input));
}
