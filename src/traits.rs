pub trait NetworkTrait<const N: usize, const D: usize>{
    fn optimize(&mut self, evaluation_function: fn(&Self) -> f64, n_iters: u32);
    fn evaluate(&self, input: &[f64; D]) -> bool;
}

pub trait VNeuronTrait {
    fn optimize(&mut self, evaluation_function: fn(&Self) -> f64, n_iters: u32);
    fn evaluate(&self, input: &Vec<f64>) -> bool;
}
