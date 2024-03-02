pub trait NeuroevolutionAlgorithm {
    fn optimize(&mut self, evaluation_function: fn(&Self) -> f64, n_iters: u32);
    fn evaluate(&self, input: &Vec<f64>) -> bool;
}
