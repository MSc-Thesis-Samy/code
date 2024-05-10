use clap::{Parser, ValueEnum};
use crate::constants::*;
use crate::benchmarks::Problem;

#[derive(Parser)]
#[command(about = "Neuroevolution framework", long_about = None)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[arg(help = "The algorithm to test", value_enum)]
    pub algorithm: AlgorithmType,
    #[arg(help = "the benchmark problem", value_enum)]
    pub problem: Problem,
    #[arg(help = "Resolution, when applicable", short, long, default_value_t = RESOLUTION)]
    pub resolution: usize,
    #[arg(help = "Number of iterations", short, long, default_value_t = N_ITERATIONS)]
    pub iterations: u32,
    // #[arg(help = "Use the continuous version of the algorithm, when applicable", short, long)]
    // pub continuous: bool,
    // #[arg(help = "Optimize using cma-es", short, long)]
    // pub es: bool,
    #[arg(help = "Number of neurons, when applicable", short, long, default_value_t = 1)]
    pub neurons: usize,
    #[arg(help = "Display visualization", short, long)]
    pub gui: bool,
    #[arg(help = "Configuration file", short, long)]
    pub file: Option<String>,
    #[arg(help = "Results output file", short, long)]
    pub output: Option<String>,
    #[arg(help = "Number of runs", short, long)]
    pub test_runs: Option<usize>,
    #[arg(help = "Max fitness tolerance", short, long, default_value_t = MAX_FITNESS_TOL)]
    pub error_tol: f64,
    #[arg(help = "Max stagnation", short, long)]
    pub stagnation: Option<u32>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum AlgorithmType {
    Oneplusonena,
    Bna,
    Neat,
    NeuralNetwork,
}
