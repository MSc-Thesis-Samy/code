use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(about = "Neuroevolution framework", long_about = None)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[arg(help = "The algorithm to test", value_enum)]
    pub algorithm: Algorithm,
    #[arg(help = "the benchmark problem", value_enum)]
    pub problem: Problem,
    #[arg(help = "Resolution, when applicable", short, long, default_value_t = 100)]
    pub resolution: usize,
    #[arg(help = "Number of iterations", short, long, default_value_t = 1000)]
    pub iterations: u32,
    #[arg(help = "Use the continuous version of the algorithm, when applicable", short, long)]
    pub continuous: bool,
    #[arg(help = "Optimize using cma-es", short, long)]
    pub es: bool,
    #[arg(help = "Number of neurons, when applicable", short, long, default_value_t = 1)]
    pub neurons: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Algorithm {
    Oneplusonena,
    Bna,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Problem {
    Half,
    Quarter,
    Twoquartes,
}
