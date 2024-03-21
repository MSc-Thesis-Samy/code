use std::f64::consts::PI;
use ggez::*;
use crate::neuroevolution_algorithm::*;
use crate::benchmarks::{SphereClassificationProblem, ClassificationProblemEval};

pub struct State {
    alg: Algorithm,
    problem: SphereClassificationProblem,
    n_iters: u32,
    iteration: u32,
}

impl State {
    pub fn new(alg: Algorithm, problem: SphereClassificationProblem, n_iters: u32) -> Self {
        State {
            alg,
            problem,
            n_iters,
            iteration: 0,
        }
    }

    fn to_cartesian(v: &Vec<f64>) -> (f64, f64) {
        let r = v[0];
        let theta = v[1];
        (r * theta.cos(), r * theta.sin())
    }

    fn cartesian_to_canvas(&self, (x, y): (f64, f64)) -> mint::Point2<f32> {
        mint::Point2{x: 400. + 250. * x as f32, y: 300. - 250. * y as f32}
    }

    fn polar_to_canvas(&self, v: &Vec<f64>) -> mint::Point2<f32> {
        let (x, y) = State::to_cartesian(v);
        mint::Point2{x: 400. + 250. * x as f32, y: 300. - 250. * y as f32}
    }

    fn cartesian_rotation((x1, y1): (f64, f64), (x2, y2): (f64, f64), theta: f64) -> (f64, f64) {
        let x = x1 + (x2 - x1) * theta.cos() - (y2 - y1) * theta.sin();
        let y = y1 + (x2 - x1) * theta.sin() + (y2 - y1) * theta.cos();
        (x, y)
    }

    fn get_decision_line_mesh(
        &self,
        mesh: &mut graphics::MeshBuilder,
        bias: f64,
        theta: f64,
        d_normal: f64,
        d_hp: f64) -> GameResult
    {
        let r = (bias * bias + d_hp * d_hp).sqrt();
        let theta1 = theta + (bias.abs() / r).acos();
        let theta2 = theta - (bias.abs() / r).acos();

        // normal
        mesh.line(
            &[
                self.polar_to_canvas(&vec![bias.abs(), theta]),
                self.polar_to_canvas(&vec![bias.abs() + if bias >= 0. {d_normal} else {-d_normal}, theta]),
            ],
            2.0,
            graphics::Color::BLUE,
        )?;

        // hypotenuse
        mesh.line(
            &[
                self.polar_to_canvas(&vec![r, theta1]),
                self.polar_to_canvas(&vec![r, theta2]),
            ],
            2.0,
            graphics::Color::BLACK,
        )?;

        Ok(())
    }

    fn get_bend_decision_mesh(
        &self,
        mesh: &mut graphics::MeshBuilder,
        bias: f64,
        theta: f64,
        d_normal: f64,
        d_bend: f64,
        bend: f64) -> GameResult
    {
        mesh.line(
            &[
                self.polar_to_canvas(&vec![bias.abs(), theta]),
                self.polar_to_canvas(&vec![bias.abs() + if bias >= 0. {d_normal} else {-d_normal}, theta]),
            ],
            2.0,
            graphics::Color::BLUE,
        )?;

        mesh.line(
            &[
                self.polar_to_canvas(&vec![bias.abs(), theta]),
                self.cartesian_to_canvas(
                    State::cartesian_rotation(
                        State::to_cartesian(&vec![bias.abs(), theta]),
                        State::to_cartesian(&vec![bias.abs() + d_bend, theta]),
                        if bias >= 0. {bend} else {PI - bend},
                    )
                )
            ],
            2.0,
            graphics::Color::RED,
        )?;

        mesh.line(
            &[
                self.polar_to_canvas(&vec![bias.abs(), theta]),
                self.cartesian_to_canvas(
                    State::cartesian_rotation(
                        State::to_cartesian(&vec![bias.abs(), theta]),
                        State::to_cartesian(&vec![bias.abs() + d_bend, theta]),
                        if bias >= 0. {-bend} else {PI + bend},
                    )
                )
            ],
            2.0,
            graphics::Color::RED,
        )?;

        Ok(())
    }
}

impl ggez::event::EventHandler<GameError> for State {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        if self.iteration < self.n_iters {
            if self.iteration % 100 == 0 {
                println!("Iteration: {}", self.iteration);
                println!("Fitness: {}", self.problem.evaluate(&self.alg));
                println!("Structure: {}", self.alg);
            }
            self.alg.optimization_step(&self.problem);
            self.iteration += 1;
        }

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::WHITE);

        let mesh = &mut graphics::MeshBuilder::new();

        mesh.circle(
            graphics::DrawMode::stroke(2.0),
            mint::Point2{x: 400.0, y: 300.0},
            250.0,
            0.1,
            graphics::Color::BLACK,
        )?;

        let d_hp = 1.;
        let d_normal = 0.1;
        let d_bend = 1.25;

        match &self.alg {
            Algorithm::ContinuousOneplusoneNA(network) => {
                let bends = network.get_angles();
                let biases = network.get_biases();
                for i in 0..bends.len() {
                    let bias = biases[i];
                    let theta = bends[i][0];

                    self.get_decision_line_mesh(mesh, bias, theta, d_normal, d_hp)?;
                }
            }
            Algorithm::DiscreteOneplusoneNA(network) => {
                let bends = network.get_angles();
                let biases = network.get_biases();
                for i in 0..bends.len() {
                    let bias = biases[i];
                    let theta = bends[i][0];

                    self.get_decision_line_mesh(mesh, bias, theta, d_normal, d_hp)?;
                }
            }
            Algorithm::ContinuousBNA(vneuron) => {
                let bias = vneuron.get_bias();
                let angle = vneuron.get_angle(0);
                let bend = vneuron.get_bend();

                self.get_bend_decision_mesh(mesh, bias, angle, d_normal, d_bend, bend)?;
            }
            Algorithm::DiscreteBNA(vneuron) => {
                let bias = vneuron.get_bias();
                let angle = vneuron.get_angle(0);
                let bend = vneuron.get_bend();

                self.get_bend_decision_mesh(mesh, bias, angle, d_normal, d_bend, bend)?;
            }
        }

        // for (point, label) in &self.points {
        //     let point = self.polar_to_canvas(point);
        //     mesh.rectangle(
        //         graphics::DrawMode::fill(),
        //         graphics::Rect::new(point.x - 5., point.y - 5., 10.0, 10.0),
        //         if *label { graphics::Color::GREEN } else { graphics::Color::RED },
        //     )?;
        // }

        let mesh = graphics::Mesh::from_data(ctx, mesh.build());

        canvas.draw(&mesh, graphics::DrawParam::new());
        canvas.finish(ctx)?;

        Ok(())
    }
}
