use ggez::*;
use crate::neuroevolution_algorithm::*;

pub struct State {
    alg: Algorithm,
    problem: fn(&Algorithm) -> f64,
    n_iters: u32,
    iteration: u32,
}

impl State {
    pub fn new(alg: Algorithm, problem: fn(&Algorithm) -> f64, n_iters: u32) -> Self {
        State {
            alg,
            problem,
            n_iters,
            iteration: 0,
        }
    }

    fn polar_to_canvas(&self, v: &Vec<f64>) -> mint::Point2<f32> {
        let x = v[0] * v[1].cos();
        let y = v[0] * v[1].sin();
        mint::Point2{x: 400. + 250. * x as f32,y: 300. - 250. * y as f32}
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
}

impl ggez::event::EventHandler<GameError> for State {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        if self.iteration < self.n_iters {
            if self.iteration % 100 == 0 {
                println!("Iteration: {}", self.iteration);
            }
            self.alg.optimization_step(self.problem);
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

        match &self.alg {
            Algorithm::ContinuousOneplusoneNA(network) => {
                let bends = network.get_angles();
                let biases = network.get_biases();
                for i in 0..bends.len() {
                    let bias = biases[i];
                    let theta = bends[i][0];
                    let d_hp = 1.;
                    let d_normal = 0.1;

                    self.get_decision_line_mesh(mesh, bias, theta, d_normal, d_hp)?;
                }
            }
            Algorithm::DiscreteOneplusoneNA(network) => {
                let bends = network.get_angles();
                let biases = network.get_biases();
                for i in 0..bends.len() {
                    let bias = biases[i];
                    let theta = bends[i][0];
                    let d_hp = 1.;
                    let d_normal = 0.1;

                    self.get_decision_line_mesh(mesh, bias, theta, d_normal, d_hp)?;
                }
            }
            _ => {}
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
