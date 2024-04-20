use ggez::*;
use crate::constants::{POLE_BALANCING_MAX_FORCE, ROAD_LENGTH, POLE_BALANCING_STEPS};
use crate::neuroevolution_algorithm::*;

const CART_Y: f64 = 300.;

pub struct State {
    pole_balancing_state: crate::pole_balancing::State,
    algorithm: crate::neuroevolution_algorithm::Algorithm,
    time_step: usize,
}

impl State {
    pub fn new(pole_balancing_state: crate::pole_balancing::State, algorithm: crate::neuroevolution_algorithm::Algorithm) -> Self {
        State {
            pole_balancing_state,
            algorithm,
            time_step: 0,
        }
    }
}

impl ggez::event::EventHandler<GameError> for State {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        if self.time_step >= POLE_BALANCING_STEPS {
            ctx.request_quit();
        }

        let input = self.pole_balancing_state.to_vec();
        let force = 2. * POLE_BALANCING_MAX_FORCE * self.algorithm.evaluate(&input) - POLE_BALANCING_MAX_FORCE;
        self.pole_balancing_state.update(force);

        self.time_step += 1;

        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, graphics::Color::WHITE);
        let mesh = &mut graphics::MeshBuilder::new();

        let cart_x = self.pole_balancing_state.get_cart_position();
        let cart_x = (cart_x + ROAD_LENGTH) / (2. * ROAD_LENGTH) * 800.;

        mesh.rectangle(
            graphics::DrawMode::fill(),
            graphics::Rect::new(cart_x as f32 - 50., CART_Y as f32 - 50., 100., 100.),
            graphics::Color::BLACK,
        )?;

        let pole_angles = self.pole_balancing_state.get_pole_angles();
        let pole_lengths = self.pole_balancing_state.get_pole_lengths();

        for (angle, length) in pole_angles.iter().zip(pole_lengths.iter()) {
            let pole_x = cart_x + 50. * length * angle.sin();
            let pole_y = CART_Y - 50. * length * angle.cos();

            mesh.line(
                &[mint::Point2 { x: cart_x as f32, y: CART_Y as f32 }, mint::Point2 { x: pole_x as f32, y: pole_y as f32 }],
                2.,
                graphics::Color::YELLOW,
            )?;

            mesh.circle(
                graphics::DrawMode::fill(),
                mint::Point2 { x: pole_x as f32, y: pole_y as f32 },
                10.,
                0.1,
                graphics::Color::RED
            )?;
        }

        mesh.line(
            &[mint::Point2 { x: cart_x as f32, y: CART_Y as f32 }, mint::Point2 { x: cart_x as f32, y: CART_Y as f32 + 50. }],
            2.,
            graphics::Color::BLACK,
        )?;

        let mut text = graphics::Text::new(format!("Time step: {}", self.time_step));

        let mesh = graphics::Mesh::from_data(ctx, mesh.build());

        canvas.draw(&mesh, graphics::DrawParam::new());
        canvas.draw(
            text.set_scale(graphics::PxScale::from(30.0)),
            graphics::DrawParam::from([10., 10.]).color(graphics::Color::BLACK),
        );
        canvas.finish(ctx)?;

        Ok(())
    }
}
