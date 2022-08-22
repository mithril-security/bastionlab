pub mod optim;
pub mod nn;
pub mod serialization;
pub mod procedures;
pub mod data;

#[cfg(test)]
mod tests {
    use tch::{TrainableCModule, Device, Tensor, TchError, Kind};
    use tch::nn::VarStore;

    use crate::nn::{Parameters, LossType};
    use crate::optim::{SGD, Optimizer};

    fn l2_loss(output: &Tensor, target: &Tensor) -> Result<Tensor, TchError> {
        output.f_sub(&target)?.f_norm_scalaropt_dim(2, &[1], false)?.f_mean(Kind::Float)
    }

    #[test]
    fn basic_sgd() {
        let vs = VarStore::new(Device::Cpu);
        let model = TrainableCModule::load("../client/tests/lreg.pt", vs.root()).unwrap();
        let parameters = Parameters::standard(&vs);
        let mut optimizer = SGD::new(parameters, 0.1);

        let data = vec![
            Tensor::of_slice::<f32>(&[0.]),
            Tensor::of_slice::<f32>(&[1.]),
        ];
        let target = vec![
            Tensor::of_slice::<f32>(&[0.]),
            Tensor::of_slice::<f32>(&[2.]),
        ];
        
        for _ in 0..100 {
            for (x, t) in data.iter().zip(target.iter()) {
                let y = model.forward_ts(&[x]).unwrap();
                let loss = (y - t).abs();
                optimizer.zero_grad().unwrap();
                loss.backward();
                optimizer.step().unwrap();
            }
        }
        let w = &optimizer.parameters.into_inner()[0];
        assert!((w - Tensor::of_slice::<f32>(&[2.])).abs().double_value(&[]) < 0.1);
    }

    #[test]
    fn private_sgd() {
        let vs = VarStore::new(Device::Cpu);
        let model = TrainableCModule::load("../client/tests/private_lreg.pt", vs.root()).unwrap();
        let parameters = Parameters::private(&vs, 1.0, 0.1, LossType::Mean(2));
        let mut optimizer = SGD::new(parameters, 0.1);

        let data = vec![
            Tensor::of_slice::<f32>(&[0.0, 1.0]).f_view([2, 1]).unwrap(),
            Tensor::of_slice::<f32>(&[0.5, 0.2]).f_view([2, 1]).unwrap(),
        ];
        let target = vec![
            Tensor::of_slice::<f32>(&[0.0, 2.0]).f_view([2, 1]).unwrap(),
            Tensor::of_slice::<f32>(&[1.0, 0.4]).f_view([2, 1]).unwrap(),
        ];
        
        for _ in 0..100 {
            for (x, t) in data.iter().zip(target.iter()) {
                let y = model.forward_ts(&[x]).unwrap();
                let loss = l2_loss(&y, &t).unwrap();
                // y.print();
                optimizer.zero_grad().unwrap();
                loss.backward();
                optimizer.step().unwrap();
            }
        }
        let w = &optimizer.parameters.into_inner()[0];
        assert!(l2_loss(w, &Tensor::of_slice::<f32>(&[2.])).unwrap().double_value(&[]) < 0.1);
    }

}
