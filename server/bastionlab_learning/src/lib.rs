pub mod data;
pub mod nn;
pub mod optim;
pub mod procedures;
pub mod serialization;

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};
    use tch::nn::VarStore;
    use tch::{Device, Kind, TchError, Tensor, TrainableCModule};

    use crate::data::privacy_guard::{
        BatchDependence, PrivacyBudget, PrivacyContext, PrivacyGuard,
    };
    use crate::nn::{LossType, Module};
    use crate::optim::{Optimizer, SGD};

    fn l2_loss(output: &Tensor, target: &Tensor) -> Result<Tensor, TchError> {
        output
            .f_sub(&target)?
            .f_norm_scalaropt_dim(2, &[1], false)?
            .f_mean(Kind::Float)
    }

    #[test]
    fn test_data_leak() {
        let vs = VarStore::new(Device::Cpu);
        let _model = TrainableCModule::load("lin.pt", vs.root()).unwrap();
        let mut a = vs.trainable_variables();
        let b = vs.trainable_variables();
        a[0].print();
        let _ = a[0].set_requires_grad(false);
        let _ = a[0].f_add_scalar_(1.0).unwrap();
        a[0].print();
        b[0].print();
        assert_eq!(a, b);
    }

    // This fails to compile
    // #[test]
    // fn containment_is_enforced_by_borrow_checker() {
    //     let mut module = Module::load_from_file("lin.pt", Device::Cpu).unwrap();
    //     let (_, a) = module.parameters();
    //     let (_, b) = module.private_parameters(1.0, 1.0, LossType::Sum);
    //     println!("{:?} {:?}", a, b);
    // }

    #[test]
    fn test_containment_leak() {
        let mut module = Module::load_from_file("lin.pt", Device::Cpu).unwrap();
        let mut a = module.parameters().1.into_inner().unwrap();
        let (_, b) = module.private_parameters(1.0, 1.0, LossType::Sum);
        let _ = a[0].f_add_scalar_(1.0).unwrap();
        assert_ne!(a[0], b.into_inner().unwrap()[0]);
    }

    #[test]
    fn basic_sgd() {
        let mut module = Module::load_from_file("lreg_base.pt", Device::Cpu).unwrap();
        let (forward, parameters) = module.parameters();
        // let mut chkpt = CheckPoint::new();
        let mut optimizer = SGD::new(parameters, 0.1);

        let data = vec![
            Tensor::of_slice::<f32>(&[0.]),
            Tensor::of_slice::<f32>(&[1.]),
        ];
        let target = vec![
            Tensor::of_slice::<f32>(&[0.]),
            Tensor::of_slice::<f32>(&[2.]),
        ];

        let context = Arc::new(RwLock::new(PrivacyContext::new(
            PrivacyBudget::NotPrivate,
            4,
        )));

        for _ in 0..100 {
            for (x, t) in data.iter().zip(target.iter()) {
                let x = PrivacyGuard::new(x.copy(), BatchDependence::Dependent, context.clone());
                let t = PrivacyGuard::new(t.copy(), BatchDependence::Dependent, context.clone());
                let y = forward.forward(vec![x]).unwrap();
                let loss = y
                    .f_mse_loss(&t, (0.0, 10.0), tch::Reduction::Mean)
                    .unwrap()
                    .0;
                optimizer.zero_grad().unwrap();
                loss.backward();
                optimizer.step().unwrap();
            }
        }
        let w = &optimizer.parameters.into_inner().unwrap()[0];
        assert!((w - Tensor::of_slice::<f32>(&[2.])).abs().double_value(&[]) < 0.1);
    }

    #[test]
    fn private_sgd() {
        let mut module = Module::load_from_file("lreg.pt", Device::Cpu).unwrap();
        let (forward, parameters) = module.private_parameters(30.0, 1.0, LossType::Mean(2));
        // let mut chkpt = CheckPoint::new();

        let mut optimizer = SGD::new(parameters, 0.1);

        let data = vec![
            Tensor::of_slice::<f32>(&[0.0, 1.0]).f_view([2, 1]).unwrap(),
            Tensor::of_slice::<f32>(&[0.5, 0.2]).f_view([2, 1]).unwrap(),
        ];
        let target = vec![
            Tensor::of_slice::<f32>(&[0.0, 2.0]).f_view([2, 1]).unwrap(),
            Tensor::of_slice::<f32>(&[1.0, 0.4]).f_view([2, 1]).unwrap(),
        ];

        let context = Arc::new(RwLock::new(PrivacyContext::new(
            PrivacyBudget::Private(300.1),
            4,
        )));

        for _ in 0..200 {
            for (x, t) in data.iter().zip(target.iter()) {
                let x = PrivacyGuard::new(x.copy(), BatchDependence::Dependent, context.clone());
                let t = PrivacyGuard::new(t.copy(), BatchDependence::Dependent, context.clone());

                let y = forward.forward(vec![x]).unwrap();
                let loss = y
                    .f_mse_loss(&t, (0.0, 10.0), tch::Reduction::Mean)
                    .unwrap()
                    .0;
                optimizer.zero_grad().unwrap();
                loss.backward();
                optimizer.step().unwrap();
            }
        }
        let w = &optimizer.parameters.into_inner().unwrap()[0];
        w.print();
        assert!(
            l2_loss(w, &Tensor::of_slice::<f32>(&[2.]))
                .unwrap()
                .double_value(&[])
                < 0.1
        );
    }
}
