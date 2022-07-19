use tch::{TchError, COptimizer, Tensor};
use tch::nn::VarStore;

#[cfg(test)]
mod tests {
    use tch::{TrainableCModule, Tensor, Kind, Device};
    use tch::nn::VarStore;

    #[test]
    fn it_works() {
        let vs = VarStore::new(Device::Cpu);
        let model = TrainableCModule::load("../client/scripted_model.pt", vs.root()).unwrap();
        let gsp = model.method_is::<tch::IValue>("grad_sample_parameters", &[]);
        println!("{:?}", gsp);
        panic!()
    }
}

pub trait Optimizer {
    fn zero_grad(&mut self) -> Result<(), TchError>;
    fn step(&mut self) -> Result<(), TchError>;
}

impl Optimizer for COptimizer {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        COptimizer::zero_grad(self)
    }
    fn step(&mut self) -> Result<(), TchError> {
        COptimizer::step(self)
    }
}

pub struct DPSGD<'a> {
    learning_rate: f64,
    parameters: Vec<IValue>, 
    update: fn(Tensor, Tensor),
}

impl<'a> Optimizer for PrivateSGD<'a> {
    fn zero_grad(&mut self) -> Result<(), TchError> {
        for iv in self.parameters.iter() {
            if let IValue::Tuple(tup) = iv {
                if let IValue::Tensor(param) = tup.0 {
                    if let IValue::Tensor(grad_sample) = tup.1 {
                        (self.update)(param, grad_sample);
                    }
                }
            }
        }
        Ok(())
    }
}
