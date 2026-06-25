#![recursion_limit = "256"]

use burn::{backend::WgpuDevice, tensor::Device};
use burn_central_example::training::{self, MnistTrainingConfig};
use tracel::{Connection, Context, experiment::ExperimentRun};

fn main() -> anyhow::Result<()> {
    Context::new(Connection::Cloud)?
        .experiment()
        .create("MNIST_Training", |session: &ExperimentRun, config| {
            training::run(
                session,
                config,
                vec![Device::autodiff(WgpuDevice::default().into())],
            )
        })
        .run(MnistTrainingConfig::default())
        .map_err(anyhow::Error::from_boxed)?;

    Ok(())
}
