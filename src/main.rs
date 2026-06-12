#![recursion_limit = "256"]

use burn::backend::FlexDevice;
use burn::tensor::Device;
use burn_central_example::training::{self, MnistTrainingConfig};
use tracel::{Context, experiment::ExperimentRun};

fn main() -> anyhow::Result<()> {
    Context::cloud()?
        .experiment()
        .create("MNIST_Training", |session: &ExperimentRun, config| {
            training::run(session, config, vec![Device::autodiff(FlexDevice.into())])
        })?
        .run(MnistTrainingConfig::default())
        .map_err(anyhow::Error::from_boxed)?;

    Ok(())
}
