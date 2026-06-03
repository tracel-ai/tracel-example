#![recursion_limit = "256"]

use burn::backend::FlexDevice;
use burn::tensor::Device;
use burn_central_example::training::{self, MnistTrainingConfig};
use tracel::{Env, experiment::ExperimentRun};
use tracel_cloud::Context;

fn main() {
    dotenvy::dotenv().ok();

    // let device = Device::default();
    // let exp_dir = "./experiments";
    // let experiment = ExperimentRun::local(exp_dir).unwrap();
    // let _tracing_guard = experiment.tracing_span();

    // let my_config = MnistTrainingConfig::default();

    // let res = _tracing_guard
    //     .in_scope(|| burn_central_example::training::run_manual(&experiment, config, vec![device]));

    // if let Err(e) = res {
    //     eprintln!("Error during training: {e}");
    // } else {
    //     println!("Training completed successfully.");
    // }

    let my_config = MnistTrainingConfig::default();

    let ctx = Context::cloud(Env::Development).unwrap_or_else(|e| {
        eprintln!("[tracel] error: {e}");
        std::process::exit(1);
    });
    let experiment_module = ctx.experiment();
    let experiment_job = experiment_module.create(|experiment, config: MnistTrainingConfig| {
        let my_devices = vec![Device::autodiff(FlexDevice.into())];
        install_ctrlc(&experiment);

        training::run_manual(experiment, config, my_devices)
    });
    experiment_job.run(my_config).unwrap();
}

fn install_ctrlc(experiment: &ExperimentRun) {
    let cancel_token = experiment.cancel_token();
    ctrlc::set_handler(move || {
        cancel_token.cancel();
        println!("Received Ctrl-C, sending cancellation request to experiment...");
    })
    .expect("Error setting Ctrl-C handler");
}
