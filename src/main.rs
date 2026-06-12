#![recursion_limit = "256"]

use burn::tensor::Device;
use burn_central::experiment::{
    ExperimentRun,
    integration::tracing::{ExperimentTracingExt, try_init_tracing_subscriber},
};
use burn_central_example::training::MnistTrainingConfig;

fn main() {
    try_init_tracing_subscriber();
    let device = Device::default();

    let exp_dir = "./experiments";
    let experiment = ExperimentRun::local(exp_dir).unwrap();
    let _tracing_guard = experiment.tracing_span();

    install_ctrlc(&experiment);

    let config = MnistTrainingConfig::default();

    let res = _tracing_guard
        .in_scope(|| burn_central_example::training::run_manual(&experiment, config, vec![device]));

    if let Err(e) = res {
        eprintln!("Error during training: {e}");
    } else {
        println!("Training completed successfully.");
    }
}

fn install_ctrlc(experiment: &ExperimentRun) {
    let cancel_token = experiment.cancel_token();
    ctrlc::set_handler(move || {
        cancel_token.cancel();
        println!("Received Ctrl-C, sending cancellation request to experiment...");
    })
    .expect("Error setting Ctrl-C handler");
}
