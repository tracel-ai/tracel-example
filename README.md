# Tracel SDK Example (MNIST)

This repository shows how to adapt the standard Burn MNIST example into a Tracel SDK project. It is a single executable that trains the model and reports the experiment (metrics, checkpoints, and artifacts) to Tracel.

## What This Example Covers

- defining a custom model artifact with `BundleEncode` and `BundleDecode`
- exposing training configuration through `MnistTrainingConfig`
- creating and running an experiment with `Context::cloud()` and `ExperimentRun`
- wiring metrics, checkpoints, and interruption handling through `ExperimentRun`

## Project Layout

- [`src/main.rs`](src/main.rs): entry point that opens a cloud `Context`, creates the `MNIST_Training` experiment, and runs it
- [`src/training.rs`](src/training.rs): training loop, evaluation, and artifact upload
- [`src/model.rs`](src/model.rs): model definition and artifact bundle serialization
- [`src/data.rs`](src/data.rs): MNIST batching and data augmentation

## Run

`Context::cloud()` needs Tracel credentials. Either set `TRACEL_API_KEY` in your environment, or authenticate once with:

```bash
burn login
```

The project's namespace and name come from [`tracel.toml`](tracel.toml). By default the example uses the `FlexDevice` backend; optional backends can be enabled with Cargo features.

```bash
cargo run
cargo run --features wgpu
cargo run --features cuda
```

The run is reported as the `MNIST_Training` experiment.

## More Details

- Tracel SDK docs: [docs.rs/tracel](https://docs.rs/tracel/latest/tracel/)
