# onnx2circom

`onnx2circom` is a tool designed to convert ONNX models into Circom circuits. The tool relies on an external repository, `keras2circom` and `circomlib-ml`, for reach the circom circuit and component.

## Installation
First, clone the `onnx2circom` repository:

```
git clone https://github.com/ValeMTo/onnx2circom
```

Then, install the dependencies. You can use pip:

```
pip install -r requirements.txt
```

If you use conda, you can also create a new environment with the following command:

```
conda env create -f environment.yml
```

`onnx2circom` also requires `keras2circom` to function properly.\
Here's how you can install keras2circom:
```
git clone https://github.com/socathie/keras2circom
```
You may need to install additional dependencies for `keras2circom`. Refer to its own README file for specific installation instructions.

## Usage
After installing `onnx2circom` and `keras2circom`, you can convert your ONNX models into Circom circuits using the following command:

```bash
python main_converter.py <model_path> [-o <output_dir>] [-v] [--raw]
```

For example, to transpile the model in `models/model_dense.onnx` into a circom circuit, you can run:

```bash
python main_converter.py models/model_dense.onnx
```

The output will be in the `output` directory.

If you want to transpile the model into a circom circuit with `--verbose` output, i.e. command line print of inputs and output of each layer, you can run:

```bash
python main_converter.py models/model_dense.onnx -v
```

Moreover, if you want to transpile the model into a circom circuit with `--raw` output, i.e. no ArgMax at the end, you can run:

```bash
python main_converter.py models/model_dense.onnx --raw
```
