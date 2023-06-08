# Onnx to Circom
## Installation

## Usage

## Documentation
### Model Generator
This file likely contains code and documentation for generating machine learning models. It may include sections for data preparation, feature engineering, model selection, hyperparameter tuning, and evaluation.
The dataset is random generated.

### Main
It contains the central logic or orchestrates the execution of various modules and functions. It handles command-line arguments, coordinate the flow of execution, and interact with other parts of the application.

### Model
The circuit is initialised here. 
Then, there is a call to transpile each node to a circom component

### Handler
A request handler that responds to specific requests and performs certain actions based on the input data. Its main rule is of on the traslation of the nodes

### Testing.ipynb
It is based of the project. 
It helps to define functions and understand how the onnx model is structured. It is used to implement and test functions at the beginning

### Circuit.circom & circuit.json
Output examples of my translation