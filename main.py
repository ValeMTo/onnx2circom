from docopt import docopt


""" Transpile a Keras model to a CIRCOM circuit.

Usage:
    main.py <model.onnx> -o <output>

Options:

"""

def main():
    args = docopt(__doc__)    
    #call the handler


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()