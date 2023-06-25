"""

Usage:
  main.py <model.onnx> [-o <output>] [-v] [--raw]
  main.py (-h | --help)

Options:
    -h --help                       Show this screen.
    -o <output> --output=<output>   Output directory [default: ./].
    -v --verbose                    Verbose output [default: False].
    --raw                           Output raw model outputs instead of the argmax of outputs [default: False].

"""

from docopt import docopt
from model import Model

def main():
    args = docopt(__doc__)
    print('Transpiling model: ' + args['<model.onnx>'])
    print('Output directory: ' + args['--output'])
    print('Verbose: ' + str(args['--verbose']))
    print('Raw: ' + str(args['--raw']))
    print('\n')

    model = Model(args['<model.onnx>'])
    model.create_circuit(args['--output'], args['--verbose'], args['--raw']) 
    


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()