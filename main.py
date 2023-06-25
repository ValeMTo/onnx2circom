"""

Usage:
  main.py <model.onnx> 
  main.py (-h | --help)

Options:
    -h --help                       Show this screen.

"""

from docopt import docopt
from model import Model

def main():
    args = docopt(__doc__)
    print(args)
    Model(args['<model.onnx>']).create_circuit() 
    


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()