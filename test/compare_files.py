"""

Usage:
  main.py <file1> <file2> -t <type>
  main.py (-h | --help)

Options:
    -h --help                       Show this screen.
    -t <type> --type=<type>         Type of file to compare [default: json].

"""

from docopt import docopt

def compare_files_json(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        return f1.read() == f2.read()

def compare_files_circom(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    # Find the line numbers where "include" lines occur in both files
    include_lines1 = [i for i, line in enumerate(lines1) if not line.startswith('include')]
    include_lines2 = [i for i, line in enumerate(lines2) if not line.startswith('include')]

    # Compare the lines after "include" lines
    return include_lines1 == include_lines2
    
def main():
    args = docopt(__doc__)

    if args['--type'] == 'json':
        result = compare_files_json(args['<file1>'], args['<file2>'])
    elif args['--type'] == 'circom':
        result = compare_files_circom(args['<file1>'], args['<file2>'])
    else:
        raise NotImplementedError('File type comparison not implemented')
    
    print(result)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()