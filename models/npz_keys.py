import sys

import numpy


if __name__ == '__main__':
    with numpy.load(sys.argv[1]) as f:
        print(f.keys())
