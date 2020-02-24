from argparse import ArgumentParser
def get_parser():
    parser = ArgumentParser()
    parser.add_argument('-batchsize', default=3000, type=int)
    parser.add_argument('-lrnrate', default=.05, type=float)
    parser.add_argument('-duser', default=30, type=int)
    parser.add_argument('-ditem', default=30, type=int)
    parser.add_argument('-dcateg', default=30, type=int)
    parser.add_argument('-nlayer', default=1, type=int)
    parser.add_argument('-nhidden', default=50, type=int)
    parser.add_argument('-nepoch', default=15, type=int)
    parser.add_argument('-splitratio', default=.2, type=float)
    return parser
