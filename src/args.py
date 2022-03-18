import argparse


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-particles', help='number of particles', default=30, type=int)
    parser.add_argument('--num-iterations', help='number of iterations', default=10, type=int)
    parser.add_argument('--data-dir', help='data directory', default='', type=str)
    parser.add_argument('--w', help='inertia weight', default=1, type=float)
    parser.add_argument('--c1', help='cognitive learning factor', default=2, type=float)
    parser.add_argument('--c2', help='social learning factor', default=2, type=float)
    return parser