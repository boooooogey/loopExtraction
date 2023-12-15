from importlib import import_module
import argparse
import numpy as np
from util import broadcast_over_diagonal_chrom

parser = argparse.ArgumentParser()
parser.add_argument("--HiC", required=True, help="Path to the HiC file")
parser.add_argument("--bin-size", required=True, type=int,
                    help="the bin size of the provided HiC matrix.")
parser.add_argument("--chromosome-name", required=True, type=str,
                    help="The name of the chromosome.")
parser.add_argument("--model", required=True, help="Model to be used for the fit.")
parser.add_argument("--model-output", required=True, help="output file")
parser.add_argument("--window-size", default=400, type=int, help="Window size for each patch.")
parser.add_argument("--overlap", default=200, type=int,
                    help="The overlap size between consecutive patches.")
parser.add_argument("--transformation", default="identity", type=str,
                    help="Transformation for each patch.")
parser.add_argument("--diag-start", default=1, type=int,
                    help="First diagonal index to be included in training.")
parser.add_argument("--diag-end", default=None, type=int,
                    help="Last diagonal index to be included in training.")

args = parser.parse_args()

model = getattr(import_module(f"model.{args.model}"), "model")
if args.transformation == "identity":
    def transformation(x):
        """No transformation"""
        return x
else:
    transformation = getattr(import_module("transformations"), args.transformation)

if args.diag_end is None:
    args.diag_end = args.window_size

data = np.load(args.HiC)
parameters = broadcast_over_diagonal_chrom(model,
                                           transformation,
                                           data,
                                           args.window_size,
                                           args.overlap,
                                           args.model_output,
                                           args.chromosome_name,
                                           args.bin_size,
                                           args.diag_start,
                                           args.diag_end)
