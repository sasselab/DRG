import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
from drg_tools.plotlib import plot_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare training losses of different models.")
    parser.add_argument('valfiles', type=str, help="Comma-separated list of validation files.")
    parser.add_argument('modnames', type=str, help="Comma-separated list of model names.")
    parser.add_argument('--percentages', action='store_true', help="Normalize epochs to percentages.")
    parser.add_argument('--combine_sets', action='store_true', help="Combine training and validation sets.")
    parser.add_argument('--logx', action='store_true', help="Use logarithmic scale for x-axis.")
    parser.add_argument('--adjust_axis', action='store_true', help="Adjust axis limits for consistency.")
    parser.add_argument('--savefig', type=str, help="Path to save the figure.")

    args = parser.parse_args()

    valfiles = args.valfiles.split(',')
    modnames = args.modnames.split(',')

    plot_losses(
        valfiles=valfiles,
        modnames=modnames,
        percentages=args.percentages,
        combine_sets=args.combine_sets,
        logx=args.logx,
        adjust_axis=args.adjust_axis,
        savefig=args.savefig
    )