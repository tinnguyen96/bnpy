"""
Define functions that read in command line arguments and functions
that return names of directories.
"""

import argparse

## ------------------------------------------------------------------
## Argument parsers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(nLap=5, K=100, use_all_data=False, nBatch=1000, max_doc=None)
    
    parser.add_argument("--nLap", type=int, dest="nLap",help='how many completed passes through data')
    
    parser.add_argument("--K", type=int, dest="K",
                    help="initial number of topics")
    
    parser.add_argument("--use_all_data", action='store_true', dest="use_all_data", help='whether to use all data')
    
    parser.add_argument("--nBatch", type=int, dest="nBatch",help='how many mini-batches to use to break up initial data')
    
    parser.add_argument("--max_doc", type=int, dest="max_doc", help='if use only subset of data, how many documents')
    
    parser.add_argument("--plot_type", type=str, dest="plot_type", help='what type of plot', choices=['top_words', 'topic_proportions'])
    

    options = parser.parse_args()
    return options