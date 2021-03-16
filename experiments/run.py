"""
====================================================
Birth and merge variational training for topic model

Running into plotting issues when running on Supercloud 
https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
Current fix: make PlotUtil.py use "Agg" backend.

====================================================

"""
import bnpy
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import names_and_parser

FIG_SIZE = (2, 2)
SMALL_FIG_SIZE = (1.5, 1.5)
MED_FIG_SIZE = (1.5, 3)

###############################################################################
# read training settings from command line

options = names_and_parser.parse_args()
print(options)

K = options.K
nLap = options.nLap
use_all_data = options.use_all_data
max_doc = options.max_doc
nBatch = options.nBatch
plot_type=options.plot_type

###############################################################################
#
# Read text dataset from file
dataset_path = os.path.join(bnpy.DATASET_PATH, 'big_wiki')
print("Reading data from %s" %dataset_path)
dataset = bnpy.data.BagOfWordsData.LoadFromFile_ldac(
    os.path.join(dataset_path, 'trainwiki997357.ldac'),
    vocabfile=os.path.join(dataset_path, 'vocab.txt'))

# don't care about minimum/maximum number of tokens 8in a documet
low_cts = 0
high_cts = np.inf

doc_ids = np.flatnonzero(np.logical_and(
    dataset.getDocTypeCountMatrix().sum(axis=1) >= low_cts,
    dataset.getDocTypeCountMatrix().sum(axis=1) < high_cts))

if not (use_all_data):
    doc_ids = doc_ids[:max_doc]
    output_path = '/home/gridsan/tdn/bnpy/experiments/tmp/small_wiki/model=hdp_topic+mult-K=%d/' %K
else:
    output_path = '/home/gridsan/tdn/bnpy/experiments/tmp/big_wiki/model=hdp_topic+mult-K=%d/' %K

dataset = dataset.make_subset(docMask=doc_ids, doTrackFullSize=False)

###############################################################################
# Train LDA topic model
# ---------------------
# 
# Using 10 clusters and a random initialization procedure.

local_step_kwargs = dict(
    # perform at most this many iterations at each document
    nCoordAscentItersLP=20,
    # stop local iters early when max change in doc-topic counts < this thr
    convThrLP=0.01,
    )
merge_kwargs = dict(
    m_startLap=1,
    )
birth_kwargs = dict(
    b_startLap=1,
    b_stopLap=3,
    b_Kfresh=5)

trained_model, info_dict = bnpy.run(
    dataset, 'HDPTopicModel', 'Mult', 'memoVB',
    output_path=output_path,
    nLap=nLap, convergeThr=0.01, nBatch=nBatch,
    K=K, initname='randomlikewang',
    gamma=50.0, alpha=0.5, lam=0.1,
    moves='birth,merge,shuffle',
    **dict(local_step_kwargs.items() + 
        merge_kwargs.items() + 
        birth_kwargs.items()))

print("Will save models to %s" %info_dict['task_output_path'])


###############################################################################
# Helper function to get topic proportions / top words at each stage of training
# after complteted laps through dataset

def show_top_words_over_time(
        task_output_path=None,
        vocabList=None,
        query_laps=[0, 1, 2, 5, None],
        ncols=10):
    '''
    '''
    
    fig = plt.figure()
    nrows = len(query_laps)
    fig_handle, ax_handles_RC = plt.subplots(
        figsize=(SMALL_FIG_SIZE[0] * ncols, SMALL_FIG_SIZE[1] * nrows),
        nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for row_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        # Plot the current model
        cur_ax_list = ax_handles_RC[row_id].flatten().tolist()
        bnpy.viz.PrintTopics.plotCompsFromHModel(
            cur_model,
            vocabList=vocabList,
            fontsize=9,
            Ktop=7,
            ax_list=cur_ax_list)
        cur_ax_list[0].set_ylabel("lap: %d" % lap_val)
        
        cur_proportions = cur_model.allocModel.E_beta()
        for i in range(np.amin([len(cur_ax_list), len(cur_proportions)])):
            ax = cur_ax_list[i]
            ax.set_title('Prop. = %.2f' %cur_proportions[i], fontsize=10)
    plt.subplots_adjust(
        wspace=0.04, hspace=0.1, 
        left=0.01, right=0.99, top=0.99, bottom=0.1)
    plt.tight_layout()
    savefigpath = output_path + 'top_words.png'
    print("Will save figure to %s" %savefigpath)
    plt.savefig(savefigpath)
    return 

def show_topic_proportions_over_time(
        task_output_path=None,
        query_laps=[0, 1, 2, 5, None],
        ncols=10):
    '''
    '''
    
    fig = plt.figure()
    fig_handle, ax_handles_RC = plt.subplots(
        figsize=(MED_FIG_SIZE[0] * ncols, MED_FIG_SIZE[1]),
        nrows=1, ncols=len(query_laps), sharey=True)
    for col_id, lap_val in enumerate(query_laps):
        cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
        # Plot the current model
        cur_ax = ax_handles_RC[col_id]
        cur_proportions = cur_model.allocModel.E_beta()
        cur_ax.set_title('Lap = %d' %lap_val)
        cur_ax.plot(cur_proportions, marker='o')
        cur_ax.set_ylabel('Proportions')
        cur_ax.set_xlabel('Topic index')
    plt.subplots_adjust(
        wspace=0.04, hspace=0.1, 
        left=0.01, right=0.99, top=0.99, bottom=0.1)
    plt.tight_layout()
    savefigpath = output_path + 'topic_proportions.png'
    print("Will save figure to %s" %savefigpath)
    plt.savefig(savefigpath)
    return 

if (plot_type == "top_words"):
    show_top_words_over_time(
        info_dict['task_output_path'], vocabList=dataset.vocabList)
elif (plot_type == "topic_proportions"):
    show_topic_proportions_over_time(
        info_dict['task_output_path'])
