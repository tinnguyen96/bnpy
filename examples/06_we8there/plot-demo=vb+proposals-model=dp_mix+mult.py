"""
=============================================
VB coordinate descent for Mixture of Multinomials
=============================================


"""
import bnpy
import numpy as np
import os

from matplotlib import pylab
import seaborn as sns

FIG_SIZE = (3, 3)
SMALL_FIG_SIZE = (1,1)
pylab.rcParams['figure.figsize'] = FIG_SIZE

###############################################################################
# Read text dataset from file

dataset_path = os.path.join(bnpy.DATASET_PATH, 'we8there', 'raw')
dataset = bnpy.data.BagOfWordsData.read_npz(
    os.path.join(dataset_path, 'dataset.npz'),
    vocabfile=os.path.join(dataset_path, 'x_csc_colnames.txt'))

# Filter out documents with less than 20 words
doc_ids = np.flatnonzero(
    dataset.getDocTypeCountMatrix().sum(axis=1) >= 20)
dataset = dataset.make_subset(docMask=doc_ids, doTrackFullSize=False)
###############################################################################
#
# Make a simple plot of the raw data
bnpy.viz.PrintTopics.plotCompsFromWordCounts(
    dataset.getDocTypeCountMatrix()[:10],
    dataset.vocabList,
    prefix='doc',
    Ktop=10)

###############################################################################
#
# Train with VB algorithm
# -----------------------
# 
# Take the best of 1 initializations

merge_kwargs = dict(
    m_startLap=10,
    m_pair_ranking_procedure='elbo',
    m_pair_ranking_direction='descending',
    m_pair_ranking_do_exclude_by_thr=1,
    m_pair_ranking_exclusion_thr=-0.0005,
    )
'''
trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'memoVB',
    output_path='/tmp/we8there/helloworld-model=dp_mix+mult-K=30/',
    nLap=1000, convergeThr=0.0001, nTask=1, nBatch=1,
    K=30, initname='bregmankmeans+lam1+iter1',
    gamma0=50.0, lam=0.1,
    moves='birth,merge,shuffle',
    b_startLap=2, b_Kfresh=5, b_stopLap=10,
    **merge_kwargs)
'''
trained_model, info_dict = bnpy.run(
    dataset, 'DPMixtureModel', 'Mult', 'memoVB',
    output_path='/tmp/we8there/helloworld-model=dp_mix+mult-K=30/',
    nLap=1000, convergeThr=0.0001, nTask=1, nBatch=1,
    K=30, initname='bregmankmeans+lam1+iter1',
    gamma0=50.0, lam=0.1,
    moves='birth,merge,shuffle',
    b_startLap=2, b_Kfresh=5, b_stopLap=10,
    **merge_kwargs)
bnpy.viz.PrintTopics.plotCompsFromHModel(
    trained_model,
    vocabList=dataset.vocabList,
    Ktop=10)

pylab.show()