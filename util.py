import bnpy
import numpy as np
import os
import pickle

from scipy import stats
from matplotlib import pyplot as plt

from bnpy.allocmodel.hmm.HMMUtil import runViterbiAlg
from bnpy.data import SpeakerDiar, MoCap124
from bnpy.util.StateSeqUtil import alignEstimatedStateSeqToTruth
from bnpy.util.StateSeqUtil import calcHammingDistance

plt.rcParams['figure.figsize'] = (18, 8)
plt.rcParams['font.size'] = 20

### Plotting utils ###

def create_title(plot_title, dataset_title, obs_model, learn_alg='VB'):
    return "%s, %s, %s - %s" % (dataset_title, obs_model, learn_alg, plot_title)

def plot_ax(axes, max_iters, y_vals, L, sparse_opt, summary):
    L_vals, zeropass_y, onepass_y, twopass_y = summary
    y_final = y_vals[-1]

    # Append y_vals so all lines in one plot have the same lengths
    tail = np.full(max_iters - len(y_vals), y_vals[-1])
    y_vals = np.concatenate((y_vals, tail))

    # Determine color and label
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    color = colors[L%5] if L > 0 else 'black'
    label = 'L = %d' % L if L > 0 else 'Dense'

    # Determine axes
    ax1, ax2, ax3 = axes
    plot_axes = []

    if L == 0 or L == 1: # Dense or Viterbi (L = 1) case
        # Include in all (0-, 1-, and 2-pass) subplots
        plot_axes.extend([ax1, ax2, ax3])

        # Summarize info for y vs L plot
        L_vals.append(L)
        zeropass_y.append(y_final)
        onepass_y.append(y_final)
        twopass_y.append(y_final)

    elif sparse_opt == 'zeropass':
        # Zero-pass subplot
        plot_axes.append(ax3)

        # Summarize info for y vs L plot
        zeropass_y.append(y_final)

    elif sparse_opt == 'onepass':
        # One-pass subplot
        plot_axes.append(ax1)

        # Summarize info for y vs L plot
        onepass_y.append(y_final)

    else:
        # Two-pass subplot
        plot_axes.append(ax2)

        # Summarize info for y vs L plot
        L_vals.append(L)
        twopass_y.append(y_final)

    # Plot
    for ax in plot_axes:
        ax.plot(np.arange(1, max_iters + 1), y_vals,
                label=label, color=color)

def plot_setup(experiment_out, plot_title, dataset_title, ylab):
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    axes = [ax1, ax2, ax3]
    max_iters = np.max([len(d['lap_history']) for _, d in experiment_out])
    obs_model = d['ReqArgs']['obsModelName']

    for i, ax in enumerate(axes):
        ax.set_xlabel('iteration')
        ax.set_ylabel(ylab)
        if i == 0:
            ax_title = 'One-pass'
        elif i == 1:
            ax_title = 'Two-pass'
        else:
            ax_title = 'L2'
        ax.set_title(ax_title)

    plt.suptitle(create_title(plot_title, dataset_title, obs_model))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return axes, max_iters, obs_model

def plot_show(axes, ymin=None, ymax=None):
    ax1, ax2, ax3 = axes
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    ax3.set_ylim(ymin, ymax)
    plt.legend()
    plt.show()

def unpack_info(info_dict):
    alg_dict = info_dict['KwArgs']['VB']
    L = alg_dict['nnzPerRowLP']
    sparse_opt = alg_dict['sparseOptLP']
    return L, sparse_opt

def compute_hamming(model_dict):
    dataset = model_dict['Data']
    N = dataset.X.shape[0]
    doc_range = dataset.doc_range
    ztrue = dataset.TrueParams['Z']
    path = model_dict['task_output_path']
    laps = np.concatenate(([0], model_dict['lap_history']))
    ham_dist_history = np.empty_like(model_dict['lap_history'], dtype=float)
    
    # Iterate over laps
    for lap in laps:
        # Load model
        model, _ = bnpy.load_model_at_lap(path, lap)
    
        # Load model parameters
        init = model.allocModel.get_init_prob_vector()
        trans = model.allocModel.get_trans_prob_matrix()
        evidence = model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)
        
        # Run Viterbi
        zhat = np.empty(N)
        for i, (start, end) in enumerate(zip(doc_range[:-1], doc_range[1:])):
            zhat[start:end] = runViterbiAlg(evidence[start:end],
                                            np.log(init), np.log(trans))
        zhat_aligned = alignEstimatedStateSeqToTruth(zhat, ztrue)

        # Compute Hamming distance
        ham_dist_history[lap - 1] = calcHammingDistance(ztrue, zhat_aligned)
        
    return ham_dist_history

def plot_y_vs_iter(experiment_out, plot_title, dataset_title,
                   get_y_vals, ylab, ymin=None, ymax=None):
    # Summarize info for y vs L plot
    L_vals = []
    zeropass_y = []
    onepass_y = []
    twopass_y = []
    summary = [L_vals, zeropass_y, onepass_y, twopass_y]

    # Plot setup
    axes, max_iters, obs_model = plot_setup(experiment_out,
                                            plot_title, dataset_title, ylab)

    # Plot each L
    for _, info_dict in experiment_out:
        L, sparse_opt = unpack_info(info_dict)
        y_vals = get_y_vals(info_dict)
        plot_ax(axes, max_iters, y_vals, L, sparse_opt, summary)
    plot_show(axes, ymin, ymax)

    # Return summary
    summary.append(obs_model)
    return summary

def plot_y_vs_L(summary, plot_title, dataset_title, ylab):
    L_vals, zeropass_y, onepass_y, twopass_y, obs_model = summary

    # Plot three sparse y values vs L
    plt.plot(L_vals[:-1], onepass_y[:-1], label='One-pass', marker='o')
    plt.plot(L_vals[:-1], twopass_y[:-1], label='Two-pass', marker='o')
    plt.plot(L_vals[:-1], zeropass_y[:-1], label='L2', marker='o')
    
    # Last element of *_y should all store the same dense y value
    assert(zeropass_y[-1] == onepass_y[-1] == twopass_y[-1])

    # Plot dense y value as a horizontal line
    plt.hlines(onepass_y[-1], L_vals[0], L_vals[-2], label='Dense')
    
    # Title, labels, legend, etc.
    plt.xlabel('L')
    plt.ylabel(ylab)
    plt.legend()
    plt.title(create_title(plot_title, dataset_title, obs_model))
    plt.show()

# Plot Hamming distance
def plot_hamming(experiment_out, dataset_title, ymin=None, ymax=None):
    ylab = 'Hamming distance'

    # Hamming distance vs iteration
    summary = plot_y_vs_iter(experiment_out, 'Hamming distance vs iteration',
                             dataset_title, compute_hamming, ylab,
                             ymin=ymin, ymax=ymax)

    # Plot Hamming distance vs L
    plot_y_vs_L(summary, 'Hamming distance vs L', dataset_title, ylab)

# Plot loss
def plot_loss(experiment_out, dataset_title, ymin=None, ymax=None):
    ylab = 'loss'

    # Loss vs iteration
    get_y_vals = lambda d: d['loss_history']
    summary = plot_y_vs_iter(experiment_out, 'Loss vs iteration', dataset_title,
                             get_y_vals, ylab, ymin=ymin, ymax=ymax)

    # Loss vs L
    plot_y_vs_L(summary, 'Loss vs L', dataset_title, ylab)

# Plot clusters (ignore time steps)
def plot_clusters(experiment_out, dataset_title):
    n_plots = len(experiment_out)
    n_rows = np.ceil(n_plots / 2.0)
    
    plt.figure(figsize=(18, n_rows * 4))
    for i, (model, info_dict) in enumerate(experiment_out):
        # Make dense model the first subplot
        if i == len(experiment_out) - 1:
            i = -1

        dataset = info_dict['Data']
        plt.subplot(n_rows, 2, i + 2)
        bnpy.viz.PlotComps.plotCompsFromHModel(model, dataset=dataset)

        # Subplot title
        L, sparse_opt = unpack_info(info_dict)
        if L == 0:
            label = 'Dense'
        elif L == 1:
            label = 'Viterbi L = 1'
        elif sparse_opt == 'zeropass':
            label = 'Zero-pass L = %d' % L
        elif sparse_opt == 'onepass':
            label = 'One-pass L = %d' % L
        else:
            label = 'Two-pass L = %d' % L

        plt.title(label)

    obs_model = info_dict['ReqArgs']['obsModelName']
    plt.suptitle(create_title('Learned components', dataset_title, obs_model))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

### Experiment utils ###

def run_experiment(dataset, alloc_model, obs_model, alg, K, out_path,
                   min_L=1, max_L=5, n_task=5, tol=1e-4, max_laps=500, save=True,
                   **kwargs):
    # Find the best seed for the dense model
    print 'Training dense model'
    dense_path = '/'.join((out_path, 'dense'))
    dense_model, dense_dict = bnpy.run(dataset, alloc_model, obs_model, alg,
                                       K=K, output_path=dense_path,
                                       convergeThr=tol, nLap=max_laps,
                                       printEvery=25, nTask=n_task,
                                       **kwargs)
    taskid = dense_dict['taskid']

    # Save trials in a sorted order from lowest to highest (dense) L
    experiment_out = []
    for L in xrange(min_L, max_L + 1):
        print 'Training one-pass sparse model L =', L
        onepass_path = '/'.join((out_path, 'onepass-L=%d' % L))
        onepass_model, onepass_dict = bnpy.run(dataset, alloc_model, obs_model, alg,
                                               K=K, output_path=onepass_path,
                                               convergeThr=tol, nLap=max_laps,
                                               printEvery=25, taskid=taskid,
                                               nnzPerRowLP=L, sparseOptLP='onepass',
                                               **kwargs)
        experiment_out.append((onepass_model, onepass_dict))

        if L > 1:
            print 'Training two-pass sparse model L =', L
            twopass_path = '/'.join((out_path, 'twopass-L=%d' % L))
            twopass_model, twopass_dict = bnpy.run(dataset, alloc_model, obs_model, alg,
                                                   K=K, output_path=twopass_path,
                                                   convergeThr=tol, nLap=max_laps,
                                                   printEvery=25, taskid=taskid,
                                                   nnzPerRowLP=L, sparseOptLP='twopass',
                                                   **kwargs)
            experiment_out.append((twopass_model, twopass_dict))

            print 'Training O(L^2) sparse model L =', L
            zeropass_path = '/'.join((out_path, 'zeropass-L=%d' % L))
            zeropass_model, zeropass_dict = bnpy.run(dataset, alloc_model, obs_model, alg,
                                                     K=K, output_path=zeropass_path,
                                                     convergeThr=tol, nLap=max_laps,
                                                     printEvery=25, taskid=taskid,
                                                     nnzPerRowLP=L, sparseOptLP='zeropass',
                                                     **kwargs)
            experiment_out.append((zeropass_model, zeropass_dict))

    experiment_out.append((dense_model, dense_dict))

    # Save experiment_out
    if save:
        pickle.dump(experiment_out, open('%s/summary.p' % out_path, 'wb'))

    return experiment_out

def run_experiment_sparse(dataset, alloc_model, obs_model, alg, K, out_path,
                          min_L=1, max_L=5, taskid=1, tol=1e-4, max_laps=500,
                          save=True, **kwargs):
    # Save trials in a sorted order from lowest to highest (dense) L
    experiment_out = []
    for L in xrange(min_L, max_L + 1):
        print 'Training one-pass sparse model L =', L
        onepass_path = '/'.join((out_path, 'onepass-L=%d' % L))
        onepass_model, onepass_dict = bnpy.run(dataset, alloc_model, obs_model, alg,
                                               K=K, output_path=onepass_path,
                                               convergeThr=tol, nLap=max_laps,
                                               printEvery=25, taskid=taskid,
                                               nnzPerRowLP=L, sparseOptLP='onepass',
                                               **kwargs)
        experiment_out.append((onepass_model, onepass_dict))

        if L > 1:
            print 'Training two-pass sparse model L =', L
            twopass_path = '/'.join((out_path, 'twopass-L=%d' % L))
            twopass_model, twopass_dict = bnpy.run(dataset, alloc_model, obs_model, alg,
                                                   K=K, output_path=twopass_path,
                                                   convergeThr=tol, nLap=max_laps,
                                                   printEvery=25, taskid=taskid,
                                                   nnzPerRowLP=L, sparseOptLP='twopass',
                                                   **kwargs)
            experiment_out.append((twopass_model, twopass_dict))

            print 'Training O(L^2) sparse model L =', L
            zeropass_path = '/'.join((out_path, 'zeropass-L=%d' % L))
            zeropass_model, zeropass_dict = bnpy.run(dataset, alloc_model, obs_model, alg,
                                                     K=K, output_path=zeropass_path,
                                                     convergeThr=tol, nLap=max_laps,
                                                     printEvery=25, taskid=taskid,
                                                     nnzPerRowLP=L, sparseOptLP='zeropass',
                                                     **kwargs)
            experiment_out.append((zeropass_model, zeropass_dict))

    # Save experiment_out
    if save:
        pickle.dump(experiment_out, open('%s/summary.p' % out_path, 'wb'))

    return experiment_out

def load_experiment(out_path):
    return pickle.load(open('%s/summary.p' % out_path, 'rb'))

### Data utils ###

# Generate a toy dataset with binvariate Gaussian observations
# of shape n_rows x n_cols grid
def generate_grid_data(N, T, n_rows=3, n_cols=3, step=10, sd=3):
    D = 2
    K = n_rows * n_cols
    
    # Hyperparameter for initial and transition probabilities
    alpha = np.full(K, 1.0)    
    
    # Initial state probabilities
    pi0 = stats.dirichlet.rvs(alpha).reshape(K)

    # Transition probabilities
    pi = stats.dirichlet.rvs(alpha, size=K)
    
    # Observation means
    mu = stats.norm.rvs(np.vstack((np.repeat(np.arange(0, n_cols * step, step), n_rows),
                                   np.tile(np.arange(0, n_rows * step, step), n_cols))).T)
    
    # t = 1
    states = np.zeros((N, T), dtype=int)
    obs = np.zeros((N, T, D))
    states[:, 0] = np.random.choice(K, size=N, replace=True, p=pi0)
    obs[:, 0, :] = stats.norm.rvs(loc=mu[states[:, 0]], scale=sd)
    
    # t = 2, ..., T
    for n in xrange(N):
        for t in xrange(1, T):
            states[n, t] = np.random.choice(K, p=pi[states[n, t-1]])
            obs[n, t, :] = stats.norm.rvs(loc=mu[states[n, t]], scale=sd)
    
    X = obs.reshape((N*T, D))
    doc_range = np.arange(0, N*T+1, T)
    Z_true = states.flatten()
    dataset = bnpy.data.GroupXData(X, doc_range, TrueZ=Z_true)

    return dataset

def load_speaker_data(meeting_id):
    return SpeakerDiar.get_data(meeting_id)

def load_mocap_data():
    dataset_path = os.path.join(bnpy.DATASET_PATH, 'mocap6', 'dataset.mat')
    return bnpy.data.GroupXData.read_mat(dataset_path)

def load_full_mocap_data():
    return MoCap124.get_data()
