import os

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def visualize_statistics(directory: str, stats: "TrainingStatistics"):
    os.makedirs(directory, exist_ok=True)
    
    sns.set_theme(style="whitegrid")

    axes = None
    if len(stats.val_history_steps) != 0:
        axes = sns.lineplot(x=stats.val_history_steps, y=stats.val_loss_history, label=f'Validation Loss (l={stats.val_loss_history[-1]:.3f}/b={min(stats.val_loss_history):.3f})')
        val_loss_min = np.argmin(stats.val_loss_history)
        axes.plot(stats.val_history_steps[val_loss_min], stats.val_loss_history[val_loss_min], 'o', markersize=3)
    if len(stats.train_history_steps) != 0:
        axes = sns.lineplot(x=stats.train_history_steps, y=stats.train_loss_history, label=f'Training Loss (l={stats.train_loss_history[-1]:.3f}/b={min(stats.train_loss_history):.3f})', ax=axes)
        train_loss_min = np.argmin(stats.train_loss_history)
        axes.plot(stats.train_history_steps[train_loss_min], stats.train_loss_history[train_loss_min], 'o', markersize=3)
    if axes is not None:
        axes.set_xlabel('Steps')
        axes.set_ylabel('Loss / Training Loss (log scale)')
        axes.set_yscale('log')
        axes.set_title('Validation Loss and Training Loss over Steps')
        axes.grid(True)
        plt.savefig(os.path.join(directory, 'loss_curve.png'), dpi=500)
        plt.clf()

    if len(stats.lr_history_steps) != 0:
        axes = sns.lineplot(x=stats.lr_history_steps, y=stats.lr_history, label=f'Learning Rate (l={stats.lr_history[-1]:.3}/h={max(stats.lr_history):.3})')
        axes.set_xlabel('Steps')
        axes.set_ylabel('Learning Rate')
        axes.set_title('Learning Rate over Steps')
        axes.grid(True)
        plt.savefig(os.path.join(directory, 'lr_curve.png'), dpi=500)
        plt.clf()

    if len(stats.sparsity_history_steps) != 0:
        axes = sns.lineplot(x=stats.sparsity_history_steps, y=stats.sparsity_history, label=f'Sparsity Ratio (l={stats.sparsity_history[-1]:.3f})')
        axes.set_xlabel('Steps')
        axes.set_ylabel('Sparsity Ratio')
        axes.set_title('Sparsity Ratio over Steps')
        axes.grid(True)
        plt.savefig(os.path.join(directory, 'sparsity_curve.png'), dpi=500)
        plt.clf()

    if len(stats.val_history_steps) != 0 and len(stats.val_ppl_history) != 0:
        axes = sns.lineplot(x=stats.val_history_steps, y=stats.val_ppl_history, label=f'Validation Perplexity (l={stats.val_ppl_history[-1]:.3f}/b={min(stats.val_ppl_history):.3f})')
        val_ppl_min = np.argmin(stats.val_ppl_history)
        axes.plot(stats.val_history_steps[val_ppl_min], stats.val_ppl_history[val_ppl_min], 'o', markersize=3)

        axes.set_xlabel('Steps')
        axes.set_ylabel('Perplexity (log scale)')
        axes.set_yscale('log')
        axes.set_title('Perplexity over Steps')
        axes.grid(True)
        plt.savefig(os.path.join(directory, 'ppl_curve.png'), dpi=500)
        plt.clf()
