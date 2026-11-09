'''
Helpers for drawing learning curves from the training history recorded by Method_CNN
'''

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Learning_Curve_Plotter:

    @staticmethod
    def plot(history, title, save_path):
        """Loss and accuracy curves (train vs. test) for a single run."""
        epochs = range(1, len(history['train_loss']) + 1)
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
        ax_loss.plot(epochs, history['train_loss'], label='train')
        ax_loss.plot(epochs, history['test_loss'], label='test')
        ax_loss.set_xlabel('epoch'); ax_loss.set_ylabel('loss'); ax_loss.set_title('Loss')
        ax_loss.legend(); ax_loss.grid(alpha=0.3)
        ax_acc.plot(epochs, history['train_acc'], label='train')
        ax_acc.plot(epochs, history['test_acc'], label='test')
        ax_acc.set_xlabel('epoch'); ax_acc.set_ylabel('accuracy'); ax_acc.set_title('Accuracy')
        ax_acc.legend(); ax_acc.grid(alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_comparison(histories, title, save_path):
        """Overlay test accuracy / training loss of several configurations. histories: {label: history}."""
        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
        for label, h in histories.items():
            epochs = range(1, len(h['train_loss']) + 1)
            ax_loss.plot(epochs, h['train_loss'], label=label)
            ax_acc.plot(epochs, h['test_acc'], label=label)
        ax_loss.set_xlabel('epoch'); ax_loss.set_ylabel('training loss'); ax_loss.set_title('Training loss')
        ax_acc.set_xlabel('epoch'); ax_acc.set_ylabel('accuracy'); ax_acc.set_title('Test accuracy')
        for ax in (ax_loss, ax_acc):
            ax.grid(alpha=0.3)
        ax_acc.legend(fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
