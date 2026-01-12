import sys
from datetime import datetime
import torch
import logging


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=15, verbose=False, delta=0):
        """
        Args:
            save_path : save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
        elif val_loss >= self.best_loss - self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """ Save model when validation loss decrease. """
        if self.verbose and self.best_loss is not None:
            logging.info(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)


# Configure logging
class DefaultStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(f'{msg}\n')
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(path):
    # Remove existing handlers if present
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure a new logger for each iteration
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(f'{path}', mode='w'),
                                  logging.StreamHandler(sys.stdout)])

def _print(msg, logger=True):
    if logger:
        logging.info(msg)
    else:
        print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - {msg}')



