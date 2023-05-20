import gc
import pickle
import logging

import model.optimiser
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import calc_measurements

logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) example_data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.example_data.Dataset instance containing local example_data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """

    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None

    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local example_data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.dataloader = DataLoader(self.data, batch_size=client_config["batch_size"], shuffle=True)
        self.local_epoch = client_config["num_local_epochs"]

    def client_update(self, mlb: MultiLabelBinarizer):
        """Update local model using local dataset."""
        self.model.train()
        self.model.to(self.device)

        optimizer = model.optimiser.adam(self.model.named_parameters())

        for e in range(self.local_epoch):
            for step, batch in enumerate(self.dataloader):
                age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch
                targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32).to(self.device)
                batch = tuple(t.to(self.device) for t in batch)
                age_ids, input_ids, posi_ids, segment_ids, attMask, _, _ = batch

                loss, logits = self.model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=attMask,
                                          labels=targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

    def client_evaluate(self, mlb: MultiLabelBinarizer):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)

        y = []
        y_label = []
        tr_loss = 0

        with torch.no_grad():
            for step, batch in enumerate(self.dataloader):
                age_ids, input_ids, posi_ids, segment_ids, attMask, targets, _ = batch
                targets = torch.tensor(mlb.transform(targets.numpy()), dtype=torch.float32).to(self.device)
                batch = tuple(t.to(self.device) for t in batch)
                age_ids, input_ids, posi_ids, segment_ids, attMask, _, _ = batch
                loss, logits = self.model(input_ids, age_ids, segment_ids, posi_ids, attention_mask=attMask,
                                          labels=targets)
                logits = logits.cpu()
                targets = targets.cpu()
                tr_loss += loss.item()
                y_label.append(targets)
                y.append(logits)

        self.model.to("cpu")
        y_label = torch.cat(y_label, dim=0)
        y = torch.cat(y, dim=0)
        aps, auc_roc, recall, f1, output, label = calc_measurements(y, y_label)

        test_loss = tr_loss / len(self.dataloader)
        print(f'client test_loss={test_loss}, len(self.dataloader)={len(self.dataloader)}, test_loss={test_loss}')

        message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test aps: {100. * aps:.2f}%\n"
        print(message, flush=True)
        logging.info(message)
        del message
        gc.collect()
        return test_loss, aps
