import numpy as np

import torch
from transformer import TransformerModel, vocab, test_data, get_batch, bptt, criterion

BEST_MODEL_PARAMS_PATH = 'models/best_model_params.pt'

def report(model, eval_data, ntokens):

    model.eval()  # turn on evaluation mode

    total_loss = 0.

    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            #print(targets)
            seq_len = data.size(0)
            output = model(data)
            #print(output)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()

    return total_loss / (len(eval_data) - 1)


def main():

    ntokens = len(vocab)  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 2  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PARAMS_PATH)) # load best model states
    test_loss = report(model, test_data, ntokens)
    print('Test loss = ', test_loss)


main()
