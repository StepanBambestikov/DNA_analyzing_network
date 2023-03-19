import torch
import torch.optim as optim

import network_service as nn_service


def save_network_ensemble(max_attempts_number, train_loader, val_loader, train_epoch_number, network_type):
    saved_model_numper = 0
    required_max_error = 3.7
    for current_attempt in range(max_attempts_number):
        nn_model = network_type()
        loss = torch.nn.MSELoss().type(torch.FloatTensor)
        optimizer = optim.Adam(nn_model.parameters(), lr=1e-2)
        *_, val_error_history = nn_service.train_model(
            nn_model, train_loader, val_loader, loss, optimizer, train_epoch_number)

        print("test_model %f, val error:%f" % (current_attempt, val_error_history[-1]))
        if val_error_history[-1] < required_max_error:
            torch.save(nn_model.state_dict(), f"test{saved_model_numper}.pth")
            saved_model_numper += 1
            if saved_model_numper == 5:
                return
