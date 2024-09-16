import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluate_model(model, eval_loader, train_dataset, eval_dates, eval_year):
    model.eval()

    dates_train, predictions_train, true_values_train = [], [], []
    dates_test, predictions_test, true_values_test = [], [], []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        counter = 0
        for X_eval, y_eval  in eval_loader:
          if eval_dates[counter].year < eval_year:
            X_train = X_eval.unsqueeze(2).to(device).to(torch.float32)
            y_train = y_eval.cpu().numpy()

            # Predict
            pred_train = model(X_train).cpu().numpy()

            predictions_train.append(pred_train)
            true_values_train.append(y_train)
            dates_train.append(eval_dates[counter: counter + X_eval.shape[0]])
          else:
            X_test = X_eval.unsqueeze(2).to(device).to(torch.float32)
            y_test = y_eval.cpu().numpy()

            # Predict
            pred_test = model(X_test).cpu().numpy()

            predictions_test.append(pred_test)
            true_values_test.append(y_test)
            dates_test.append(eval_dates[counter: counter + X_eval.shape[0]])
          counter += X_eval.shape[0]


    print(counter)
    # Reshape lists into arrays

    predictions_train = np.concatenate(predictions_train)
    true_values_train = np.concatenate(true_values_train)
    dates_train = np.concatenate(dates_train)

    predictions_test = np.concatenate(predictions_test)
    true_values_test = np.concatenate(true_values_test)
    dates_test = np.concatenate(dates_test)

    # Inverse scale the data

    predictions_rescaled_train = train_dataset.inverse_transform(predictions_train)
    y_rescaled_train = train_dataset.inverse_transform(true_values_train.reshape(-1, 1))

    predictions_rescaled_test = train_dataset.inverse_transform(predictions_test)
    y_rescaled_test = train_dataset.inverse_transform(true_values_test.reshape(-1, 1))

    # Plot predictions vs actual

    # plt.plot(dates_train, y_rescaled_train, label='True Returns, Train Data')
    # plt.plot(dates_train, predictions_rescaled_train, label='Predicted, Train Data')
    plt.plot(dates_test, y_rescaled_test, label='True Returns')
    plt.plot(dates_test, predictions_rescaled_test, label='LSTM Predicted Returns')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
