from train import train_model
from evaluate import evaluate_model

# Train the model
sequence_length=10
sp500Data_Train = web.DataReader('SP500', 'fred', start= "2010", end= "2022")
sp500Data_Train.dropna(inplace= True)

train_loader, train_dataset = create_dataloader( sp500Data_Train["SP500"], batch_size = 8, sequence_length = sequence_length, shuffle=False)
model, loss_history = train_model(train_loader, train_dataset, epochs=1000, lr=0.001, batch_size=64, sequence_length=60)
