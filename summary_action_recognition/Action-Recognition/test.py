import torch
import sys
import numpy as np
import itertools
from models import *
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="/home/swati/Documents/Action-Recognition-master/data/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
    parser.add_argument("--img_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--checkpoint_model", default = "/home/swati/Documents/Action-Recognition-master/model_checkpoints/ConvLSTM_95.pth",  type=str, help="Optional path to checkpoint model")
#    parser.add_argument("--sequence_length", type=int, default=40, help="Number of frames in each sequence")
    opt = parser.parse_args()
    print(opt)

    assert opt.checkpoint_model, "Specify path to checkpoint model using arg. '--checkpoint_model'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls_criterion = nn.CrossEntropyLoss().to(device)

    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    # Define test set
    test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length= None,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print("split_number", opt.split_number)
    print("testdataloader", len(test_dataloader))

    # Define network
    model = ConvLSTM(
        num_classes= 101,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
    )
    model = model.to(device)
    model.load_state_dict(torch.load(opt.checkpoint_model))
    model.eval()

    test_accuracies = []
    test_metrics = {"loss": [], "acc": []}
#    print("test_metrics", test_metrics)
    for batch_i, (X, y) in enumerate(test_dataloader):
        
        image_sequences = Variable(X.to(device), requires_grad=False)
        labels = Variable(y, requires_grad=False).to(device)
        with torch.no_grad():
             model.lstm.reset_hidden_state()
                # Get sequence predictions
             predictions = model(image_sequences)
            # Compute metrics
             acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
             
             
             
             loss = cls_criterion(predictions, labels).item()
             print("labels", labels.item())
            # Keep track of loss and accuracy
             test_metrics["loss"].append(loss)
             test_metrics["acc"].append(acc)
             
             f = open("100testepoch.txt", "a+")
             loss_1 = np.mean(test_metrics["loss"])
#             print("loss1", loss_1)
             acc_1 = np.mean(test_metrics["acc"])
    
             
             s = 'batch_i---' + str(batch_i) + '\t'  + 'loss---' + str(loss_1) + '\t' +'accuracy----' + str(acc_1) + '\n'
             f.write(s)
#             print("l", l)
             sys.stdout.write(
               "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                %(
                    batch_i,
                    len(test_dataloader),
                    loss,
                    np.mean(test_metrics["loss"]),
                    acc,
                    np.mean(test_metrics["acc"]),
                )
            )
    accaverage= np.average(acc_1)
    print("\n"+ "accaverage", accaverage)
#        with open("accuracies.txt","w") as file:
#                file.writelines(batch_i,
#                    len(test_dataloader),
#                    loss,
#                    np.mean(test_metrics["loss"]),
#                    acc,
#                    np.mean(test_metrics["acc"]))
   