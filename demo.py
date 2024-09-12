import math, random
import os
from torch.utils.data import random_split
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tkinter import Tk
from tkinter.filedialog import askdirectory
root = Tk()
root.withdraw()

# Open a dialog to select the folder containing m4a files
audio_folder = askdirectory(title='Select Folder Containing m4a Files')
folder_name = os.path.basename(audio_folder)

class AudioUtil:
   # ----------------------------
   # Load an audio file. Return the signal as a tensor and the sample rate
   # ----------------------------
   @staticmethod
   def open(audio_file):
      sig, sr = torchaudio.load(audio_file)
      return sig, sr
   
   # ----------------------------
   # Convert the given audio to the desired number of channels
   # ----------------------------
   @staticmethod
   def rechannel(aud, new_channel):
      sig, sr = aud

      if sig.shape[0] == new_channel:
         return aud

      if new_channel == 1:
         resig = sig[:1, :]
      else:
         resig = torch.cat([sig, sig])

      return resig, sr

   # ----------------------------
   # Resample audio to a new sample rate
   # ----------------------------
   @staticmethod
   def resample(aud, newsr):
      sig, sr = aud

      if sr == newsr:
         return aud

      num_channels = sig.shape[0]
      resig = transforms.Resample(sr, newsr)(sig[:1, :])
      if num_channels > 1:
         retwo = transforms.Resample(sr, newsr)(sig[1:, :])
         resig = torch.cat([resig, retwo])

      return resig, newsr
   
   # ----------------------------
   # Pad or truncate the signal to a fixed length 'max_ms' in milliseconds
   # ----------------------------
   @staticmethod
   def pad_trunc(aud, max_ms):
      sig, sr = aud
      num_rows, sig_len = sig.shape
      max_len = sr // 1000 * max_ms

      if sig_len > max_len:
         sig = sig[:, :max_len]
      elif sig_len < max_len:
         pad_begin_len = random.randint(0, max_len - sig_len)
         pad_end_len = max_len - sig_len - pad_begin_len

         pad_begin = torch.zeros((num_rows, pad_begin_len))
         pad_end = torch.zeros((num_rows, pad_end_len))

         sig = torch.cat((pad_begin, sig, pad_end), 1)
      
      return sig, sr
   
   # ----------------------------
   # Time shift the signal by some percent
   # ----------------------------
   @staticmethod
   def time_shift(aud, shift_limit):
      sig, sr = aud
      _, sig_len = sig.shape
      shift_amt = int(random.random() * shift_limit * sig_len)
      return sig.roll(shift_amt), sr
   
   # ----------------------------
   # Generate a Spectrogram
   # ----------------------------
   @staticmethod
   def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
      sig, sr = aud
      spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
      spec = transforms.AmplitudeToDB(top_db=80)(spec)
      return spec
   
   # ----------------------------
   # Spectrogram augmentation
   # ----------------------------
   @staticmethod
   def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
      _, n_mels, n_steps = spec.shape
      mask_value = spec.mean()
      aug_spec = spec

      freq_mask_param = max_mask_pct * n_mels
      for _ in range(n_freq_masks):
         aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec)

      time_mask_param = max_mask_pct * n_steps
      for _ in range(n_time_masks):
         aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec)

      return aug_spec

class SoundDS(Dataset):
   def __init__(self, df, data_path):
      self.df = df
      self.data_path = str(data_path)
      self.duration = 4000
      self.sr = 44100
      self.channel = 2
      self.shift_pct = 0.4
            
   def __len__(self):
      return len(self.df)
      
   def __getitem__(self, idx):
      audio_file = self.data_path + self.df.loc[idx, 'relative_path']
      class_id = self.df.loc[idx, 'classID']

      aud = AudioUtil.open(audio_file)
      reaud = AudioUtil.resample(aud, self.sr)
      rechan = AudioUtil.rechannel(reaud, self.channel)

      dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
      shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
      sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
      aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

      return aug_sgram, class_id

myds = SoundDS(df, audio_folder)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

# Model creation
class AudioClassifier(nn.Module):
   def __init__(self):
      super().__init__()
      conv_layers = []

      self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
      self.relu1 = nn.ReLU()
      self.bn1 = nn.BatchNorm2d(8)
      nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
      self.conv1.bias.data.zero_()
      conv_layers += [self.conv1, self.relu1, self.bn1]

      self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.relu2 = nn.ReLU()
      self.bn2 = nn.BatchNorm2d(16)
      nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
      self.conv2.bias.data.zero_()
      conv_layers += [self.conv2, self.relu2, self.bn2]

      self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.relu3 = nn.ReLU()
      self.bn3 = nn.BatchNorm2d(32)
      nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
      self.conv3.bias.data.zero_()
      conv_layers += [self.conv3, self.relu3, self.bn3]

      self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      self.relu4 = nn.ReLU()
      self.bn4 = nn.BatchNorm2d(64)
      nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
      self.conv4.bias.data.zero_()
      conv_layers += [self.conv4, self.relu4, self.bn4]

      self.ap = nn.AdaptiveAvgPool2d(output_size=1)
      self.lin = nn.Linear(in_features=64, out_features=10)

      self.conv = nn.Sequential(*conv_layers)

   def forward(self, x):
      x = self.conv(x)
      x = self.ap(x)
      x = x.view(x.shape[0], -1)
      x = self.lin(x)
      return x

# Create the model and put it on the GPU if available
model = nn.Parallel(AudioClassifier())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = myModel.to(device)
# Check that it is on Cuda
next(model.parameters()).device

# Training function
def training(model, train_dl, num_epochs):
   writer = SummaryWriter()
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                   steps_per_epoch=len(train_dl),
                                                   epochs=num_epochs)

   for epoch in range(num_epochs):
      running_loss = 0.0
      correct_prediction = 0
      total_prediction = 0

      for i, data in enumerate(train_dl):
         inputs, labels = data[0].to(device), data[1].to(device)

         inputs_m, inputs_s = inputs.mean(), inputs.std()
         inputs = (inputs - inputs_m) / inputs_s

         optimizer.zero_grad()

         outputs = model(inputs)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()
         scheduler.step()

         running_loss += loss.item()

         _, prediction = torch.max(outputs, 1)
         correct_prediction += (prediction == labels).sum().item()
         total_prediction += prediction.shape[0]

      avg_loss = running_loss / len(train_dl)
      avg_acc = correct_prediction / total_prediction
      writer.add_scalar("Loss/train", avg_loss, epoch)
      writer.add_scalar("Acc/train", avg_acc, epoch)
      print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {avg_acc:.2f}')

   torch.save(model.state_dict(), 'model.pt')
   print('Finished Training')

num_epochs=100
training(myModel, train_dl, num_epochs)
# ----------------------------
# Inference
# ----------------------------
def inference (model, test_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in test_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]
        
    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

# Run inference on trained model with the validation set load best model weights
model_inf = nn.DataParallel(AudioClassifier())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_inf = model_inf.to(device)
model_inf.load_state_dict(torch.load('model.pt'))
model_inf.eval()

inference(model_inf, val_dl)