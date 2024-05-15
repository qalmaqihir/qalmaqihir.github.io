================
by Jawad Haider

- <a href="#rnn-for-text-generation" id="toc-rnn-for-text-generation">RNN
  for Text Generation</a>
  - <a href="#generating-text-encoded-variables"
    id="toc-generating-text-encoded-variables">Generating Text (encoded
    variables)</a>
  - <a href="#imports" id="toc-imports">Imports</a>
  - <a href="#get-text-data" id="toc-get-text-data">Get Text Data</a>
  - <a href="#encode-entire-text" id="toc-encode-entire-text">Encode Entire
    Text</a>
  - <a href="#one-hot-encoding" id="toc-one-hot-encoding">One Hot
    Encoding</a>
  - <a href="#section" id="toc-section">————–</a>
- <a href="#creating-training-batches"
  id="toc-creating-training-batches">Creating Training Batches</a>
  - <a href="#section-1" id="toc-section-1">—————–</a>
    - <a href="#example-of-generating-a-batch"
      id="toc-example-of-generating-a-batch">Example of generating a batch</a>
  - <a href="#gpu-check" id="toc-gpu-check">GPU Check</a>
- <a href="#creating-the-lstm-model"
  id="toc-creating-the-lstm-model">Creating the LSTM Model</a>
  - <a href="#instance-of-the-model" id="toc-instance-of-the-model">Instance
    of the Model</a>
    - <a href="#optimizer-and-loss" id="toc-optimizer-and-loss">Optimizer and
      Loss</a>
  - <a href="#training-data-and-validation-data"
    id="toc-training-data-and-validation-data">Training Data and Validation
    Data</a>
- <a href="#training-the-network" id="toc-training-the-network">Training
  the Network</a>
  - <a href="#variables" id="toc-variables">Variables</a>
  - <a href="#section-2" id="toc-section-2">——-</a>
  - <a href="#saving-the-model" id="toc-saving-the-model">Saving the
    Model</a>
  - <a href="#load-model" id="toc-load-model">Load Model</a>
- <a href="#generating-predictions"
  id="toc-generating-predictions">Generating Predictions</a>

# RNN for Text Generation

## Generating Text (encoded variables)

We saw how to generate continuous values, now let’s see how to
generalize this to generate categorical sequences (such as words or
letters).

## Imports

``` python
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## Get Text Data

``` python
with open('../Data/shakespeare.txt','r',encoding='utf8') as f:
    text = f.read()
```

``` python
text[:1000]
```

    "\n                     1\n  From fairest creatures we desire increase,\n  That thereby beauty's rose might never die,\n  But as the riper should by time decease,\n  His tender heir might bear his memory:\n  But thou contracted to thine own bright eyes,\n  Feed'st thy light's flame with self-substantial fuel,\n  Making a famine where abundance lies,\n  Thy self thy foe, to thy sweet self too cruel:\n  Thou that art now the world's fresh ornament,\n  And only herald to the gaudy spring,\n  Within thine own bud buriest thy content,\n  And tender churl mak'st waste in niggarding:\n    Pity the world, or else this glutton be,\n    To eat the world's due, by the grave and thee.\n\n\n                     2\n  When forty winters shall besiege thy brow,\n  And dig deep trenches in thy beauty's field,\n  Thy youth's proud livery so gazed on now,\n  Will be a tattered weed of small worth held:  \n  Then being asked, where all thy beauty lies,\n  Where all the treasure of thy lusty days;\n  To say within thine own deep su"

``` python
print(text[:1000])
```


                         1
      From fairest creatures we desire increase,
      That thereby beauty's rose might never die,
      But as the riper should by time decease,
      His tender heir might bear his memory:
      But thou contracted to thine own bright eyes,
      Feed'st thy light's flame with self-substantial fuel,
      Making a famine where abundance lies,
      Thy self thy foe, to thy sweet self too cruel:
      Thou that art now the world's fresh ornament,
      And only herald to the gaudy spring,
      Within thine own bud buriest thy content,
      And tender churl mak'st waste in niggarding:
        Pity the world, or else this glutton be,
        To eat the world's due, by the grave and thee.


                         2
      When forty winters shall besiege thy brow,
      And dig deep trenches in thy beauty's field,
      Thy youth's proud livery so gazed on now,
      Will be a tattered weed of small worth held:  
      Then being asked, where all thy beauty lies,
      Where all the treasure of thy lusty days;
      To say within thine own deep su

``` python
len(text)
```

    5445609

## Encode Entire Text

``` python
all_characters = set(text)
```

``` python
# all_characters
```

``` python
decoder = dict(enumerate(all_characters))
```

``` python
# decoder
# decoder.items()
```

``` python
encoder = {char: ind for ind,char in decoder.items()}
```

``` python
# encoder
```

``` python
encoded_text = np.array([encoder[char] for char in text])
```

``` python
encoded_text[:500]
```

    array([51, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
           26, 26, 26, 26, 26, 41, 51, 26, 26, 31, 29, 39, 17, 26, 73, 14, 60,
           29, 46, 78, 21, 26, 35, 29, 46, 14, 21,  6, 29, 46, 78, 26,  2, 46,
           26, 65, 46, 78, 60, 29, 46, 26, 60, 55, 35, 29, 46, 14, 78, 46, 63,
           51, 26, 26, 62,  7, 14, 21, 26, 21,  7, 46, 29, 46, 74, 38, 26, 74,
           46, 14,  6, 21, 38, 77, 78, 26, 29, 39, 78, 46, 26, 17, 60, 82,  7,
           21, 26, 55, 46, 27, 46, 29, 26, 65, 60, 46, 63, 51, 26, 26, 67,  6,
           21, 26, 14, 78, 26, 21,  7, 46, 26, 29, 60, 22, 46, 29, 26, 78,  7,
           39,  6, 43, 65, 26, 74, 38, 26, 21, 60, 17, 46, 26, 65, 46, 35, 46,
           14, 78, 46, 63, 51, 26, 26, 36, 60, 78, 26, 21, 46, 55, 65, 46, 29,
           26,  7, 46, 60, 29, 26, 17, 60, 82,  7, 21, 26, 74, 46, 14, 29, 26,
            7, 60, 78, 26, 17, 46, 17, 39, 29, 38, 10, 51, 26, 26, 67,  6, 21,
           26, 21,  7, 39,  6, 26, 35, 39, 55, 21, 29, 14, 35, 21, 46, 65, 26,
           21, 39, 26, 21,  7, 60, 55, 46, 26, 39,  2, 55, 26, 74, 29, 60, 82,
            7, 21, 26, 46, 38, 46, 78, 63, 51, 26, 26, 31, 46, 46, 65, 77, 78,
           21, 26, 21,  7, 38, 26, 43, 60, 82,  7, 21, 77, 78, 26, 73, 43, 14,
           17, 46, 26,  2, 60, 21,  7, 26, 78, 46, 43, 73, 49, 78,  6, 74, 78,
           21, 14, 55, 21, 60, 14, 43, 26, 73,  6, 46, 43, 63, 51, 26, 26,  3,
           14, 28, 60, 55, 82, 26, 14, 26, 73, 14, 17, 60, 55, 46, 26,  2,  7,
           46, 29, 46, 26, 14, 74,  6, 55, 65, 14, 55, 35, 46, 26, 43, 60, 46,
           78, 63, 51, 26, 26, 62,  7, 38, 26, 78, 46, 43, 73, 26, 21,  7, 38,
           26, 73, 39, 46, 63, 26, 21, 39, 26, 21,  7, 38, 26, 78,  2, 46, 46,
           21, 26, 78, 46, 43, 73, 26, 21, 39, 39, 26, 35, 29,  6, 46, 43, 10,
           51, 26, 26, 62,  7, 39,  6, 26, 21,  7, 14, 21, 26, 14, 29, 21, 26,
           55, 39,  2, 26, 21,  7, 46, 26,  2, 39, 29, 43, 65, 77, 78, 26, 73,
           29, 46, 78,  7, 26, 39, 29, 55, 14, 17, 46, 55, 21, 63, 51, 26, 26,
           56, 55, 65, 26, 39, 55, 43, 38, 26,  7, 46, 29, 14, 43, 65, 26, 21,
           39, 26, 21,  7, 46, 26, 82, 14,  6, 65, 38, 26, 78, 22, 29, 60, 55,
           82, 63, 51, 26, 26, 40, 60, 21,  7, 60, 55, 26, 21,  7, 60, 55, 46,
           26, 39,  2, 55, 26, 74,  6])

## One Hot Encoding

As previously discussed, we need to one-hot encode our data inorder for
it to work with the network structure. Make sure to review numpy if any
of these operations confuse you!

``` python
def one_hot_encoder(encoded_text, num_uni_chars):
    '''
    encoded_text : batch of encoded text
    
    num_uni_chars = number of unique characters (len(set(text)))
    '''
    
    # METHOD FROM:
    # https://stackoverflow.com/questions/29831489/convert-encoded_textay-of-indices-to-1-hot-encoded-numpy-encoded_textay
      
    # Create a placeholder for zeros.
    one_hot = np.zeros((encoded_text.size, num_uni_chars))
    
    # Convert data type for later use with pytorch (errors if we dont!)
    one_hot = one_hot.astype(np.float32)

    # Using fancy indexing fill in the 1s at the correct index locations
    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
    

    # Reshape it so it matches the batch sahe
    one_hot = one_hot.reshape((*encoded_text.shape, num_uni_chars))
    
    return one_hot
```

``` python
one_hot_encoder(np.array([1,2,0]),3)
```

    array([[0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.]], dtype=float32)

## ————–

# Creating Training Batches

We need to create a function that will generate batches of characters
along with the next character in the sequence as a label.

## —————–

``` python
example_text = np.arange(10)
```

``` python
example_text
```

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

``` python
# If we wanted 5 batches
example_text.reshape((5,-1))
```

    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])

``` python
def generate_batches(encoded_text, samp_per_batch=10, seq_len=50):
    
    '''
    Generate (using yield) batches for training.
    
    X: Encoded Text of length seq_len
    Y: Encoded Text shifted by one
    
    Example:
    
    X:
    
    [[1 2 3]]
    
    Y:
    
    [[ 2 3 4]]
    
    encoded_text : Complete Encoded Text to make batches from
    batch_size : Number of samples per batch
    seq_len : Length of character sequence
       
    '''
    
    # Total number of characters per batch
    # Example: If samp_per_batch is 2 and seq_len is 50, then 100
    # characters come out per batch.
    char_per_batch = samp_per_batch * seq_len
    
    
    # Number of batches available to make
    # Use int() to roun to nearest integer
    num_batches_avail = int(len(encoded_text)/char_per_batch)
    
    # Cut off end of encoded_text that
    # won't fit evenly into a batch
    encoded_text = encoded_text[:num_batches_avail * char_per_batch]
    
    
    # Reshape text into rows the size of a batch
    encoded_text = encoded_text.reshape((samp_per_batch, -1))
    

    # Go through each row in array.
    for n in range(0, encoded_text.shape[1], seq_len):
        
        # Grab feature characters
        x = encoded_text[:, n:n+seq_len]
        
        # y is the target shifted over by 1
        y = np.zeros_like(x)
       
        #
        try:
            y[:, :-1] = x[:, 1:]
            y[:, -1]  = encoded_text[:, n+seq_len]
            
        # FOR POTENTIAL INDEXING ERROR AT THE END    
        except:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, 0]
            
        yield x, y
```

### Example of generating a batch

``` python
sample_text = encoded_text[:20]
```

``` python
sample_text
```

    array([51, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
           26, 26, 26])

``` python
batch_generator = generate_batches(sample_text,samp_per_batch=2,seq_len=5)
```

``` python
# Grab first batch
x, y = next(batch_generator)
```

``` python
x
```

    array([[51, 26, 26, 26, 26],
           [26, 26, 26, 26, 26]])

``` python
y
```

    array([[26, 26, 26, 26, 26],
           [26, 26, 26, 26, 26]])

------------------------------------------------------------------------

## GPU Check

Remember this will take a lot longer on CPU!

``` python
torch.cuda.is_available()
```

    True

# Creating the LSTM Model

**Note! We will have options for GPU users and CPU users. CPU will take
MUCH LONGER to train and you may encounter RAM issues depending on your
hardware. If that is the case, consider using cloud services like AWS,
GCP, or Azure. Note, these may cost you money to use!**

``` python
class CharModel(nn.Module):
    
    def __init__(self, all_chars, num_hidden=256, num_layers=4,drop_prob=0.5,use_gpu=False):
        
        
        # SET UP ATTRIBUTES
        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_gpu = use_gpu
        
        #CHARACTER SET, ENCODER, and DECODER
        self.all_chars = all_chars
        self.decoder = dict(enumerate(all_chars))
        self.encoder = {char: ind for ind,char in decoder.items()}
        
        
        self.lstm = nn.LSTM(len(self.all_chars), num_hidden, num_layers, dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc_linear = nn.Linear(num_hidden, len(self.all_chars))
      
    
    def forward(self, x, hidden):
                  
        
        lstm_output, hidden = self.lstm(x, hidden)
        
        
        drop_output = self.dropout(lstm_output)
        
        drop_output = drop_output.contiguous().view(-1, self.num_hidden)
        
        
        final_out = self.fc_linear(drop_output)
        
        
        return final_out, hidden
    
    
    def hidden_state(self, batch_size):
        '''
        Used as separate method to account for both GPU and CPU users.
        '''
        
        if self.use_gpu:
            
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda(),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda())
        else:
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden))
        
        return hidden
        
```

## Instance of the Model

``` python
model = CharModel(
    all_chars=all_characters,
    num_hidden=512,
    num_layers=3,
    drop_prob=0.5,
    use_gpu=True,
)
```

``` python
total_param  = []
for p in model.parameters():
    total_param.append(int(p.numel()))
```

Try to make the total_parameters be roughly the same magnitude as the
number of characters in the text.

``` python
sum(total_param)
```

    5470292

``` python
len(encoded_text)
```

    5445609

### Optimizer and Loss

``` python
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()
```

## Training Data and Validation Data

``` python
# percentage of data to be used for training
train_percent = 0.1
```

``` python
len(encoded_text)
```

    5445609

``` python
int(len(encoded_text) * (train_percent))
```

    544560

``` python
train_ind = int(len(encoded_text) * (train_percent))
```

``` python
train_data = encoded_text[:train_ind]
val_data = encoded_text[train_ind:]
```

# Training the Network

## Variables

Feel free to play around with these values!

``` python
## VARIABLES

# Epochs to train for
epochs = 50
# batch size 
batch_size = 128

# Length of sequence
seq_len = 100

# for printing report purposes
# always start at 0
tracker = 0

# number of characters in text
num_char = max(encoded_text)+1
```

------------------------------------------------------------------------

``` python
# Set model to train
model.train()


# Check to see if using GPU
if model.use_gpu:
    model.cuda()

for i in range(epochs):
    
    hidden = model.hidden_state(batch_size)
    
    
    for x,y in generate_batches(train_data,batch_size,seq_len):
        
        tracker += 1
        
        # One Hot Encode incoming data
        x = one_hot_encoder(x,num_char)
        
        # Convert Numpy Arrays to Tensor
        
        inputs = torch.from_numpy(x)
        targets = torch.from_numpy(y)
        
        # Adjust for GPU if necessary
        
        if model.use_gpu:
            
            inputs = inputs.cuda()
            targets = targets.cuda()
            
        # Reset Hidden State
        # If we dont' reset we would backpropagate through all training history
        hidden = tuple([state.data for state in hidden])
        
        model.zero_grad()
        
        lstm_output, hidden = model.forward(inputs,hidden)
        loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
        
        loss.backward()
        
        # POSSIBLE EXPLODING GRADIENT PROBLEM!
        # LET"S CLIP JUST IN CASE
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
        
        optimizer.step()
        
        
        
        ###################################
        ### CHECK ON VALIDATION SET ######
        #################################
        
        if tracker % 25 == 0:
            
            val_hidden = model.hidden_state(batch_size)
            val_losses = []
            model.eval()
            
            for x,y in generate_batches(val_data,batch_size,seq_len):
                
                # One Hot Encode incoming data
                x = one_hot_encoder(x,num_char)
                

                # Convert Numpy Arrays to Tensor

                inputs = torch.from_numpy(x)
                targets = torch.from_numpy(y)

                # Adjust for GPU if necessary

                if model.use_gpu:

                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    
                # Reset Hidden State
                # If we dont' reset we would backpropagate through 
                # all training history
                val_hidden = tuple([state.data for state in val_hidden])
                
                lstm_output, val_hidden = model.forward(inputs,val_hidden)
                val_loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
        
                val_losses.append(val_loss.item())
            
            # Reset to training model after val for loop
            model.train()
            
            print(f"Epoch: {i} Step: {tracker} Val Loss: {val_loss.item()}")
```

    Epoch: 0 Step: 25 Val Loss: 3.241183280944824
    Epoch: 1 Step: 50 Val Loss: 3.2209343910217285
    Epoch: 1 Step: 75 Val Loss: 3.2246036529541016
    Epoch: 2 Step: 100 Val Loss: 3.103549003601074
    Epoch: 2 Step: 125 Val Loss: 3.0078160762786865
    Epoch: 3 Step: 150 Val Loss: 2.8424694538116455
    Epoch: 4 Step: 175 Val Loss: 2.7311224937438965
    Epoch: 4 Step: 200 Val Loss: 2.6245357990264893
    Epoch: 5 Step: 225 Val Loss: 2.530056953430176
    Epoch: 5 Step: 250 Val Loss: 2.511744737625122
    Epoch: 6 Step: 275 Val Loss: 2.424506187438965
    Epoch: 7 Step: 300 Val Loss: 2.3757734298706055
    Epoch: 7 Step: 325 Val Loss: 2.3281121253967285
    Epoch: 8 Step: 350 Val Loss: 2.287860631942749
    Epoch: 8 Step: 375 Val Loss: 2.258666515350342
    Epoch: 9 Step: 400 Val Loss: 2.219432830810547
    Epoch: 10 Step: 425 Val Loss: 2.1962826251983643
    Epoch: 10 Step: 450 Val Loss: 2.1531155109405518
    Epoch: 11 Step: 475 Val Loss: 2.12485408782959
    Epoch: 11 Step: 500 Val Loss: 2.102055072784424
    Epoch: 12 Step: 525 Val Loss: 2.0815775394439697
    Epoch: 13 Step: 550 Val Loss: 2.065098524093628
    Epoch: 13 Step: 575 Val Loss: 2.045565366744995
    Epoch: 14 Step: 600 Val Loss: 2.024740695953369
    Epoch: 14 Step: 625 Val Loss: 2.002650737762451
    Epoch: 15 Step: 650 Val Loss: 1.9918841123580933
    Epoch: 16 Step: 675 Val Loss: 1.973698616027832
    Epoch: 16 Step: 700 Val Loss: 1.9563044309616089
    Epoch: 17 Step: 725 Val Loss: 1.941154956817627
    Epoch: 17 Step: 750 Val Loss: 1.9296284914016724
    Epoch: 18 Step: 775 Val Loss: 1.9155606031417847
    Epoch: 19 Step: 800 Val Loss: 1.9066412448883057
    Epoch: 19 Step: 825 Val Loss: 1.8944147825241089
    Epoch: 20 Step: 850 Val Loss: 1.8825374841690063
    Epoch: 20 Step: 875 Val Loss: 1.8753138780593872
    Epoch: 21 Step: 900 Val Loss: 1.8679431676864624
    Epoch: 22 Step: 925 Val Loss: 1.8626611232757568
    Epoch: 22 Step: 950 Val Loss: 1.8534228801727295
    Epoch: 23 Step: 975 Val Loss: 1.8416558504104614
    Epoch: 23 Step: 1000 Val Loss: 1.8408966064453125
    Epoch: 24 Step: 1025 Val Loss: 1.832461953163147
    Epoch: 24 Step: 1050 Val Loss: 1.8274987936019897
    Epoch: 25 Step: 1075 Val Loss: 1.8215422630310059
    Epoch: 26 Step: 1100 Val Loss: 1.8141027688980103
    Epoch: 26 Step: 1125 Val Loss: 1.8090591430664062
    Epoch: 27 Step: 1150 Val Loss: 1.808109998703003
    Epoch: 27 Step: 1175 Val Loss: 1.798502802848816
    Epoch: 28 Step: 1200 Val Loss: 1.8020660877227783
    Epoch: 29 Step: 1225 Val Loss: 1.7935495376586914
    Epoch: 29 Step: 1250 Val Loss: 1.7842048406600952
    Epoch: 30 Step: 1275 Val Loss: 1.7775088548660278
    Epoch: 30 Step: 1300 Val Loss: 1.7796084880828857
    Epoch: 31 Step: 1325 Val Loss: 1.778605341911316
    Epoch: 32 Step: 1350 Val Loss: 1.778555154800415
    Epoch: 32 Step: 1375 Val Loss: 1.7726141214370728
    Epoch: 33 Step: 1400 Val Loss: 1.7713408470153809
    Epoch: 33 Step: 1425 Val Loss: 1.7647587060928345
    Epoch: 34 Step: 1450 Val Loss: 1.7639307975769043
    Epoch: 35 Step: 1475 Val Loss: 1.7668451070785522
    Epoch: 35 Step: 1500 Val Loss: 1.7553269863128662
    Epoch: 36 Step: 1525 Val Loss: 1.7537274360656738
    Epoch: 36 Step: 1550 Val Loss: 1.7476931810379028
    Epoch: 37 Step: 1575 Val Loss: 1.7471405267715454
    Epoch: 38 Step: 1600 Val Loss: 1.748685359954834
    Epoch: 38 Step: 1625 Val Loss: 1.7501276731491089
    Epoch: 39 Step: 1650 Val Loss: 1.7491378784179688
    Epoch: 39 Step: 1675 Val Loss: 1.73957097530365
    Epoch: 40 Step: 1700 Val Loss: 1.7412303686141968
    Epoch: 41 Step: 1725 Val Loss: 1.7421422004699707
    Epoch: 41 Step: 1750 Val Loss: 1.7420353889465332
    Epoch: 42 Step: 1775 Val Loss: 1.732686161994934
    Epoch: 42 Step: 1800 Val Loss: 1.7336872816085815
    Epoch: 43 Step: 1825 Val Loss: 1.7360546588897705
    Epoch: 44 Step: 1850 Val Loss: 1.7357029914855957
    Epoch: 44 Step: 1875 Val Loss: 1.736457109451294
    Epoch: 45 Step: 1900 Val Loss: 1.7330776453018188
    Epoch: 45 Step: 1925 Val Loss: 1.7337615489959717
    Epoch: 46 Step: 1950 Val Loss: 1.738358736038208
    Epoch: 47 Step: 1975 Val Loss: 1.7346129417419434
    Epoch: 47 Step: 2000 Val Loss: 1.743545413017273
    Epoch: 48 Step: 2025 Val Loss: 1.7326579093933105
    Epoch: 48 Step: 2050 Val Loss: 1.7226899862289429
    Epoch: 49 Step: 2075 Val Loss: 1.7329885959625244
    Epoch: 49 Step: 2100 Val Loss: 1.7302632331848145

## ——-

## Saving the Model

https://pytorch.org/tutorials/beginner/saving_loading_models.html

``` python
# Be careful to overwrite our original name file!
model_name = 'example.net'
```

``` python
torch.save(model.state_dict(),model_name)
```

## Load Model

``` python
# MUST MATCH THE EXACT SAME SETTINGS AS MODEL USED DURING TRAINING!

model = CharModel(
    all_chars=all_characters,
    num_hidden=512,
    num_layers=3,
    drop_prob=0.5,
    use_gpu=True,
)
```

``` python
model.load_state_dict(torch.load(model_name))
model.eval()
```

    CharModel(
      (lstm): LSTM(84, 512, num_layers=3, batch_first=True, dropout=0.5)
      (dropout): Dropout(p=0.5)
      (fc_linear): Linear(in_features=512, out_features=84, bias=True)
    )

# Generating Predictions

------------------------------------------------------------------------

``` python
def predict_next_char(model, char, hidden=None, k=1):
        
        # Encode raw letters with model
        encoded_text = model.encoder[char]
        
        # set as numpy array for one hot encoding
        # NOTE THE [[ ]] dimensions!!
        encoded_text = np.array([[encoded_text]])
        
        # One hot encoding
        encoded_text = one_hot_encoder(encoded_text, len(model.all_chars))
        
        # Convert to Tensor
        inputs = torch.from_numpy(encoded_text)
        
        # Check for CPU
        if(model.use_gpu):
            inputs = inputs.cuda()
        
        
        # Grab hidden states
        hidden = tuple([state.data for state in hidden])
        
        
        # Run model and get predicted output
        lstm_out, hidden = model(inputs, hidden)

        
        # Convert lstm_out to probabilities
        probs = F.softmax(lstm_out, dim=1).data
        
        
        
        if(model.use_gpu):
            # move back to CPU to use with numpy
            probs = probs.cpu()
        
        
        # k determines how many characters to consider
        # for our probability choice.
        # https://pytorch.org/docs/stable/torch.html#torch.topk
        
        # Return k largest probabilities in tensor
        probs, index_positions = probs.topk(k)
        
        
        index_positions = index_positions.numpy().squeeze()
        
        # Create array of probabilities
        probs = probs.numpy().flatten()
        
        # Convert to probabilities per index
        probs = probs/probs.sum()
        
        # randomly choose a character based on probabilities
        char = np.random.choice(index_positions, p=probs)
       
        # return the encoded value of the predicted char and the hidden state
        return model.decoder[char], hidden
```

``` python
def generate_text(model, size, seed='The', k=1):
        
      
    
    # CHECK FOR GPU
    if(model.use_gpu):
        model.cuda()
    else:
        model.cpu()
    
    # Evaluation mode
    model.eval()
    
    # begin output from initial seed
    output_chars = [c for c in seed]
    
    # intiate hidden state
    hidden = model.hidden_state(1)
    
    # predict the next character for every character in seed
    for char in seed:
        char, hidden = predict_next_char(model, char, hidden, k=k)
    
    # add initial characters to output
    output_chars.append(char)
    
    # Now generate for size requested
    for i in range(size):
        
        # predict based off very last letter in output_chars
        char, hidden = predict_next_char(model, output_chars[-1], hidden, k=k)
        
        # add predicted character
        output_chars.append(char)
    
    # return string of predicted text
    return ''.join(output_chars)
```

``` python
print(generate_text(model, 1000, seed='The ', k=3))
```

    The will true and breathed to me.
        If thou wert better to the stare and send thee,
        Which hath any trives and sound and stretged,
        That have the better send of the constance,
        That then that thou shaltst but that have seem surpet
        And we had been the self-fight and had their strange,
        With his sward shall strave a servant state.
        Where this't she is that to the wind of held
        That have this serve that she he with the child
        Which they were beauty of their command strowes
        And truth and strength to the serves and song.
        If thou say'st he that hath seen this should still
        To she with his both shall see him.
        The world was a solder thou to heaven with me,
        And should this can stay that I heave make
        Which his charge in her shames, and to his state.
        That have tho stol'd of this starts to have,  
        And we and to the cheeks that to the stol'd
        To serve the courtier time of that sense is.
        In the summer that that shall not,
        That he will s

<center>

<a href=''> ![Logo](../logo1.png) </a>

</center>
<center>
<em>Copyright Qalmaqihir</em>
</center>
<center>
<em>For more information, visit us at
<a href='http://www.github.com/qalmaqihir/'>www.github.com/qalmaqihir/</a></em>
</center>
