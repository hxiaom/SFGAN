import logging
import random
import time
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer, AutoModel, AutoConfig, BertPreTrainedModel

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss, average_precision_score, ndcg_score

from nsfc_data_loader import read_data, read_data_test


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 300
num_class = 45
num_epochs = 5
batch_size = 8

class MyDataset(Dataset):
    def __init__(self, *list_items):
        assert all(len(list_items[0]) == len(item) for item in list_items)
        super(MyDataset, self).__init__()
        self.list_items = list_items

    def __getitem__(self, index):
        return tuple(item[index] for item in self.list_items)

    def __len__(self):
        return len(self.list_items[0])

X_train, y_train, X_test, y_test = read_data()

tokenizer = AutoTokenizer.from_pretrained('./scibert_scivocab_uncased')
# tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

encoding = tokenizer(X_train, return_tensors='pt', padding=True, 
                    truncation=True, max_length=MAX_LENGTH)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

print("input_ids first example", input_ids[0])
print("length of first example", len(input_ids[0]))

print(tokenizer.convert_ids_to_tokens([ 102,   238,  2575,  7415,   147,  2035,   502,  7337,   131,  2010]))
trainset = MyDataset(input_ids, attention_mask, y_train)
# trainSampler = RandomSampler(trainset)
# trainloader = DataLoader(trainset, sampler=trainSampler, batch_size=batch_size)
trainloader = DataLoader(trainset, batch_size=batch_size)

encoding_test = tokenizer(X_test, return_tensors='pt', 
                        padding=True, truncation=True, max_length=MAX_LENGTH)
input_ids_test = encoding_test['input_ids']
attention_mask_test = encoding_test['attention_mask']
testset = MyDataset(input_ids_test, attention_mask_test, y_test)
# testSampler = SequentialSampler(testset)
# testloader = DataLoader(testset, sampler=testSampler, batch_size=1)
testloader = DataLoader(testset, batch_size=1)

model = AutoModel.from_pretrained("./scibert_scivocab_uncased")
# model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
config = AutoConfig.from_pretrained('./scibert_scivocab_uncased')

class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids, lr=0.0

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, model, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.bert = model
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.sig = nn.Sigmoid()

        self.init_weights()
        # def init_weights(module):
        #     if isinstance(module, (nn.Linear, nn.Embedding)):
        #         # Slightly different from the TF version which uses truncated_normal for initialization
        #         # cf https://github.com/pytorch/pytorch/pull/5617
        #         module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        #     # elif isinstance(module, BERTLayerNorm):
        #     #     module.beta.data.normal_(mean=0.0, std=config.initializer_range)
        #     #     module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
        #     if isinstance(module, nn.Linear):
        #         module.bias.data.zero_()
        # self.apply(init_weights)

    def forward(self, input_ids, attention_mask):
        #input_ids = torch.tensor(input_ids, dtype = torch.long)

        #input_ids = input_ids.unsqueeze(0)
        #attention_mask = attention_mask.unsqueeze(0)
        
        # print(input_ids)
        output = self.bert(input_ids, attention_mask=attention_mask)
        
        pooled_output =  output.pooler_output
        # print('before drop out')
        # print(pooled_output)
        # print(pooled_output.shape)
        pooled_output = self.dropout(pooled_output)
        # print('drop out')
        # print(pooled_output)
        # print(pooled_output.shape)
        logits = self.classifier(pooled_output)
        # print('logits')
        # print(logits)
        logits = self.sig(logits)
        # print('sigmoid')
        # print(logits)

        return logits

        # if labels is not None:
        #     loss_fct = BCELoss()
        #     labels = labels.unsqueeze(0)
        #     loss = loss_fct(logits, labels)
        #     return loss, logits
        # else:
        #     return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if not torch.cuda.is_available():
    raise ValueError("No GPU Available")

set_seed(666)

device = torch.device("cuda")

scibert_cls_model = BertForSequenceClassification(model, config, num_class)
scibert_cls_model.to(device)
# print(input_ids.shape)
# print(attention_mask.numpy().shape)
# print(torchsummary.summary(scibert_cls_model, [(100), (100)]))
pytorch_total_params = sum(p.numel() for p in scibert_cls_model.parameters() if p.requires_grad)
print("Total number of parameters: ", pytorch_total_params)


criterion = nn.BCELoss()
optimizer = optim.Adam(scibert_cls_model.parameters())


logger.info("***** Running training *****")
logger.info("  Num Examples = %d", len(input_ids))
logger.info("  Num Epochs = %d", num_epochs)
logger.info("  Num Batch_size = %d", batch_size)

scibert_cls_model.train()
epoch_loss = []
for epoch in trange(num_epochs):  # loop over the dataset multiple times
    logger.info("Epoch %d  is running", epoch + 1)
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs; data is a list of [inputs, labels]
        data = tuple(t.to(device) for t in data)
        inputs, attention_mask, labels = data
        labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = scibert_cls_model(inputs, attention_mask)
        # labels = labels.unsqueeze(0)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    epoch_loss.append(running_loss)
    time_consume = time.time()-start_time
    logger.info("Epoch %d : %f s, loss: %f", epoch + 1, time_consume, running_loss)
print('Finished Training')
print(epoch_loss)

## Evaluation
# Predict
scibert_cls_model.eval()
test_result = []
for i, data in enumerate(tqdm(testloader), 0):
    # get the inputs; data is a list of [inputs, labels]
    data = tuple(t.to(device) for t in data)
    inputs, attention_mask, labels = data
    labels = labels.float()

    # forward + backward + optimize
    # torch.cuda.empty_cache()
    with torch.no_grad():
        logits = scibert_cls_model(inputs, attention_mask)

    test_result.append(logits.cpu().numpy()[0])

test_result = np.array(test_result)
print(test_result)


# # threshold method
# test_result_label = test_result
# test_result_label[test_result_label>=0.3] = 1
# test_result_label[test_result_label<0.3] = 0
# print(y_test)

# argsort method
idxs = np.argsort(test_result, axis=1)[:,-2:]
test_result_label = test_result
test_result_label.fill(0)
for i in range(idxs.shape[0]):
    for j in range(idxs.shape[1]):
        # if test_result[i][idxs[i][j]] >= 0.5:
        test_result_label[i][idxs[i][j]] = 1
# test_true = y_test.argmax(axis=-1)

print(idxs)
print(test_result)
print(y_test)

y_test_label = y_test
y_test_label[y_test_label>=0.5] = 1
# Partitions Evaluation
# Precision
precision = precision_score(y_test_label, test_result_label, average=None)
precision_macro = precision_score(y_test_label, test_result_label, average='micro')
print('Precision:', precision_macro)
print(precision)

# Recall
recall = recall_score(y_test_label, test_result_label, average=None)
recall_macro = recall_score(y_test_label, test_result_label, average='micro')
print('Recall:', recall_macro)
print(recall)

# F1_score
F1 = f1_score(y_test_label, test_result_label, average=None)
F1_macro = f1_score(y_test_label, test_result_label, average='micro')
print('F1:', F1_macro)
print(F1)

# Hamming Loss
hamming = hamming_loss(y_test_label, test_result_label)
print('Hamming Loss:', hamming)
# Ranking Loss
rl = label_ranking_loss(y_test, test_result)
print('Ranking Loss:', rl)


# Rankings Evaluation
# Coverage
coverage = coverage_error(y_test, test_result)
print('Coverage Error:', coverage)

# Average Precision Score
lrap = average_precision_score(y_test, test_result)
print('Average Precision Score:', lrap)

# Ranking Loss
rl = label_ranking_loss(y_test, test_result)
print('Ranking Loss:', rl)

# ndcg_score
ndcg = ndcg_score(y_test, test_result)
print('ndcg:', ndcg)