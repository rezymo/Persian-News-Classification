# Libraries
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from BERT_Model import BERT
import torch.optim as optim
import seaborn as sns
import pickle
import torch
import sys


class NewsClassification:

    def __init__(self):

        self.bert_name = 'bert-base-multilingual-cased'

        self.path_lbl2indx = "resources/lbl2indx.pkl"

        self.destination_folder = "outputs/"

        self.source_folder = "resources/"

        self.trainset_name = "trainset.csv"

        self.testset_name = "testset.csv"

        self.validset_name = "validset.csv"

    @staticmethod
    def pickle_reader(path):

        with open(path, mode="rb") as file:

            data = pickle.load(file)

            return data

    def configuration(self):

        self.lbl2indx = self.pickle_reader(self.path_lbl2indx)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)

        self.model_parameters()

    def model_parameters(self):

        # Model parameter

        self.MAX_SEQ_LEN = 128

        self.batch_size = 16

        self.num_epochs = 5

        self.PAD_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        self.UNK_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)

    def data_configuration(self):

        # Fields

        self.label_field = Field(sequential=False, use_vocab=False,
                                 batch_first=True, dtype=torch.float)

        self.text_field = Field(use_vocab=False, tokenize=self.tokenizer.encode,
                                lower=False, include_lengths=False, batch_first=True,
                                fix_length=self.MAX_SEQ_LEN, pad_token=self.PAD_INDEX,
                                unk_token=self.UNK_INDEX)

        self.fields = [('label', self.label_field),
                       ('title', self.text_field),
                       ('text', self.text_field),
                       ('titletext', self.text_field)]

        # TabularDataset

        self.trainset, self.validset, self.testset = TabularDataset.splits(path=self.source_folder,
                                                                           train=self.trainset_name,
                                                                           validation=self.testset_name,
                                                                           test=self.validset_name,
                                                                           format='CSV',
                                                                           fields=self.fields,
                                                                           skip_header=True)

        # Iterators

        self.train_iter = BucketIterator(self.trainset,
                                         batch_size=self.batch_size,
                                         sort_key=lambda x: len(x.text),
                                         device=self.device,
                                         train=True,
                                         sort=True,
                                         sort_within_batch=True)

        self.valid_iter = BucketIterator(self.validset,
                                         batch_size=self.batch_size,
                                         sort_key=lambda x: len(x.text),
                                         device=self.device,
                                         train=True,
                                         sort=True,
                                         sort_within_batch=True)

        self.test_iter = Iterator(self.testset,
                                  batch_size=self.batch_size,
                                  device=self.device,
                                  train=False,
                                  shuffle=False,
                                  sort=False)

    # Save and Load Functions

    def save_checkpoint(self, save_path, model, valid_loss):
        if save_path == None:
            return

        state_dict = {'model_state_dict': model.state_dict(),
                      'valid_loss': valid_loss}

        torch.save(state_dict, save_path)

        # print(f'Model saved to ==> {save_path}')

    def load_checkpoint(self, load_path, model):
        if load_path == None:
            return

        state_dict = torch.load(load_path, map_location=self.device)
        # print(f'Model loaded from <== {load_path}')

        model.load_state_dict(state_dict['model_state_dict'])
        return state_dict['valid_loss']

    def save_metrics(self, save_path, train_loss_list, valid_loss_list, global_steps_list):

        if save_path == None:
            return

        state_dict = {'train_loss_list': train_loss_list,
                      'valid_loss_list': valid_loss_list,
                      'global_steps_list': global_steps_list}

        torch.save(state_dict, save_path)
        print(f'Model saved to ==> {save_path}')

    def load_metrics(self, load_path):

        if load_path == None:
            return

        state_dict = torch.load(load_path, map_location=self.device)
        print(f'Model loaded from <== {load_path}')

        return state_dict['train_loss_list'], state_dict['valid_loss_list'], \
               state_dict['global_steps_list']

    # Training Function

    def train(self,
              model,
              optimizer,
              best_valid_loss=float("Inf")):

        # initialize running values
        eval_every = len(self.train_iter) // 2
        running_loss = 0.0
        valid_running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []

        # training loop
        model.train()

        for epoch in range(self.num_epochs):

            for (labels, title, text, titletext), _ in self.train_iter:

                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)
                text = text.type(torch.LongTensor)
                text = text.to(self.device)

                output = model(text, labels)
                loss, _ = output

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update running values
                running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % eval_every == 0:

                    model.eval()

                    with torch.no_grad():

                        # validation loop
                        for (labels, title, text, titletext), _ in self.valid_iter:

                            labels = labels.type(torch.LongTensor)
                            labels = labels.to(self.device)
                            text = text.type(torch.LongTensor)
                            text = text.to(self.device)
                            output = model(text, labels)
                            loss, _ = output

                            valid_running_loss += loss.item()

                    # evaluation
                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(self.valid_iter)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)

                    # resetting running values
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()

                    # print progress
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                          .format(epoch + 1, self.num_epochs, global_step,
                                  self.num_epochs * len(self.train_iter),
                                  average_train_loss, average_valid_loss))

                    # checkpoint
                    if best_valid_loss > average_valid_loss:

                        best_valid_loss = average_valid_loss

                        self.save_checkpoint(self.destination_folder + '/' + 'model.pt',
                                             model, best_valid_loss)

                        self.save_metrics(self.destination_folder + '/' + 'metrics.pt',
                                          train_loss_list, valid_loss_list, global_steps_list)

        self.save_metrics(self.destination_folder + '/' + 'metrics.pt', train_loss_list,
                          valid_loss_list, global_steps_list)

        self.loss_curve()

        print('Finished Training!')

    def loss_curve(self):

        train_loss_list, valid_loss_list, global_steps_list = \
            self.load_metrics(self.destination_folder + '/metrics.pt')

        plt.plot(global_steps_list, train_loss_list, label='Train')

        plt.plot(global_steps_list, valid_loss_list, label='Valid')

        plt.xlabel('Global Steps')

        plt.ylabel('Loss')

        plt.legend()

        plt.show()

        plt.savefig(self.destination_folder + "/loss_curve.png")

    # Testing Function

    def test(self, model, text):

        model.eval()

        with torch.no_grad():

            text = torch.FloatTensor(self.tokenizer.encode(text)).unsqueeze(0)

            labels = torch.FloatTensor([1.0])

            labels = labels.type(torch.LongTensor)

            labels = labels.to(self.device)

            text = text.type(torch.LongTensor)

            text = text.to(self.device)

            output = model(text, labels)

            _, output = output

            return torch.argmax(output, 1).tolist()

    # Evaluation Function

    def evaluate(self, model):
        y_pred = []
        y_true = []

        model.eval()
        with torch.no_grad():
            for (labels, title, text, titletext), _ in self.test_iter:
                labels = labels.type(torch.LongTensor)
                labels = labels.to(self.device)

                text = text.type(torch.LongTensor)
                text = text.to(self.device)

                output = model(text, labels)

                _, output = output

                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())

        print('Classification Report:')

        print(classification_report(y_true=y_true,
                                    y_pred=y_pred,
                                    labels=list(self.lbl2indx.values()),
                                    digits=4))

        print("F1-Score (Macro):",
              f1_score(y_true=y_true,
              y_pred=y_pred,
              average="macro"))

        print("F1-Score (Micro):",
              f1_score(y_true=y_true,
              y_pred=y_pred,
              average="micro"))

        cm = confusion_matrix(y_true, y_pred, labels=list(self.lbl2indx.values()))

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)

        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

        ax.set_title('Confusion Matrix')

        ax.set_xlabel('Predicted Labels')

        ax.set_ylabel('True Labels')

        fig.savefig(self.destination_folder + '/confusion_matrix.png')

    def configuration_optimizer(self, model):

        self.optimizer = optim.Adam(model.parameters(), lr=2e-5)

        return self.optimizer

if __name__ == '__main__':

    news_classification = NewsClassification()

    news_classification.configuration()

    model = BERT().to(news_classification.device)

    args = sys.argv

    # for training
    if args[1] == "-tr":

        optimizer = news_classification.configuration_optimizer(model)

        news_classification.data_configuration()

        news_classification.train(model=model, optimizer=optimizer)

    # for evaluating
    elif args[1] == "-e":

        news_classification.load_checkpoint(news_classification.destination_folder + '/model.pt', model)

        news_classification.data_configuration()

        news_classification.evaluate(model, news_classification.test_iter)

    # for user testing

    elif args[1] == "-t" and \
            args[2] and \
            isinstance(args[2], str):

        test_text = args[2]

        news_classification.load_checkpoint(news_classification.destination_folder + '/model.pt', model)

        result = news_classification.test(model, test_text)

        result = list(news_classification.lbl2indx.keys())[
            list(news_classification.lbl2indx.values()).index(result[0])]

        print(f"Class is : {result}")

    else:

        print("Input information is incorrect, \n"
              "Please try again")