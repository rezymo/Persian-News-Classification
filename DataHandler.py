from hazm import word_tokenize, Normalizer
import matplotlib.pyplot as plt
import json
import csv
import re

class DataHandler:

    def __init__(self, path_dataset, path_stopwords):

        self.path_dataset = path_dataset

        self.path_stopwords = path_stopwords

        self.stopwords = self.file_reader(self.path_stopwords)

        self.normalizer = Normalizer()

    def file_reader(self, path):

        words = {}

        with open(path, mode="r", encoding="utf8") as file:

            for word in file:

                words[word.strip()] = ""

        return words

    def pre_processor(self, record):

        # remove url

        record['body'] = re.sub(r'^https?:\/\/.*[\r\n]*', '', record['body'], flags=re.MULTILINE)

        record['body'] = re.sub(r'^www?:\/\/.*[\r\n]*', '', record['body'], flags=re.MULTILINE)

        record['body'] = " ".join(word for word in record['body'].split()
                                  if not word.endswith(".ir") and
                                  not word.endswith(".com") and
                                  not word.endswith(".org") and
                                  not word.startswith("www.")).strip()

        record['body'] = re.sub(r'\s+\d+\s+', ' ', record['body'])

        # removeing stop words
        record['body'] = " ".join(word for word in word_tokenize(
            self.normalizer.normalize(record['body']))
                                  if word not in self.stopwords).strip()



        record['body'] = re.sub('\u00a0', ' ', record['body'])

        record['body'] = re.sub(r'\([^)]*\)', '', record['body']).\
            strip().strip("انتهای پیام")

        return record

    def reader(self):

        self.data = []

        words = {}
        for indx, record in enumerate(open(self.path_dataset, 'r')):

            record = json.loads(record)

            record["category"] = record["category"].split("-")[0].strip()

            record = self.pre_processor(record)

            self.data.append(record)

            for word in word_tokenize(record['body']):

                if word in words:

                    words[word] += 1

                else:

                    words[word] = 1

            if indx != 0 and indx % 100 == 0:

                print(indx)


        for indx, record in enumerate(self.data):

            if len(word_tokenize(record['body'])) <= 512:

                continue

            count = {}

            for word in word_tokenize(record['body']):

                count[word] = words[word]


            valid_word = [item[0] for item in sorted(words.items(), key=lambda kv: kv[1], reverse=True)[:512]]

            record['body'] = " ".join(word for indx, word in enumerate(word_tokenize(record['body']))
                                      if word in valid_word and
                                      indx <= 512).strip()

            if indx % 100 == 0:

                print("*", indx)

        # plt.hist(x=words.values(),
        #          bins=40)
        # plt.show()

        return self.data

    def writer(self, path, data):

        with open(path, "w") as f:

            writer = csv.writer(f)

            writer.writerow(["label", "title", "text", "titletext"])

            for row in data:
                writer.writerow([row['category'],
                                 row['title'],
                                 self.normalizer.normalize(row['body']),
                                 row["subtitle"]])



if __name__ == '__main__':
    
    path_dataset = "resources/news.json"

    path_trainset = "resources/trainset.csv"

    path_validset = "resources/validset.csv"

    path_testset = "resources/testset.csv"

    path_stopword = "resources/stopwords.txt"

    import pandas as pd

    # data = pd.read_csv(path_testset, index_col=False)

    x=1

    data_handler = DataHandler(path_dataset=path_dataset,
                               path_stopwords=path_stopword)

    data = data_handler.reader()

    data_handler.writer(path_trainset, data[:64800])

    data_handler.writer(path_validset, data[64800:72000])

    data_handler.writer(path_testset, data[72000:])

    x = 1