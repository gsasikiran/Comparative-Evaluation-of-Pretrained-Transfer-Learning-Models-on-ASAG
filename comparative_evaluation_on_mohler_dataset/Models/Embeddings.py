from nltk.tokenize import word_tokenize
from allennlp.commands.elmo import ElmoEmbedder
from bert_embedding import BertEmbedding

import torch

class Embedding2Array:
    def __init__(self):
        pass

    def elmo(self, sentence):
        ''' Creates the list of arrays of each corresponding word
        parameters
        embedding : ndarray
        returns: list
          Returns the list of elmo embedding of each word
        '''
        embed_instant = self.Embeddings(sentence)
        embed = embed_instant.get_elmo_embedding()
        word_array = []

        for i in range(len(embed[2])):
            word_array.append(embed[0][i])
        return word_array

    def bert(self, sentence):

        embed_instant = self.Embeddings(sentence)
        embed = embed_instant.get_bert_embedding()

        word_array = []
        for i in range(len(embed)):
            word_array.append(embed[i][1][0])
        return word_array

    def gpt(self, sentence):
        ''' Creates the list of arrays of each corresponding word
            param
            embedding: tensor
            returns: list
              Returns the list of GPT embedding of each word
            '''
        embed_instant = self.Embeddings(sentence)
        embed = embed_instant.get_gpt_embedding()

        word_array = []

        for i in range(embed[0].shape[1]):
            word_array.append(embed[0][0][i].tolist())

        return word_array

    def gpt2(self, sentence):
        ''' Creates the list of arrays of each corresponding word
        param
        embedding: tensor
        returns: list
          Returns the list of GPT2 embedding of each word
        '''
        embed_instant = self.Embeddings(sentence)
        embed = embed_instant.get_gpt2_embedding()
        word_array = []

        for i in range(embed[0].size()[1]):
            word_array.append(embed[0][0][i].tolist())

        return word_array

    class Embeddings:

        def __init__(self, sentence):

            if not list:
                self.tokenized_sent = word_tokenize(self.sentence)
            else:
                self.tokenized_sent = sentence


        def get_elmo_embedding(self):
            '''Creates ELMo word embeddings for the given words
            param: list, list
            returns: ndarray, ndarray
              Returns the ELMo embeddings of the tokens of two sentences'''

            elmo = ElmoEmbedder()
            elmo_embedding = elmo.embed_sentence(self.tokenized_sent)

            return elmo_embedding

        def get_bert_embedding(self):
            '''Creates word embeddings taken from BERT language representation
            returns: list, list
              Returns the BERT embeddings of the tokens of two sentences'''

            bert_embedding = BertEmbedding().embedding(sentences=self.tokenized_sent)

            return bert_embedding

        def get_gpt_embedding(self):
            '''Creates word embeddings of GPT
            returns: tensor
              Returns the GPT embeddings of the tokens of sentence'''

            # model= torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTModel', 'openai-gpt')
            # tokenizer=torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')

            tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'openai-gpt')
            model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'openai-gpt')

            indexed_token = tokenizer.convert_tokens_to_ids(self.tokenized_sent)
            tokens_tensor = torch.tensor([indexed_token])

            gpt_embedding = model(tokens_tensor)

            return gpt_embedding


        def get_gpt2_embedding(self):
            '''Creates word embeddings of GPT
                returns: tensor
            Returns the GPT2 embeddings of the tokens of two sentences'''
            tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
            model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'gpt2')

            indexed_token = tokenizer.convert_tokens_to_ids(self.tokenized_sent)
            tokens_tensor = torch.tensor([indexed_token])
            gpt2_embedding = model(tokens_tensor)

            return gpt2_embedding

