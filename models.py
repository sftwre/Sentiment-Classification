# models.py
import torch
from torch.utils.tensorboard import SummaryWriter
from sentiment_data import *
from net import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, model:nn.Module):
        self.model = model
        self.model.eval()

    def predict(self, ex_words: List[str]) -> int:

        log_probs = self.model(ex_words)
        y_hat = torch.argmax(log_probs, dim=1).item()
        return y_hat



def pad_to_length(ex, length):
    """
    Forces np_arr to length by either truncation (if longer) or zero-padding (if shorter)
    :param np_arr:
    :param length: Length to pad to
    :return: a new numpy array with the data from np_arr padded to be of length length. If length is less than the
    length of the base array, truncates instead.
    """
    np_arr = np.array(ex)

    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


def train_ffnn(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_vectors: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Train a feedforward neural network on the given training examples, using dev_exs for development, and returns
    a trained NeuralSentimentClassifier.
    :param args: Command-line args so you can access them here
    :param train_exs:
    :param dev_exs:
    :param word_vectors:
    :return: the trained NeuralSentimentClassifier model -- this model should wrap your PyTorch module and implement
    prediction on new examples
    """

    n_classes = 2
    n_samples = len(train_exs)
    lr = args.lr
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    n_epochs = args.num_epochs

    # instantiate model
    model = FeedForward(word_vectors)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter()

    def _train(model:FeedForward, train_idxs:np.ndarray, train_exs:List[SentimentExample]) -> float:
        """
        Performs on epoch of training on the model
        :returns: avg. loss over the epoch
        """

        model.train()

        # shuffle dataset
        np.random.shuffle(train_idxs)

        total_loss = 0.0

        # train model on examples
        for idx in train_idxs:
            ex = train_exs[idx]

            x = ex.words
            y = torch.tensor([ex.label])

            # zero out previous gradients
            model.zero_grad()

            log_probs = model(x)
            loss = F.nll_loss(log_probs, y)
            total_loss += loss

            loss.backward()
            optim.step()

        avg_loss = total_loss / n_samples
        return avg_loss


    def _test(model:FeedForward, dev_exs:List[SentimentExample]) -> dict:
        """
        Tests the model on the development examples
        :returns: F1, Recall, and Precision scores for binary sentiment class. on the dev set
        """
        model.eval()

        golds = list()
        predictions = list()

        with torch.no_grad():
            for ex in dev_exs:

                x = ex.words
                y = ex.label
                golds.append(y)

                log_probs = model(x)
                y_hat = torch.argmax(log_probs, dim=1)
                predictions.extend(y_hat.tolist())

        num_pred = 0
        num_gold = 0
        num_total = 0
        num_correct = 0
        num_pos_correct = 0

        # compute F1, Recall, and Precision
        for gold, prediction in zip(golds, predictions):

            if prediction == gold:
                num_correct += 1
            if prediction == 1:
                num_pred += 1
            if gold == 1:
                num_gold += 1
            if prediction == 1 and gold == 1:
                num_pos_correct += 1
            num_total += 1

        acc = float(num_correct) / num_total
        prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
        rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0

        metrics = {'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec}
        return metrics

    train_idxs = np.arange(len(train_exs))

    # train for one epoch then evaluate
    for e in range(n_epochs):

        avg_loss = _train(model, train_idxs, train_exs)
        metrics = _test(model, dev_exs)

        f1 = metrics['f1']
        acc = metrics['acc']

        print(f"===> Epoch: {e}, Avg. loss: {avg_loss:.3f}, F1: {f1:.3f}, Acc: {acc:.3f}")

        writer.add_scalar('Train/ Avg. loss', avg_loss, e)
        writer.add_scalar('Dev/ F1', f1, e)
        writer.add_scalar('Dev/ Accuracy', acc, e)

    writer.flush()
    writer.close()

    # return trained model
    nsf = NeuralSentimentClassifier(model)
    return nsf


# Analogous to train_ffnn, but trains your fancier model.
def train_fancy(args, train_exs: List[SentimentExample],
                dev_exs: List[SentimentExample],
                word_vectors: WordEmbeddings) -> NeuralSentimentClassifier:

    lr = args.lr
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    n_epochs = args.num_epochs

    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    n_samples = len(train_exs)
    n_batches = n_samples // batch_size

    # instantiate model
    model = RNN(word_vectors, h_size=hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # convert training set to np array
    train_exs_arr = np.array(train_exs)
    train_idxs = np.arange(train_exs_arr.size)

    def _word2idxs(ex:SentimentExample):
        """
        Converts words in the sentence to it's corresponding index in the vocabulary.
        """
        words = ex.words

        indices = list()

        for w in words:
            idx = word_vectors.word_indexer.index_of(w)
            if idx == -1:
                idx = word_vectors.word_indexer.index_of("UNK")
            indices.append(idx)

        return indices

    def _train(model:RNN) -> float:
        """
        Trains RNN on a single epoch of batched examples
        """
        model.train()

        # shuffle dataset
        np.random.shuffle(train_idxs)

        total_loss = 0.0

        # train model on batches
        for start in range(0, train_idxs.size, batch_size):

            # get batch of samples
            end = start + batch_size
            batch_idxs = train_idxs[start:end]

            input = np.zeros((batch_size, seq_max_len))
            exs = train_exs_arr[batch_idxs]

            # convert sentences to indices and pad to max seq length
            indices = np.array(list(map(_word2idxs, exs)))

            for i, ex in enumerate(indices):
                input[i,:] = pad_to_length(ex, seq_max_len)

            input = torch.LongTensor(input)

            # gold labels
            y = torch.tensor(list(map(lambda x: x.label, exs)))

            # pack input and labels
            # lengths = torch.tensor(np.full(batch_size, seq_max_len))
            #
            # input = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
            # y = nn.utils.rnn.pack_padded_sequence(y, lengths)

            # zero out gradients from previous computations
            model.zero_grad()

            log_probs = model(input)
            loss = F.nll_loss(log_probs, y)
            total_loss += loss

            loss.backward()
            optim.step()

        avg_loss = total_loss / n_batches
        return avg_loss

    def _test(model:RNN):

        model.eval()

        golds = list()
        predictions = list()

        with torch.no_grad():
            for ex in dev_exs:
                x = ex.words
                y = ex.label
                golds.append(y)

                # convert words to indices
                x = _word2idxs(ex)
                x = torch.LongTensor(pad_to_length(x, seq_max_len)).reshape((1, seq_max_len))

                log_probs = model(x)
                y_hat = torch.argmax(log_probs, dim=1)
                predictions.append(y_hat.item())

        num_pred = 0
        num_gold = 0
        num_total = 0
        num_correct = 0
        num_pos_correct = 0

        # compute F1, Recall, and Precision
        for gold, prediction in zip(golds, predictions):

            if prediction == gold:
                num_correct += 1
            if prediction == 1:
                num_pred += 1
            if gold == 1:
                num_gold += 1
            if prediction == 1 and gold == 1:
                num_pos_correct += 1
            num_total += 1

        acc = float(num_correct) / num_total
        prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
        rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0

        metrics = {'f1': f1, 'acc': acc, 'prec': prec, 'rec': rec}
        return metrics

    # train for one epoch then evaluate
    for e in range(n_epochs):
        avg_loss = _train(model)
        metrics = _test(model)

        f1 = metrics['f1']
        acc = metrics['acc']

        print(f"===> Epoch: {e}, Avg. loss: {avg_loss:.3f}, F1: {f1:.3f}, Acc: {acc:.3f}")

    nsc = NeuralSentimentClassifier(model)
    return nsc