from math import ceil
from multiprocessing import cpu_count
from transformers import BertTokenizer
from joblib import Parallel, delayed
from torch.utils.data import TensorDataset, random_split
import torch


class Bert():
    def __init__(self, model='bert-base-uncased', max_length=64, quiet=False):
        self.quiet = quiet
        self.max_length = max_length

        self.check_hardware()
        # Load the BERT tokenizer.

        if not quiet:
            print('Loading BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained(model,
                                                       do_lower_case=True)

    def check_hardware(self):
        """
        Use the best possible architecture
        """
        if torch.cuda.is_available():
            self.gpu = True

            # Tell PyTorch to use the GPU.
            self.device = torch.device("cuda")

            if not self.quiet:
                print("There are %d GPU(s) available."
                      % torch.cuda.device_count())
                print('Using the GPU:', torch.cuda.get_device_name(0))
        else:
            self.gpu = False
            self.device = torch.device("cpu")
            self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1

            if not self.quiet:
                print('No GPU available :(')
                print(f"Using {self.num_cpus} CPUs instead.")

    def encode(self, X):
        """
        `encode_plus` will:
        (1) Tokenize the sentence.
        (2) Prepend the `[CLS]` token to the start.
        (3) Append the `[SEP]` token to the end.
        (4) Map tokens to their IDs.
        (5) Pad or truncate the sentence to `max_length`
        (6) Create attention masks for [PAD] tokens.
        """
        encoded_dict = self.tokenizer.batch_encode_plus(
            X,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt')

        # Add the encoded sentence
        input_ids = encoded_dict['input_ids']

        # And its attention mask (simply differentiates padding from
        # non-padding).
        attention_masks = encoded_dict['attention_mask']

        return input_ids, attention_masks

    def tokenize(self, X, y, showcase=False):
        # Run, in parallel, the encoding of the dataset X
        chunk_size = ceil(len(X) / self.num_cpus)
        chunks = [X[x:x+chunk_size] for x in range(0, len(X), chunk_size)]
        results = Parallel(n_jobs=self.num_cpus)(
                delayed(self.encode)(X=chunk) for chunk in chunks)

        # Convert the lists into tensors. For each result (each CPU
        # calculations), concatenate (with cat) the input ids together as well
        # as the attention_masks
        input_ids = torch.cat([result[0] for result in results])
        attention_masks = torch.cat([result[1] for result in results])
        labels = torch.tensor(y)

        if showcase:
            # Print sentence 0, now as a list of IDs.
            print('Original: ', X[0])
            print('Token IDs:', input_ids[0])

        return input_ids, attention_masks, labels

    def make_dataloader(self):
        pass


# TESTING ====
bert = Bert()

X = ["Testing BERT one time", "Testing BERT two times", "k", "j", "a", "b", "a", "a"]
y = [0, 1, 1, 1, 1, 1, 1, 1]

assert len(X) == len(y)

bert.tokenize(X, y, True)
