from math import ceil
from multiprocessing import cpu_count
from transformers import BertTokenizer
from joblib import Parallel, delayed
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import torch
from types import FunctionType
from trainer import train_model


class Bert():
    def __init__(self, labelizer=None, model='bert-base-uncased', max_length=64,
                 quiet=False):

        self.quiet = quiet
        self.max_length = max_length
        self.model_name = model

        if labelizer is not None and not isinstance(labelizer, FunctionType):
            raise TypeError("labelizer should be a callable function.")
        self.labelizer = labelizer

        self.check_hardware()

        # Load the BERT tokenizer.
        if not quiet:
            print('Loading the BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained(model,
                                                       do_lower_case=True)

        # Let's not load the model right away to save some memory
        self.model = None

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
        `batch_encode_plus` will:
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
        """
        Applies the BERT tokenizer to the data in order to retrieve the token
        IDs associated to each input as well as the attention_masks. The actual
        tokenization is done in the function `encode` but this function divides
        the data in order for each CPU to tokenize batches of data and also
        transforms the output of the tokenizer into tensors.
        """
        # Run, in parallel, the encoding of the dataset X
        chunk_size = ceil(len(X) / self.num_cpus)

        chunks = [X[x:x+chunk_size] for x in range(0, len(X), chunk_size)]

        try:
            results = Parallel(n_jobs=self.num_cpus)(
                delayed(self.encode)(X=chunk) for chunk in chunks)
        except:
            print("Something went wrong with the parallelization...")
            results = [self.encode(X)]

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

    def make_dataloader(self, input_ids, attention_masks, labels,
                        batch_size=32):
        """
        Create an iterator for our dataset using the torch DataLoader class.
        This helps save on memory during training because, unlike a for loop,
        with an iterator the entire dataset does not need to be loaded into
        memory
        """
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, test_dataset = random_split(dataset,
                                                   [train_size, test_size])

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=batch_size)

        # For validation the order doesn't matter, so we'll just read them
        # sequentially.
        test_dataloader = DataLoader(test_dataset,
                                     sampler=SequentialSampler(test_dataset),
                                     batch_size=batch_size)

        return train_dataloader, test_dataloader

    def load_model(self, num_labels):
        if not self.quiet:
            print('Loading the BERT model...')

        # Load BertForSequenceClassification, the pretrained BERT model with a
        # single linear classification layer on top.
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False)

        if self.gpu:
            self.model.cuda()

    def get_optimizer(self):
        """
        Note: AdamW is a class from the huggingface library (as opposed to
        pytorch) I believe the 'W' stands for 'Weight Decay fix"
        """
        return AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

    def get_scheduler(self, optimizer, total_steps):
        # Create the learning rate scheduler.
        return get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=0,
                                               num_training_steps=total_steps)


    def transform(self, X, y=None):
        return X

    def fit(self, X, y=None, path_to_save_stats=None, path_to_save_model=None,
            epochs=4, seed_val=42):

        if self.labelizer is not None:
            if y is not None:
                raise Exception("Argument y of function transform is uncompatible\
                                with the given labelizer")

            X, y = self.labelizer(X)

        assert len(X) == len(y), f"len(X) != len(y)\
                                  [len(X) = {len(X)}, len(y) = {len(y)}]"

        input_ids, attention_masks, labels = self.tokenize(X, y)

        train_dl, test_dl = self.make_dataloader(input_ids,
                                                 attention_masks,
                                                 labels)

        # Total number of training steps is [number of batches] x [number of
        # epochs].  (Note that this is not the same as the number of training
        # samples).
        total_steps = len(train_dl) * epochs

        self.load_model(len(y))
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer, total_steps)

        train_model(
            self.model,
            train_dl,
            test_dl,
            optimizer,
            scheduler,
            epochs,
            self.device,
            path_to_save_stats,
            path_to_save_model,
            seed_val
        )





# TESTING ====
bert = Bert()
