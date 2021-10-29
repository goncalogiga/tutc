import torch
from LOTClassModel import LOTClassModel
from multiprocessing import cpu_count
from transformers import BertTokenizer

MODEL_MAX_LEN = 512


class LOTClass():
    def __init__(self, labels, quiet=False):
        """
        Initiates all important paramaters for the classifier.
        """
        # Number of CPUS: usefull for optimized parallelization
        self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1
        self.quiet = quiet
        if not self.quiet:
            print(f"[I] Using {self.num_cpus} CPUs.")

        # Using the pretrained BERT model
        self.pretrained_lm = 'bert-base-uncased'

        # BERT Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm,
                                                       do_lower_case=True)
        # Vocabulary
        self.vocab = self.tokenizer.get_vocab()
        if not self.quiet:
            print("[I] tokenizer.get_vocab() (first 3 items):",
                  dict(list(self.vocab.items())[:3]))

        # Vocabulary size
        self.vocab_size = len(self.vocab)
        if not self.quiet:
            print(f"[I] Total Vocab size: {self.vocab_size}")

        # mask_id : The id of the special token representing a masked token in
        # BERT
        self.mask_id = self.vocab[self.tokenizer.mask_token]

        # Inverse dictionary of the vocabular
        self.inv_vocab = {k: v for v, k in self.vocab.items()}
        if not self.quiet:
            print("[I] inv_vocab (first 3 items):",
                  dict(list(self.inv_vocab.items())[:3]))

        # Fetch label names and intresting label-related objects
        self.label_class = LOTClassLabels(labels)

        # Total number of classes (labels)
        self.num_class = len(self.label_class.label_name_dict)

        # Initiating the LOTClassModel
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   num_labels=self.num_class)

        # Instantiate loss functions
        self.mcp_loss = torch.CrossEntropyLoss()
        self.st_loss = torch.KLDivLoss(reduction='batchmean')

    def encode_data(self, X):
        """
        Encodes the data stored in X with the tokenizer of the model.
        Returns the input ids as well as the attention_mask
        """
        encoded = self.tokenizer.batch_encode_plus(X, add_special_tokens=True,
                                                   max_length=MODEL_MAX_LEN,
                                                   padding='max_length',
                                                   return_attention_mask=True,
                                                   truncation=True,
                                                   return_tensors='pt')
        if not self.quiet:
            print("[I] Example of encoded data:")
            print("[I] Textual input:")
            print(">>>", X[0])
            print("[I] Encoded (dict - truncated 512 elements tensor to 10):")
            print(">>>", {key: encoded[key][0][:10] for key in encoded})

        return encoded["input_ids"], encoded["attention_mask"]

    def encoded2tensors(self, X):
        """
        Converts the encoded() function output into pytorch tensors.
        Returns the converted information into a dictionary.

        This is the function that should be called when encoding
        the data X.
        """
        results = self.encode_data(X)
        # Weird use of results [TODO: check if this is not a mistake]
        input_ids = torch.cat([result[0] for result in results])
        attention_masks = torch.cat([result[1] for result in results])

        data = {"input_ids": input_ids, "attention_masks": attention_masks}

        if not self.quiet:
            print(f"[I : encoded2tensors] data: {data}")

        self.data_dict = {}

        return data


class LOTClassLabels(LOTClass):
    def __init__(self, labels):
        """
        Instantiate three paramaters (label2class, all_label_name_ids and
        all_label_names) which all contain information about the
        given labels
        """
        self.label_name_dict = {i: labels[i] for i in range(len(labels))}

        if not self.quiet:
            print("[I] Label names used for each class:",
                  self.label_name_dict)

        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        self.label2class = {}

        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                assert word not in self.label2class, f"\"{word}\" label used by\
                                                      multiple classes!"

                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

        if not self.quiet:
            print(f"[I] label2class: {self.label2class}")
            print(f"[I] all_label_names_ids: {self.all_label_name_ids}")
            print(f"[I] all_label_names: {self.all_label_names}")

    def label_name_in_doc(self, doc):
        """
        TODO
        """
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(MODEL_MAX_LEN, dtype=torch.long)
        new_doc = []
        wordpcs = []

        # index starts at 1 due to [CLS] token
        idx = 1

        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            # last index will be [SEP] token:
            if idx >= MODEL_MAX_LEN - 1:
                break
            if i == len(doc) - 1 or not doc[i+1].startswith("##"):
                word = ''.join(wordpcs)
                if word in self.label2class:
                    label_idx[idx] = self.label2class[word]
                    # replace label names that are not in tokenizer's
                    # vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        return None

    def encode_data_w_labels(self, docs):
        """
        TODO
        """
        text_with_label = []
        label_name_idx = []

        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))

        if len(text_with_label) > 0:
            encoded = self.tokenizer.batch_encode_plus(text_with_label,
                                                       add_special_tokens=True,
                                                       max_length=MODEL_MAX_LEN,
                                                       padding='max_length',
                                                       return_attention_mask=True,
                                                       truncation=True,
                                                       return_tensors='pt')
            input_ids_with_label_name = encoded['input_ids']
            attention_masks_with_label_name = encoded['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, MODEL_MAX_LEN,
                                                   dtype=torch.long)

            attention_masks_with_label_name = torch.ones(0, MODEL_MAX_LEN,
                                                         dtype=torch.long)

            label_name_idx = torch.ones(0, MODEL_MAX_LEN, dtype=torch.long)

        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

l= LOTClass([["Light"], ["Dark"]])
l.encoded2tensors(["ola que tal", "oi"])
