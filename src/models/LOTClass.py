import torch
from LOTClassModel import LOTClassModel
from multiprocessing import cpu_count
from transformers import BertTokenizer

MODEL_MAX_LEN = 512


class LOTClass():
    def __init__(self, labels, quiet=False):
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
        self.read_label_names(labels)

        # Total number of classes (labels)
        self.num_class = len(self.label_name_dict)

        # Initiating the LOTClassModel
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   num_labels=self.num_class)

    def read_label_names(self, labels):
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

    def encode_data(self, X, quiet=False):
        encoded_dict = self.tokenizer.batch_encode_plus(X, add_special_tokens=True,
                                                        max_length=MODEL_MAX_LEN,
                                                        padding='max_length',
                                                        return_attention_mask=True,
                                                        truncation=True,
                                                        return_tensors='pt')
        if not self.quiet and not quiet:
            print("[I] Example of encoded data:")
            print("[I] Textual input:")
            print(">>>", X[0])
            print("[I] Encoded result (dict - truncated 512 elements tensor to 10):")
            print(">>>", {key: encoded_dict[key][0][:10] for key in encoded_dict})

        return encoded_dict["input_ids"], encoded_dict["attention_mask"]

    def encoded2tensors(self, X):
        results = self.encode_data(X)
        input_ids = torch.cat([result[0] for result in results])
        attention_masks = torch.cat([result[1] for result in results])
        data = {"input_ids": input_ids, "attention_masks": attention_masks}

        if not self.quiet:
            print(f"[I : encoded2tensors] data: {data}")

        return data


l= LOTClass([["Light"], ["Dark"]])
l.encoded2tensors(["ola que tal", "oi"])
