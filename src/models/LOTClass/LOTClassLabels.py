import torch
from LOTClass import LOTClass

class LOTClassLabels(LOTClass):
    def __init__(self):
        """
        Instantiate three paramaters (label2class, all_label_name_ids and
        all_label_names) which all contain information about the
        given labels
        """
        self.label_name_dict = {i: self.labels[i] for i in range(len(self.labels))}

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
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)
        new_doc = []
        wordpcs = []

        # index starts at 1 due to [CLS] token
        idx = 1

        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            # last index will be [SEP] token:
            if idx >= self.max_len - 1:
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
                                                       max_length=self.max_len,
                                                       padding='max_length',
                                                       return_attention_mask=True,
                                                       truncation=True,
                                                       return_tensors='pt')
            input_ids_with_label_name = encoded['input_ids']
            attention_masks_with_label_name = encoded['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len,
                                                   dtype=torch.long)

            attention_masks_with_label_name = torch.ones(0, self.max_len,
                                                         dtype=torch.long)

            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)

        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx
