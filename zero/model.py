from zero.ner.model import LukeForNamedEntityRecognition


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class Zero(nn.Module):
    def __init__(self, args, luke: LukeForNamedEntityRecognition, label_indices, domain_features):
        super(Zero, self).__init__()
        self.args = args
        self.luke = luke
        self.src_domain, self.trg_domain = args.dev_domain, args.test_domain
        self.all_domains = sorted(list(set(self.args.train_domains + [self.args.dev_domain] + [self.args.test_domain])))

        self.label_indices = label_indices

        self.features = torch.FloatTensor(domain_features).to(args.device)
        self.domain2index = {
            domain: i for i, domain in enumerate(self.all_domains)
        }

        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)

        if args.task_name == "zero":
            feature_size = domain_features.shape[1]

            self.fcn_input = args.model_config.hidden_size * 3
            self.fcn_output = feature_size

            self.null_label_feature = nn.Parameter(torch.FloatTensor(1, self.fcn_output)).to(self.args.device)
            nn.init.xavier_uniform_(self.null_label_feature)
            self.null_concept_feature = nn.Parameter(torch.FloatTensor(1, self.fcn_output)).to(self.args.device)
            nn.init.xavier_uniform_(self.null_concept_feature)
        elif args.task_name == "luke":
            self.fcn_input = args.model_config.hidden_size * 3
            self.fcn_output = len(label_indices[args.dev_domain]) + 1  # +1 to account for nil
        else:
            raise ValueError("Unsupported task {}".format(args.task_name))

        self.fcn = nn.Linear(self.fcn_input, self.fcn_output)

    def luke_encode(self, tag, **kwargs):
        encoder_outputs = self.luke.encode(kwargs["{}_word_ids".format(tag)],
                                           kwargs["{}_word_segment_ids".format(tag)],
                                           kwargs["{}_word_attention_mask".format(tag)],
                                           kwargs["{}_entity_ids".format(tag)],
                                           kwargs["{}_entity_position_ids".format(tag)],
                                           kwargs["{}_entity_segment_ids".format(tag)],
                                           kwargs["{}_entity_attention_mask".format(tag)])
        # word_hidden_states = (batch_size, embed_size, max_seq)
        word_hidden_states, entity_hidden_states = encoder_outputs[:2]
        batch_size = word_hidden_states.size(0)
        hidden_size = word_hidden_states.size()[-1]
        return word_hidden_states, entity_hidden_states, batch_size, hidden_size

    def pad_2d(self, mtx_list):
        max_mtx_len = max([len(x) for x in mtx_list])
        hidden_state_list, mask_list = [], []

        for mtx in mtx_list:
            pad_len = max_mtx_len - mtx.shape[0]
            if pad_len == 0:
                hidden_state = mtx
                mask = torch.ones(max_mtx_len)
            else:
                pad_mtx = torch.zeros(pad_len, self.fcn_output).to(self.args.device)
                mask = torch.cat([torch.ones(mtx.shape[0]), torch.zeros(pad_len)], dim=0)
                hidden_state = torch.cat([mtx, pad_mtx], dim=0)
            hidden_state_list.append(hidden_state)
            mask_list.append(mask)

        return torch.stack(hidden_state_list, dim=0), \
            torch.stack(mask_list, dim=0).type(torch.LongTensor).to(self.args.device), \
            max_mtx_len

    def gather_states(self, states, positions, size):
        return torch.gather(states, -2, positions.unsqueeze(-1).expand(-1, -1, size))

    def encode(self, **kwargs):

        word_hidden_states, entity_hidden_states, batch_size, hidden_size = \
            self.luke_encode("source", **kwargs)

        ontology_features = [self.features[self.label_indices[domain]] for domain in kwargs["source_domains"]]

        start_states = self.gather_states(word_hidden_states, kwargs["source_entity_start_positions"], hidden_size)
        end_states = self.gather_states(word_hidden_states, kwargs["source_entity_end_positions"], hidden_size)
        feature_vector = torch.cat([start_states, end_states, entity_hidden_states], dim=2)

        # feature_vector = self.dropout(feature_vector)

        domain_features = [torch.cat([self.null_label_feature, features], dim=0) for features in ontology_features]
        padded_domain_features, domain_mask, max_domain_labels = self.pad_2d(domain_features)

        # feature_vector = (batch_size, #words + #entities, luke_hidden_state * 3 + rgcn_hidden_state * 3)
        feature_vector = self.fcn(feature_vector)
        logits = self.zero_shot_classification(feature_vector, padded_domain_features, domain_mask,
                                               batch_size, max_domain_labels)

        return logits, max_domain_labels, word_hidden_states, entity_hidden_states

    def forward_basic(self, **kwargs):
        logits, max_domain_labels, _, _ = self.encode(**kwargs)
        return logits, max_domain_labels

    def zero_shot_classification(self, feature_vector, padded_domain_features, domain_mask,
                                 batch_size, max_domain_labels):
        domain_vector = padded_domain_features.transpose(-1, -2)
        outputs = torch.matmul(feature_vector, domain_vector)
        domain_mask = domain_mask.unsqueeze(1).expand(batch_size, feature_vector.size(1), max_domain_labels)
        masked_outputs = outputs.masked_fill((1 - domain_mask).bool(), float('-inf'))

        return masked_outputs

    def forward_luke(self, **kwargs):
        word_hidden_states, entity_hidden_states, batch_size, hidden_size = \
            self.luke_encode("source", **kwargs)

        start_states = self.gather_states(word_hidden_states, kwargs["source_entity_start_positions"], hidden_size)
        end_states = self.gather_states(word_hidden_states, kwargs["source_entity_end_positions"], hidden_size)

        feature_vector = torch.cat([start_states, end_states, entity_hidden_states], dim=2)
        logits = self.fcn(feature_vector)

        return logits, self.fcn_output

    def forward(self, **kwargs):
        if self.args.task_name == "zero":
            outputs = self.forward_basic(**kwargs)
        elif self.args.task_name == "luke":
            outputs = self.forward_luke(**kwargs)
        else:
            raise ValueError("Unsupported task {}".format(self.args.task_name))
        logits, output_size = outputs[0], outputs[1]
        if "source_labels" not in kwargs or kwargs["source_labels"] is None:
            return logits

        ner_loss_fn = CrossEntropyLoss(ignore_index=-1)
        ner_loss = ner_loss_fn(logits.view(-1, output_size), kwargs["source_labels"].view(-1))
        total_loss = ner_loss
        reports = {
            "total_loss": total_loss,
            "ner_loss": ner_loss
        }
        return reports
