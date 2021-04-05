import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils_io import *
from zero.ner.utils import NERProcessor, convert_examples_to_features
import numpy as np

logger = logging.getLogger(__name__)

WORD_EMBED_SIZE = 300


def load_domain_paths(args, domain):
    _entity_path, _relation_path, _feature_path = os.path.join(args.conceptnet_dir, "{}.nodes.tsv".format(domain)), \
                                                  os.path.join(args.conceptnet_dir,
                                                               "{}.relations.tsv".format(domain)), \
                                                  os.path.join(args.conceptnet_dir,
                                                               "{}.embedding.pickle".format(domain))
    return _entity_path, _relation_path, _feature_path


def load_unique_relations(args, domains):
    unique_relations = set()

    for domain in domains:
        _entity_path, relation_path, _feature_path = load_domain_paths(args, domain)
        relation_df = pd.read_csv(relation_path, delimiter="\t")
        relations = relation_df["relation"].tolist()
        unique_relations.update(relations)

    return sorted(list(unique_relations))


def load_word_embedding(embedding_path, embed_size, all_entities):
    word_embeddings = np.zeros((len(all_entities), embed_size)).astype(np.float32)
    init_embedding = pd.read_table(embedding_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
    for i, word in enumerate(all_entities):
        try:
            word_embeddings[i] = init_embedding.loc[word].values
        except KeyError:
            word_embeddings[i] = np.random.normal(size=(embed_size,))

    return word_embeddings


def load_domain_features(args, src_domain=None, trg_domain=None, data_dir=None,
                         embed=None):
    data_dir = args.data_dir if args is not None else data_dir

    src_entities = load_text_as_list(os.path.join(data_dir, "{}.entities.txt".format(src_domain)))
    trg_entities = load_text_as_list(os.path.join(data_dir, "{}.entities.txt".format(trg_domain)))
    src_labels = load_text_as_list(os.path.join(data_dir, "{}.labels.txt".format(src_domain)))
    trg_labels = load_text_as_list(os.path.join(data_dir, "{}.labels.txt".format(trg_domain)))
    src_labels = [x.replace(" ", "_") for x in src_labels]
    trg_labels = [x.replace(" ", "_") for x in trg_labels]

    all_entities = sorted(list(set(src_entities + trg_entities + src_labels + trg_labels)))
    embedding_path = os.path.join(data_dir, f"embeddings-{embed}.txt")
    word_embedding = load_word_embedding(embedding_path, WORD_EMBED_SIZE, all_entities)
    src_indices, trg_indices = [all_entities.index(x) for x in src_labels], [all_entities.index(x) for x in trg_labels]
    domain_indices = {
        src_domain: src_indices,
        trg_domain: trg_indices
    }
    return domain_indices, word_embedding, all_entities


def load_domain_pair_data(args=None, src_domain=None, trg_domain=None, device=None, concept_graph_dir=None):
    src_domain, trg_domain = (args.dev_domain, args.test_domain) if args is not None else (src_domain, trg_domain)
    device = args.device if args is not None else device
    concept_graph_dir = args.concept_graph_dir if args is not None else concept_graph_dir
    sorted_domains = sorted([src_domain, trg_domain])
    domain_tag = "{}_{}".format(*sorted_domains)

    # source and target domain ontology will be similar
    domain_ontologies, domain_label_indices, domain_features = {}, {}, {}

    domain_ontology, domain1_entity_indices, domain2_entity_indices = load_from_pickle(
        os.path.join(concept_graph_dir, domain_tag, "data.pickle"))
    src_entity_indices = domain1_entity_indices if src_domain == sorted_domains[0] else domain2_entity_indices
    trg_entity_indices = domain2_entity_indices if trg_domain == sorted_domains[1] else domain1_entity_indices
    domain_ontology = domain_ontology.to(device)
    domain_feature = load_from_pickle(os.path.join(concept_graph_dir, domain_tag, "domain_embeddings.pickle"))

    domain_ontologies[src_domain], domain_label_indices[src_domain], domain_features[src_domain] = \
        domain_ontology, src_entity_indices, domain_feature
    domain_ontologies[trg_domain], domain_label_indices[trg_domain], domain_features[trg_domain] = \
        domain_ontology, trg_entity_indices, domain_feature

    return domain_ontologies, domain_label_indices, domain_features


def load_domain_multi_data(args):
    all_domains = args.train_domains + [args.dev_domain] + [args.test_domain]
    domain_ontologies, domain_label_indices, domain_features = {}, {}, {}

    for domain in all_domains:
        domain_ontology, domain_labels = load_from_pickle(os.path.join("data", "ner", domain, "data.pickle"))
        domain_feature = load_from_pickle(os.path.join("data", "ner", domain, "domain_embeddings.pickle"))
        domain_ontology = domain_ontology.to(args.device)
        domain_ontologies[domain], domain_label_indices[domain], domain_features[domain] = \
            domain_ontology, domain_labels, domain_feature

    return domain_ontologies, domain_label_indices, domain_features


def get_saved_paths(args, tag=None):
    tag = ".{}".format(tag) if tag is not None else ""
    dozen_path = os.path.join(args.output_dir, args.exp_name, "dozen_model{}.bin".format(tag))
    luke_path = os.path.join(args.output_dir, args.exp_name, "luke_model{}.bin".format(tag))
    rgcn_path = os.path.join(args.output_dir, args.exp_name, "rgcn_model{}.bin".format(tag))
    return dozen_path, luke_path, rgcn_path


def load_and_cache_examples(args, fold, inter_domain_entities, random_sampling=True):
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = NERProcessor(os.path.join(args.data_dir, "ner"),
                             args.train_domains, args.dev_domain, args.test_domain)
    if fold == "train":
        examples = processor.get_train_examples()
    elif fold == "dev":
        examples = processor.get_dev_examples()
    else:
        examples = processor.get_test_examples()

    if fold == "train" and args.train_on_dev_set:
        examples += processor.get_dev_examples()

    domain_label_map = processor.get_domain_labels()

    bert_model_name = args.model_config.bert_model_name

    cache_file = os.path.join(
        args.data_dir,
        "cached_"
        + "_".join(
            (
                bert_model_name.split("-")[0],
                str(args.max_seq_length),
                str(args.max_entity_length),
                str(args.max_mention_length),
                str(args.train_on_dev_set),
                "train_{}".format("_".join(args.train_domains)),
                "dev_{}".format(args.dev_domain),
                "test_{}".format(args.test_domain),
                fold,
            )
        )
        + ".pkl",
    )
    inter_domain_map = {entity: i+1 for i, entity in enumerate(inter_domain_entities)}

    if os.path.exists(cache_file):
        logger.info("Loading features from the cached file %s", cache_file)
        features = torch.load(cache_file)
    else:
        logger.info("Creating features from the dataset...")

        features = convert_examples_to_features(
            examples, inter_domain_map, domain_label_map, args.tokenizer, args.max_seq_length, args.max_entity_length, args.max_mention_length
        )

        if args.local_rank in (-1, 0):
            torch.save(features, cache_file)

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value):
            if target == "concept_ratios":
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.float) for o in batch]
            elif isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        ret = dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_start_positions=create_padded_sequence("entity_start_positions", 0),
            entity_end_positions=create_padded_sequence("entity_end_positions", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            concepts=create_padded_sequence("concepts", 0),
        )
        if args.no_entity_feature:
            ret["entity_ids"].fill_(0)
            ret["entity_attention_mask"].fill_(0)

        if fold == "train":
            ret["labels"] = create_padded_sequence("labels", -1)
        ret["feature_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)

        ret["domains"] = [o[1].domain for o in batch]
        return ret

    if fold == "train":
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
    elif random_sampling:
        sampler = RandomSampler(features)
    else:
        sampler = None

    dataloader = DataLoader(
        list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
    )

    return dataloader, examples, features, processor
