import logging
from argparse import Namespace
from datetime import datetime

import click
import torch

from luke.utils.entity_vocab import MASK_TOKEN

from zero.model import Zero
from zero.ner.model import LukeForNamedEntityRecognition
from zero.utils import set_seed
from zero.utils.trainer import Trainer, trainer_args

from zero.utils.evaluator import evaluate
from zero.utils.loader import load_domain_features, load_and_cache_examples, \
    get_saved_paths
from utils_io import *

logger = logging.getLogger(__name__)


def get_exp_name(task_name):
    return "{}-{}".format(task_name, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))


@click.group(name="ner")
def cli():
    pass


@cli.command()
@click.option("--log-dir", default="runs", type=click.Path())
@click.option("--task-name", default="zero")
@click.option("--data-dir", default="data", type=click.Path(exists=True))
@click.option("--train-domains", default="science")
@click.option("--dev-domain", default="science")
@click.option("--test-domain", default="music")
@click.option("--max-seq-length", default=512)
@click.option("--max-entity-length", default=128)
@click.option("--max-mention-length", default=25)
@click.option("--no-word-feature", is_flag=True)
@click.option("--no-entity-feature", is_flag=True)
@click.option("--do-train/--no-train", default=True)
@click.option("--train-batch-size", default=2)
@click.option("--num-train-epochs", default=5.0)
@click.option("--do-eval/--no-eval", default=True)
@click.option("--eval-batch-size", default=32)
@click.option("--train-on-dev-set", is_flag=True)
@click.option("--seed", default=35)
@trainer_args
@click.pass_obj
def run(common_args, **task_args):
    common_args.update(task_args)
    args = Namespace(**common_args)
    args.exp_name = get_exp_name(args.task_name)

    args.train_domains = args.train_domains.split(",")
    domain_label_indices, domain_features, all_entities = \
        load_domain_features(args, src_domain=args.dev_domain, trg_domain=args.test_domain, data_dir=args.data_dir)

    set_seed(args.seed)

    args.experiment.log_parameters({p.name: getattr(args, p.name) for p in run.params})

    args.model_config.entity_vocab_size = 2
    entity_emb = args.model_weights["entity_embeddings.entity_embeddings.weight"]
    mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)
    args.model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])

    train_source_dataloader, _, _, processor = load_and_cache_examples(args, "train", all_entities)
    #train_target_dataloader, _, _, processor = load_and_cache_examples(args, "train", all_entities)
    train_target_dataloader, _, _, _ = load_and_cache_examples(args, "test", all_entities)
    
    #train_dataloader, _, _, processor = load_examples(args, "train")

    results = {}

    if args.do_train:
        pretrained_luke = LukeForNamedEntityRecognition(args, len(processor.get_labels()))
        pretrained_luke.load_state_dict(args.model_weights, strict=False)

        dozen = Zero(args, pretrained_luke, domain_label_indices, domain_features)
        dozen.to(args.device)

        num_train_steps_per_epoch = len(train_source_dataloader) // args.gradient_accumulation_steps
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

        trainer = Trainer(args, model=dozen, all_entities=all_entities,
                          train_source_dataloader=train_source_dataloader,
                          train_target_dataloader=train_target_dataloader,
                          num_train_steps=num_train_steps)

        trainer.train()

        if args.local_rank in (0, -1):
            os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
            logger.info("Saving the model checkpoint to %s", args.output_dir)
            dozen_path, luke_path, rgcn_path = get_saved_paths(args, tag="last")
            torch.save(pretrained_luke.state_dict(), luke_path)
            torch.save(dozen.state_dict(), dozen_path)

    if args.local_rank not in (0, -1):
        return {}

    torch.cuda.empty_cache()

    if args.do_eval:
        dozen_path, luke_path, rgcn_path = get_saved_paths(args, tag="latest")

        luke = LukeForNamedEntityRecognition(args, len(processor.get_labels()))
        luke.load_state_dict(torch.load(luke_path, map_location="cpu"))
        luke.to(args.device)

        zero = Zero(args, luke, domain_label_indices, domain_features)
        zero.load_state_dict(torch.load(dozen_path, map_location="cpu"))
        zero.to(args.device)

        train_output_file = os.path.join(args.output_dir, "train_predictions.txt")
        dev_output_file = os.path.join(args.output_dir, "dev_predictions.txt")
        test_output_file = os.path.join(args.output_dir, "test_predictions.txt")
        results.update({f"train_{k}": v for k, v in evaluate(args, zero, "train", train_output_file).items()})
        results.update({f"dev_{k}": v for k, v in evaluate(args, zero, "dev", dev_output_file).items()})
        results.update({f"test_{k}": v for k, v in evaluate(args, zero, "test", test_output_file).items()})

    logger.info("Results: %s", json.dumps(results, indent=2, sort_keys=True))
    args.experiment.log_metrics(results)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)

    return results

def evaluate(args, model, fold, output_file=None):
    dataloader, examples, features, processor = load_examples(args, fold)
    label_list = processor.get_labels()
    all_predictions = defaultdict(dict)

    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "feature_indices"}
        with torch.no_grad():
            logits = model(**inputs)

        for i, feature_index in enumerate(batch["feature_indices"]):
            feature = features[feature_index.item()]
            for j, span in enumerate(feature.original_entity_spans):
                if span is not None:
                    all_predictions[feature.example_index][span] = logits[i, j].detach().cpu().max(dim=0)

    assert len(all_predictions) == len(examples)

    final_labels = []
    final_predictions = []

    for example_index, example in enumerate(examples):
        predictions = all_predictions[example_index]
        doc_results = []
        for span, (max_logit, max_index) in predictions.items():
            if max_index != 0:
                doc_results.append((max_logit.item(), span, label_list[max_index.item()]))

        predicted_sequence = ["O"] * len(example.words)
        for _, span, label in sorted(doc_results, key=lambda o: o[0], reverse=True):
            if all([o == "O" for o in predicted_sequence[span[0] : span[1]]]):
                predicted_sequence[span[0]] = "B-" + label
                if span[1] - span[0] > 1:
                    predicted_sequence[span[0] + 1 : span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

        final_predictions += predicted_sequence
        final_labels += example.labels

    # convert IOB2 -> IOB1
    prev_type = None
    for n, label in enumerate(final_predictions):
        if label[0] == "B" and label[2:] != prev_type:
            final_predictions[n] = "I" + label[1:]
        prev_type = label[2:]

    if output_file:
        all_words = [w for e in examples for w in e.words]
        with open(output_file, "w") as f:
            for item in zip(all_words, final_labels, final_predictions):
                f.write(" ".join(item) + "\n")

    assert len(final_predictions) == len(final_labels)
    print("The number of labels:", len(final_labels))
    print(seqeval.metrics.classification_report(final_labels, final_predictions, digits=4))

    return dict(
        f1=seqeval.metrics.f1_score(final_labels, final_predictions),
        precision=seqeval.metrics.precision_score(final_labels, final_predictions),
        recall=seqeval.metrics.recall_score(final_labels, final_predictions),
    )


def load_examples(args, fold): # from original luke
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = CoNLLProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    if fold == "train" and args.train_on_dev_set:
        examples += processor.get_dev_examples(args.data_dir)

    label_list = processor.get_labels()

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples, label_list, args.tokenizer, args.max_seq_length, args.max_entity_length, args.max_mention_length
    )

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(target, padding_value):
            if isinstance(target, str):
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
        )
        if args.no_entity_feature:
            ret["entity_ids"].fill_(0)
            ret["entity_attention_mask"].fill_(0)

        if fold == "train":
            ret["labels"] = create_padded_sequence("labels", -1)
        else:
            ret["feature_indices"] = torch.tensor([o[0] for o in batch], dtype=torch.long)

        return ret

    if fold == "train":
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(
            list(enumerate(features)), sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn
        )
    else:
        dataloader = DataLoader(list(enumerate(features)), batch_size=args.eval_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, processor