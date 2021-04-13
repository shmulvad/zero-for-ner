from collections import defaultdict
import seqeval.metrics
import torch
from tqdm import tqdm


from zero.utils.loader import load_and_cache_examples


def evaluate(args, model, fold, all_entities, output_file=None, return_report=False):
    dataloader, examples, features, processor = load_and_cache_examples(args, fold, all_entities,
                                                                        n_example_per_label=args.n_example_per_label)
    domain_label_map = processor.get_domain_labels()
    all_predictions = defaultdict(dict)

    for batch in tqdm(dataloader, desc="Eval"):
        model.eval()
        inputs = {"source_" + k: v.to(args.device)
                  if k not in ["domains"] else v for k, v in batch.items()
                  if k != "feature_indices"}
        if fold == "train":
            inputs["source_labels"] = None
        with torch.no_grad():
            inputs["eval"] = True
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
        label_list = domain_label_map[example.domain]
        predictions = all_predictions[example_index]
        doc_results = []
        for span, (max_logit, max_index) in predictions.items():
            if max_index != 0:
                doc_results.append((max_logit.item(), span, label_list[max_index.item()]))

        predicted_sequence = ["O"] * len(example.words)
        for _, span, label in sorted(doc_results, key=lambda o: o[0], reverse=True):
            if all([o == "O" for o in predicted_sequence[span[0]: span[1]]]):
                predicted_sequence[span[0]] = "B-" + label
                if span[1] - span[0] > 1:
                    predicted_sequence[span[0] + 1: span[1]] = ["I-" + label] * (span[1] - span[0] - 1)

        final_predictions += predicted_sequence
        final_labels += example.labels

    # convert IOB2 -> IOB1
    print("Num predicted nil {}".format(len([x for x in final_predictions if x == "O"])))
    print("Num actual nil {}".format(len([x for x in final_labels if x == "O"])))
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
    final_labels, final_predictions = [final_labels], [final_predictions]
    report = seqeval.metrics.classification_report(final_labels, final_predictions, digits=4)
    print(report)
    metrics = dict(
        f1=seqeval.metrics.f1_score(final_labels, final_predictions, average="micro"),
        precision=seqeval.metrics.precision_score(final_labels, final_predictions, average="micro"),
        recall=seqeval.metrics.recall_score(final_labels, final_predictions, average="micro"),
    )

    if return_report:
        return metrics, report
    else:
        return metrics
