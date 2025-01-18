import json
import os
import logging
logger = logging.getLogger(__name__)
def run():
    predictions = []
    labels = []
    result_dir = r"/root/autodl-tmp/spt/test_epoch2"
    for file in os.listdir(result_dir):
        path = os.path.join(result_dir, file)
        if path.endswith(".json"):
            with open(path, encoding='ISO-8859-1') as f:
                lines = f.read()
                data = json.loads(lines)
                p = data["test_predictions"].split(",")
                l = data["test_labels"].split(",")
                predictions.extend(p)
                labels.extend(l)

    from sklearn.metrics import recall_score
    recall = recall_score(labels, predictions)
    from sklearn.metrics import precision_score
    precision = precision_score(labels, predictions)
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, predictions)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        if key != 'predictions' and key != 'labels':
            logger.info("  %s = %s", key, str(round(result[key], 4)))


if __name__ == '__main__':
    run()