from typing import Any, Dict


class MMStarAccuracyMetric:
    """Metric for MMStar multiple-choice accuracy.

    Matches the predicted text against the ground-truth option letter using two
    patterns: (1) the last character is the option letter; (2) the whole output
    equals "<letter>. <option text>". Tracks overall and per-category accuracy.
    """

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.category = {}  # category -> [correct, total]
        self.l2_category = {}  # l2_category -> [correct, total]

    def _is_correct(self, prediction: str, answer: str, options: dict) -> bool:
        prediction = prediction.strip()
        answer = answer.strip()
        if not prediction:
            return False
        # pattern 1: last letter is the prediction
        if prediction[-1] == answer:
            return True
        # pattern 2: full "A. (option A text)" format
        if options:
            for letter, optiontext in options.items():
                if prediction.strip(".") == letter + ". " + str(optiontext).strip("."):
                    return letter == answer
        return False

    def update(self, pred: Any, gt: dict):
        """Update with prediction string and a sample-like ground-truth dict."""
        item = {**gt, "prediction": pred}
        is_correct = self._is_correct(
            item["prediction"], item["answer"], item.get("options")
        )
        self.total += 1
        if is_correct:
            self.correct += 1

        cat = item.get("category")
        if cat is not None:
            self.category.setdefault(cat, [0, 0])
            self.category[cat][1] += 1
            if is_correct:
                self.category[cat][0] += 1

        l2 = item.get("l2_category")
        if l2 is not None:
            self.l2_category.setdefault(l2, [0, 0])
            self.l2_category[l2][1] += 1
            if is_correct:
                self.l2_category[l2][0] += 1

    def compute(self) -> Dict[str, float]:
        """Compute accuracy metrics.

        Returns:
            Dict[str, float]: overall "accuracy" (percent) plus per-category and
                per-l2-category accuracies, prefixed with "cat/" and "l2/".
        """
        result = {"accuracy": (self.correct / self.total * 100) if self.total else 0.0}
        for cat, (c, t) in sorted(self.category.items()):
            result[f"cat/{cat}"] = (c / t * 100) if t else 0.0
        for l2, (c, t) in sorted(self.l2_category.items()):
            result[f"l2/{l2}"] = (c / t * 100) if t else 0.0
        return result
