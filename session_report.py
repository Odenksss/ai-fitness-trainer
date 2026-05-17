import csv
import os
from datetime import datetime

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def calculate_rep_score(angle_checks):
    """Return a 0-100 score from weighted angle deviation checks."""
    total_weight = 0
    weighted_penalty = 0

    for check in angle_checks:
        angle = check.get("angle")
        if angle is None:
            continue

        weight = check.get("weight", 1)
        tolerance = max(check.get("tolerance", 1), 1)
        target = check.get("target")

        if target is not None:
            deviation = abs(angle - target)
        elif angle < check.get("min", float("-inf")):
            deviation = check["min"] - angle
        elif angle > check.get("max", float("inf")):
            deviation = angle - check["max"]
        else:
            deviation = 0

        weighted_penalty += min(deviation / tolerance, 1) * weight * 100
        total_weight += weight

    if total_weight == 0:
        return 100.0

    return round(max(0, 100 - (weighted_penalty / total_weight)), 1)


class SessionLogger:
    def __init__(self, exercise_name, unit_label="Rep", output_dir=None):
        self.exercise_name = exercise_name
        self.unit_label = unit_label
        self.scores = []

        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      "session_logs")
        os.makedirs(output_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{exercise_name.lower().replace(' ', '_')}_{stamp}.csv"
        self.csv_path = os.path.join(output_dir, filename)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([unit_label.lower(), "score", "errors", "timestamp"])

    def log(self, unit_number, score, errors):
        if score is None:
            return

        self.scores.append(score)
        errors_text = "; ".join(errors) if errors else "None"
        timestamp = datetime.now().isoformat(timespec="seconds")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([unit_number, score, errors_text, timestamp])

    def show_summary_chart(self):
        if not self.scores:
            print(f"No {self.unit_label.lower()} scores logged.")
            return

        if plt is None:
            print(f"matplotlib is not installed. CSV saved to: {self.csv_path}")
            return

        units = list(range(1, len(self.scores) + 1))
        average = sum(self.scores) / len(self.scores)

        plt.figure(figsize=(8, 4))
        plt.plot(units, self.scores, marker="o", linewidth=2, label="Score")
        plt.axhline(average, linestyle="--", color="orange",
                    label=f"Average: {average:.1f}%")
        plt.title(f"{self.exercise_name} Session Summary")
        plt.xlabel(self.unit_label)
        plt.ylabel("Form Score (%)")
        plt.ylim(0, 105)
        if len(units) <= 20:
            plt.xticks(units)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Session CSV saved to: {self.csv_path}")
