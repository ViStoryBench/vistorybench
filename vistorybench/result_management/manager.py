import os
import json
import argparse
import hashlib
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def blue_print(text, bright=True):
    color_code = "\033[94m" if bright else "\033[34m"
    print(f"{color_code}{text}\033[0m")


def green_print(text):
    print(f"\033[92m{text}\033[0m")


class ResultManager:
    """
    Manage and organize evaluation results for ViStoryBench.

    Unified structure (see docs/result_schema.md):
    data/bench_results/{method}/{mode}/{language}/{timestamp}/
      - metadata.json
      - manifest.json
      - summary.json
      - metrics/{metric}/
          - items.jsonl
          - story_results.json
          - scores.json
          - details/...  (optional)
          - meta.json     (optional)
    """

    def __init__(
        self,
        method_name: str,
        mode: Optional[str] = None,
        language: Optional[str] = None,
        dataset_name: str = "ViStory",
        timestamp: Optional[str] = None,
        base_path: str = "data/bench_results",
    ):
        """
        Initialize a ResultManager instance.

        Args:
            method_name: The name of the method (model) being evaluated.
            mode: Method mode (e.g. base, SD3); aligns with README path layout.
            language: Dataset language (e.g. en, ch); aligns with README path layout.
            dataset_name: Dataset identifier (default ViStory).
            timestamp: Evaluation timestamp. If None, latest is selected or created.
            base_path: Base directory where all results are stored.
        """
        self.method_name = method_name
        self.mode = mode or "default"
        self.language = language or "en"
        self.dataset_name = dataset_name
        self.base_path = base_path

        # Resolve timestamp
        if timestamp:
            self.timestamp = timestamp
        else:
            # Search for latest under {base}/{method}/{mode}/{language}
            try:
                timestamps = self.list_method_timestamps(
                    method_name=self.method_name,
                    mode=self.mode,
                    language=self.language,
                    base_path=self.base_path,
                )
                if not timestamps:
                    print(
                        f"Warning: No timestamps found for {self.method_name}/{self.mode}/{self.language}. Creating a new one."
                    )
                    self.timestamp = self.create_timestamp()
                else:
                    self.timestamp = sorted(timestamps)[-1]
                    print(f"No timestamp provided, selected the latest: {self.timestamp}")
            except FileNotFoundError:
                print(
                    f"Warning: Result directory for {self.method_name}/{self.mode}/{self.language} not found. Creating a new one."
                )
                self.timestamp = self.create_timestamp()

        # Build root path
        self.result_path = os.path.join(
            self.base_path, self.method_name, self.mode, self.language, self.timestamp
        )
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "metrics"), exist_ok=True)

        # Initialize minimal metadata if absent
        self._write_metadata_if_absent()

    # ----------------------------
    # Path helpers
    # ----------------------------
    def _metrics_root(self) -> str:
        return os.path.join(self.result_path, "metrics")

    def get_metric_dir(self, metric_name: str) -> str:
        p = os.path.join(self._metrics_root(), metric_name)
        os.makedirs(p, exist_ok=True)
        return p

    def _items_path(self, metric_name: str) -> str:
        return os.path.join(self.get_metric_dir(metric_name), "items.jsonl")

    def _story_results_path(self, metric_name: str) -> str:
        return os.path.join(self.get_metric_dir(metric_name), "story_results.json")

    def _scores_path(self, metric_name: str) -> str:
        return os.path.join(self.get_metric_dir(metric_name), "scores.json")

    def _manifest_path(self) -> str:
        return os.path.join(self.result_path, "manifest.json")

    def _metadata_path(self) -> str:
        return os.path.join(self.result_path, "metadata.json")

    # ----------------------------
    # Run metadata and manifest
    # ----------------------------
    def _write_metadata_if_absent(self) -> None:
        meta_path = self._metadata_path()
        if os.path.exists(meta_path):
            return
        metadata = {
            "run": {
                "method": self.method_name,
                "mode": self.mode,
                "language": self.language,
                "dataset_name": self.dataset_name,
                "timestamp": self.timestamp,
            }
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def load_manifest(self) -> Dict[str, Any]:
        path = self._manifest_path()
        if not os.path.exists(path):
            return {"version": 1, "resume_default": "verify", "metrics": {}}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        path = self._manifest_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
        os.replace(tmp, path)

    def update_manifest(
        self,
        metric_name: str,
        story_id: str,
        record: Dict[str, Any],
    ) -> None:
        manifest = self.load_manifest()
        metrics_map = manifest.setdefault("metrics", {})
        metric_map = metrics_map.setdefault(metric_name, {})
        metric_map[str(story_id)] = record
        self.save_manifest(manifest)

    def has_story_result(
        self,
        metric_name: str,
        story_id: str,
        expected_version: Optional[str] = None,
        outputs_hash: Optional[str] = None,
        resume_policy: str = "verify",
    ) -> bool:
        """
        Decide whether to skip re-evaluation for a given metric×story.

        Policies:
          - verify (default): skip only if status==complete AND (expected_version is None or matches) AND (outputs_hash matches if provided)
          - skip: skip if status==complete (ignores version/hash)
          - overwrite: never skip
        """
        if resume_policy == "overwrite":
            return False

        manifest = self.load_manifest()
        metric_map = manifest.get("metrics", {}).get(metric_name, {})
        rec = metric_map.get(str(story_id))
        if not rec:
            return False

        status = rec.get("status")
        if status != "complete":
            return False

        if resume_policy == "skip":
            return True

        # verify
        if expected_version is not None:
            if rec.get("version") != expected_version:
                return False
        if outputs_hash is not None:
            if rec.get("outputs_hash") != outputs_hash:
                return False
        return True

    # ----------------------------
    # IO utils
    # ----------------------------
    @staticmethod
    def create_timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def stable_sha1(text: str) -> str:
        h = hashlib.sha1()
        h.update(text.encode("utf-8"))
        return f"sha1:{h.hexdigest()}"

    def append_items(self, metric_name: str, items: List[Dict[str, Any]]) -> None:
        """
        Append item-level records as JSONL.
        """
        path = self._items_path(metric_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ----------------------------
    # Save/load story and dataset results
    # ----------------------------
    def save_story_result(
        self,
        metric_name: str,
        story_id: str,
        data: Dict[str, Any],
        metric_version: Optional[str] = None,
        outputs_hash: Optional[str] = None,
        duration_ms: Optional[int] = None,
        status: str = "complete",
    ):
        """
        Save evaluation result for a single story under metrics/{metric}/story_results.json

        Also updates manifest.json entry for this story.
        """
        # Wrap into unified structure if needed
        wrapped = {
            "run": {
                "method": self.method_name,
                "mode": self.mode,
                "language": self.language,
                "dataset": self.dataset_name,
                "timestamp": self.timestamp,
            },
            "metric": {"name": metric_name},
            "scope": {"level": "story", "story_id": str(story_id)},
            "status": status,
        }
        if metric_version:
            wrapped["metric"]["version"] = metric_version

        # If data already contains "metrics", keep; else wrap as metrics
        if isinstance(data, dict) and "metrics" in data:
            wrapped.update(data)
        else:
            wrapped["metrics"] = data

        if duration_ms is not None:
            wrapped["duration_ms"] = duration_ms

        # Load existing story_results
        sr_path = self._story_results_path(metric_name)
        if os.path.exists(sr_path):
            try:
                with open(sr_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
        else:
            all_results = {}

        all_results[str(story_id)] = wrapped

        tmp = sr_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)
        os.replace(tmp, sr_path)

        # Update manifest
        rec = {
            "status": status,
            "version": metric_version,
            "outputs_hash": outputs_hash,
            "last_updated": datetime.utcnow().isoformat() + "Z",
        }
        if duration_ms is not None:
            rec["duration_ms"] = duration_ms
        self.update_manifest(metric_name, str(story_id), rec)

    def save_dataset_metric(self, metric_name: str, scores: Dict[str, Any]) -> None:
        """
        Save dataset-level results for a metric to metrics/{metric}/scores.json
        """
        p = self._scores_path(metric_name)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4)
        os.replace(tmp, p)

    def save_summary(self, summary_data: Dict[str, Any]):
        """
        Save cross-metric dataset-level summary to summary.json
        """
        summary_file = os.path.join(self.result_path, "summary.json")
        tmp = summary_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        os.replace(tmp, summary_file)

    def load_summary(self) -> Dict[str, Any]:
        summary_file = os.path.join(self.result_path, "summary.json")
        if not os.path.exists(summary_file):
            return {}
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_all_results(self) -> Dict[str, Any]:
        """
        Load dataset-level scores for all metrics from metrics/*/scores.json
        """
        all_results: Dict[str, Any] = {}
        metrics_root = self._metrics_root()
        if not os.path.isdir(metrics_root):
            return all_results

        for metric_name in os.listdir(metrics_root):
            metric_dir = os.path.join(metrics_root, metric_name)
            if os.path.isdir(metric_dir):
                scores_file = os.path.join(metric_dir, "scores.json")
                if os.path.exists(scores_file):
                    with open(scores_file, "r", encoding="utf-8") as f:
                        all_results[metric_name] = json.load(f)
        return all_results

    def load_metric_result(
        self, metric_name: str, result_type: str = "scores"
    ) -> Optional[Union[Dict[str, Any], pd.DataFrame]]:
        """
        Load results for a single metric:
         - 'scores'   -> dataset-level scores.json
         - 'story'    -> story_results.json
         - 'detailed' -> {metric}_detailed.xlsx (if present)
        """
        metric_path = self.get_metric_dir(metric_name)
        if not os.path.isdir(metric_path):
            print(f"Warning: Metric directory not found: {metric_path}")
            return None

        if result_type == "scores":
            scores_file = os.path.join(metric_path, "scores.json")
            if not os.path.exists(scores_file):
                return None
            with open(scores_file, "r", encoding="utf-8") as f:
                return json.load(f)

        elif result_type == "story":
            story_file = os.path.join(metric_path, "story_results.json")
            if not os.path.exists(story_file):
                return None
            with open(story_file, "r", encoding="utf-8") as f:
                return json.load(f)

        elif result_type == "detailed":
            if pd is None:
                print("Warning: 'pandas' is not installed. Cannot load detailed Excel data.")
                return None

            detailed_file = os.path.join(metric_path, f"{metric_name}_detailed.xlsx")
            if not os.path.exists(detailed_file):
                print(f"Warning: Detailed results file not found: {detailed_file}")
                return None

            try:
                return pd.read_excel(detailed_file)
            except Exception as e:
                print(f"Error: Failed to load detailed data: {e}")
                return None

        else:
            print(f"Error: Invalid result type '{result_type}'. Choose 'scores', 'story', or 'detailed'.")
            return None

    @classmethod
    def list_available_methods(cls, base_path: str = "data/bench_results") -> List[str]:
        if not os.path.isdir(base_path):
            return []
        return [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    @classmethod
    def list_method_timestamps(
        cls, method_name: str, mode: str, language: str, base_path: str = "data/bench_results"
    ) -> List[str]:
        run_root = os.path.join(base_path, method_name, mode, language)
        if not os.path.isdir(run_root):
            return []
        return [name for name in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, name))]

    # ----------------------------
    # Aggregations
    # ----------------------------
    def _aggregate_story_metrics_mean(self, story_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Aggregate story_results.json -> mean for each numeric key under 'metrics'
        """
        # Collect keys
        numeric_keys: Dict[str, List[float]] = {}
        for _sid, obj in story_results.items():
            metrics = obj.get("metrics") if isinstance(obj, dict) else None
            if not isinstance(metrics, dict):
                # If stored raw dict, try flatten numeric values at top-level
                metrics = obj if isinstance(obj, dict) else {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    numeric_keys.setdefault(k, []).append(float(v))
        # Mean per key
        return {k: (sum(vals) / len(vals) if vals else 0.0) for k, vals in numeric_keys.items()}

    def compute_and_save_summary(self, metrics: Optional[List[str]] = None) -> None:
        """
        Compute cross-metric dataset-level summary.json.

        For each metric:
          - Prefer metrics/{metric}/scores.json if present
          - Else derive from metrics/{metric}/story_results.json by mean on numeric metrics
        """
        metrics_root = self._metrics_root()
        if not os.path.isdir(metrics_root):
            self.save_summary(
                {
                    "run": {
                        "method": self.method_name,
                        "mode": self.mode,
                        "language": self.language,
                        "dataset": self.dataset_name,
                        "timestamp": self.timestamp,
                    },
                    "metrics": {},
                }
            )
            return

        result: Dict[str, Any] = {
            "run": {
                "method": self.method_name,
                "mode": self.mode,
                "language": self.language,
                "dataset": self.dataset_name,
                "timestamp": self.timestamp,
            },
            "metrics": {},
        }

        metric_names = metrics or [
            name for name in os.listdir(metrics_root) if os.path.isdir(os.path.join(metrics_root, name))
        ]

        for metric in metric_names:
            mdir = self.get_metric_dir(metric)
            scores_file = os.path.join(mdir, "scores.json")
            story_file = os.path.join(mdir, "story_results.json")

            if os.path.exists(scores_file):
                with open(scores_file, "r", encoding="utf-8") as f:
                    result["metrics"][metric] = json.load(f)
            elif os.path.exists(story_file):
                with open(story_file, "r", encoding="utf-8") as f:
                    story_results = json.load(f)
                mean_map = self._aggregate_story_metrics_mean(story_results)
                result["metrics"][metric] = {"metrics": mean_map}
            else:
                result["metrics"][metric] = {"metrics": {}}

        self.save_summary(result)

    # ----------------------------
    # Legacy analysis helpers (kept for convenience)
    # ----------------------------
    def analyze_total_average(self):
        """
        Load scores for all metrics, compute and display the average for each metric.
        """
        blue_print(
            f"\nCalculating total average scores for method='{self.method_name}', mode='{self.mode}', "
            f"language='{self.language}' (timestamp: {self.timestamp})..."
        )

        all_results = self.get_all_results()
        if not all_results:
            print(f"Error: No results found for analysis in {self.result_path}.")
            return

        summary = {}
        blue_print("\n--- Average scores per metric ---", bright=True)

        for metric, data in all_results.items():
            scores = []

            def extract_scores(d):
                if isinstance(d, dict):
                    for value in d.values():
                        if isinstance(value, (int, float)):
                            scores.append(value)
                        elif isinstance(value, dict):
                            extract_scores(value)

            extract_scores(data)

            if scores:
                average_score = mean(scores)
                summary[metric] = average_score
                print(f"{metric.capitalize():<20}: {average_score:.4f}")
            else:
                summary[metric] = "N/A"
                print(f"{metric.capitalize():<20}: N/A (no valid scores)")

        green_print("\nTotal average analysis complete.")

    def analyze_cids_by_char_action(self):
        """
        Load detailed CIDS results and compute average CIDS scores by character and action.
        """
        blue_print(
            f"\nAnalyzing CIDS detailed results for method='{self.method_name}', mode='{self.mode}', "
            f"language='{self.language}' (timestamp: {self.timestamp})..."
        )

        cids_df = self.load_metric_result("cids", result_type="detailed")

        if cids_df is None or not isinstance(cids_df, pd.DataFrame):
            print(
                f"Failed to load detailed CIDS results. Please check: {os.path.join(self.get_metric_dir('cids'), 'cids_detailed.xlsx')}"
            )
            return

        if cids_df.empty:
            print("CIDS detailed results are empty, cannot analyze.")
            return

        required_columns = ["character_name", "action_description", "cids_score"]
        if not all(col in cids_df.columns for col in required_columns):
            print(
                f"Error: DataFrame missing required columns. Required: {required_columns}, actual: {list(cids_df.columns)}"
            )
            return

        blue_print("\n--- CIDS average scores by character/action ---", bright=True)

        # Compute average by character
        char_avg = cids_df.groupby("character_name")["cids_score"].mean().sort_values(ascending=False)
        print("\n-- Average by character --")
        print(char_avg.to_string())

        # Compute average by action description
        action_avg = cids_df.groupby("action_description")["cids_score"].mean().sort_values(ascending=False)
        print("\n-- Average by action description --")
        print(action_avg.to_string())

        green_print("\nCIDS detailed analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage and analyze ViStoryBench evaluation results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--method", type=str, required=True, help="Name of the method (model) to analyze.")
    parser.add_argument("--mode", type=str, default="default", help="Method mode name (e.g., base, SD3).")
    parser.add_argument("--language", type=str, default="en", help="Language (e.g., en, ch).")
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Specific timestamp to analyze. If not provided, the latest is selected automatically.",
    )
    parser.add_argument(
        "--analysis_type",
        type=str,
        required=True,
        choices=["total_avg", "cids_char_action"],
        help="Type of analysis to perform:\n"
        "  - total_avg: compute and display average scores for all metrics.\n"
        "  - cids_char_action: analyze CIDS detailed scores by character and action.",
    )
    parser.add_argument("--base_path", type=str, default="data/bench_results", help="Base path to store results.")

    args = parser.parse_args()

    manager = ResultManager(
        method_name=args.method,
        mode=args.mode,
        language=args.language,
        timestamp=args.timestamp,
        base_path=args.base_path,
    )

    if args.analysis_type == "total_avg":
        manager.analyze_total_average()
    elif args.analysis_type == "cids_char_action":
        manager.analyze_cids_by_char_action()