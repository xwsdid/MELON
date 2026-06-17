import os
import re
import sys
import time
import subprocess

METHODS = ["melon", "melon_r", "melon_rp"]

TASK_PAIRS = [
    ("user_task_11", "injection_task_11"),
    ("user_task_12", "injection_task_12"),
    ("user_task_13", "injection_task_13"),
    ("user_task_14", "injection_task_14"),
    ("user_task_15", "injection_task_15"),
    ("user_task_16", "injection_task_16"),
    ("user_task_17", "injection_task_17"),
    ("user_task_18", "injection_task_18"),
    ("user_task_19", "injection_task_19"),
    ("user_task_20", "injection_task_20"),
]

TIMEOUT_SECONDS = 600


def extract_metric(pattern, text, name="metric"):
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1)
    else:
        print(f"[WARN] Failed to extract {name}")
        return "N/A"


def detect_skipped(output):
    """
    更鲁棒的 skipped 判断
    """
    patterns = [
        r"already run",
        r"skipping",
        r"cached",
        r"exists.*skip",
    ]
    for p in patterns:
        if re.search(p, output, re.IGNORECASE):
            return True
    return False


def run_one(method, user_task, injection_task):
    cmd = [
        sys.executable,
        "-m",
        "agentdojo.scripts.benchmark",
        "--model", "GPT_4O_2024_05_13",
        "--attack", "direct",
        "--benchmark-version", "v1.2.2",
        "--defense", method,
        "-s", "banking",
        "--user-task", user_task,
        "--injection-task", injection_task,
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"

    print("=" * 100)
    print(f"Running {method} | {user_task} | {injection_task}")
    print(" ".join(cmd))
    print("=" * 100)

    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            env=env,
        )

        elapsed = round(time.time() - start, 2)
        output = result.stdout + "\n" + result.stderr

        # ✅ 打印更多信息（避免你现在那种看不到关键字段的问题）
        print("\n[DEBUG] OUTPUT HEAD (first 2000 chars):")
        print(output[:2000])

        print("\n[DEBUG] OUTPUT TAIL (last 2000 chars):")
        print(output[-2000:])

        security = extract_metric(
            r"Average security[:\s]*([0-9.]+%?)",
            output,
            "security"
        )

        utility = extract_metric(
            r"Average utility[:\s]*([0-9.]+%?)",
            output,
            "utility"
        )

        skipped = detect_skipped(output)

        print(f"[PARSED] security={security}, utility={utility}, skipped={skipped}")

        return {
            "method": method,
            "user_task": user_task,
            "injection_task": injection_task,
            "returncode": result.returncode,
            "seconds": elapsed,
            "security": security,
            "utility": utility,
            "skipped": skipped,
            "raw_output": output,  # 🔥关键：方便你之后debug
        }

    except subprocess.TimeoutExpired:
        return {
            "method": method,
            "user_task": user_task,
            "injection_task": injection_task,
            "returncode": "TIMEOUT",
            "seconds": TIMEOUT_SECONDS,
            "security": "N/A",
            "utility": "N/A",
            "skipped": False,
            "raw_output": "",
        }


def pct_to_float(x):
    if x == "N/A":
        return None
    try:
        return float(x.replace("%", ""))
    except:
        return None


def summarize_by_method(results):
    by_method = {}

    for r in results:
        method = r["method"]
        by_method.setdefault(method, []).append(r)

    print("\n" + "=" * 100)
    print("SUMMARY BY METHOD")
    print("=" * 100)

    for method, rows in by_method.items():
        ok_rows = [r for r in rows if r["returncode"] == 0]
        fresh_rows = [r for r in ok_rows if not r["skipped"]]
        skipped_rows = [r for r in ok_rows if r["skipped"]]
        failed_rows = [r for r in rows if r["returncode"] != 0]

        sec_values = [
            pct_to_float(r["security"])
            for r in fresh_rows
            if pct_to_float(r["security"]) is not None
        ]

        util_values = [
            pct_to_float(r["utility"])
            for r in fresh_rows
            if pct_to_float(r["utility"]) is not None
        ]

        avg_sec = sum(sec_values) / len(sec_values) if sec_values else None
        avg_util = sum(util_values) / len(util_values) if util_values else None
        avg_time = (
            sum(r["seconds"] for r in fresh_rows) / len(fresh_rows)
            if fresh_rows else None
        )

        print(f"\nDefense: {method}")
        print(f"  total tasks:      {len(rows)}")
        print(f"  fresh completed:  {len(fresh_rows)}")
        print(f"  skipped cached:   {len(skipped_rows)}")
        print(f"  failed:           {len(failed_rows)}")

        print(f"  avg security:     {avg_sec:.2f}%" if avg_sec is not None else "  avg security:     N/A")
        print(f"  avg utility:      {avg_util:.2f}%" if avg_util is not None else "  avg utility:      N/A")
        print(f"  avg fresh time:   {avg_time:.2f}s" if avg_time is not None else "  avg fresh time:   N/A")

        if failed_rows:
            failed_names = [
                f"{r['user_task']}+{r['injection_task']}"
                for r in failed_rows
            ]
            print(f"  failed tasks:     {', '.join(failed_names)}")


def main():
    results = []

    for method in METHODS:
        for user_task, injection_task in TASK_PAIRS:
            results.append(run_one(method, user_task, injection_task))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for r in results:
        print(
            f"{r['method']:9s} | "
            f"{r['user_task']:12s} | "
            f"{r['injection_task']:16s} | "
            f"code={str(r['returncode']):7s} | "
            f"time={r['seconds']:7.2f}s | "
            f"security={r['security']:8s} | "
            f"utility={r['utility']:8s} | "
            f"skipped={r['skipped']}"
        )

    summarize_by_method(results)


if __name__ == "__main__":
    main()