"""Standalone Test Runner for Promob Render Enhancer 4-Tier E2E Test Suite.

Executes all 4 test tiers using pytest, tracks granular per-tier statistics,
prints a structured summary table, and exits with code 0 on success.

Usage:
    python tests/e2e/run_e2e_tests.py
    python tests/e2e/run_e2e_tests.py --tier 1
    python tests/e2e/run_e2e_tests.py --verbose
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

import pytest


@dataclass
class TierResult:
    name: str
    target_count: int
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0

    @property
    def status_label(self) -> str:
        if self.failed > 0 or self.errors > 0:
            return "FAIL"
        if self.passed > 0 and self.passed + self.skipped == self.total:
            return "PASS"
        if self.skipped == self.total and self.total > 0:
            return "SKIP (Awaiting Backend)"
        return "UNKNOWN"


class TierMetricsCollector:
    """Pytest plugin collecting execution metrics segmented by test tier."""

    TIER_MAP = {
        "test_tier1_features.py": "Tier 1: Feature Coverage",
        "test_tier1_generative_pbr.py": "Tier 1: Feature Coverage",
        "test_tier1_relighting_contracts.py": "Tier 1: Feature Coverage",
        "test_tier2_boundaries.py": "Tier 2: Boundary & Corner Cases",
        "test_tier2_generative_boundaries.py": "Tier 2: Boundary & Corner Cases",
        "test_tier2_relighting_boundaries.py": "Tier 2: Boundary & Corner Cases",
        "test_tier3_combinations.py": "Tier 3: Pairwise Combinations",
        "test_tier3_generative_combinations.py": "Tier 3: Pairwise Combinations",
        "test_tier4_workloads.py": "Tier 4: Real-World Workloads",
        "test_tier4_generative_workloads.py": "Tier 4: Real-World Workloads",
    }

    TARGETS = {
        "Tier 1: Feature Coverage": 108,
        "Tier 2: Boundary & Corner Cases": 98,
        "Tier 3: Pairwise Combinations": 30,
        "Tier 4: Real-World Workloads": 13,
    }

    def __init__(self):
        self.tiers: Dict[str, TierResult] = {
            name: TierResult(name=name, target_count=self.TARGETS[name])
            for name in self.TARGETS
        }
        self.failures: List[str] = []

    def pytest_runtest_logreport(self, report):
        if report.when != "call" and not (report.when == "setup" and report.skipped):
            if not (report.when in ("setup", "teardown") and report.failed):
                return

        file_key = os.path.basename(report.fspath)
        tier_name = self.TIER_MAP.get(file_key, "Other")
        if tier_name not in self.tiers:
            self.tiers[tier_name] = TierResult(name=tier_name, target_count=0)

        tier = self.tiers[tier_name]
        tier.total += 1
        tier.duration += getattr(report, "duration", 0.0)

        if report.passed:
            tier.passed += 1
        elif report.skipped:
            tier.skipped += 1
        elif report.failed:
            if report.when in ("setup", "teardown"):
                tier.errors += 1
            else:
                tier.failed += 1
            self.failures.append(f"[{tier_name}] {report.nodeid}: {report.longreprtext}")


def print_summary_table(
    collector: TierMetricsCollector,
    total_elapsed: float,
    allow_skip: bool = False,
    pytest_exit_code: int = 0
) -> int:
    """Formats and prints the executive E2E test report table."""
    print("\n" + "=" * 80)
    print(" " * 18 + "PROMOB RENDER ENHANCER — E2E TEST REPORT")
    print("=" * 80)
    print(f" Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f" Target URL: {os.environ.get('BASE_URL', 'http://127.0.0.1:8000')}")
    print(f" Total Duration: {total_elapsed:.2f}s")
    print("-" * 80)
    header = f"{'Test Tier':<35} | {'Target':<6} | {'Total':<5} | {'Pass':<5} | {'Fail':<5} | {'Skip':<5} | {'Status'}"
    print(header)
    print("-" * 80)

    total_target = 0
    total_total = 0
    total_pass = 0
    total_fail = 0
    total_skip = 0

    for name, tier in collector.tiers.items():
        total_target += tier.target_count
        total_total += tier.total
        total_pass += tier.passed
        total_fail += (tier.failed + tier.errors)
        total_skip += tier.skipped

        row = (
            f"{tier.name:<35} | "
            f"{tier.target_count:<6} | "
            f"{tier.total:<5} | "
            f"{tier.passed:<5} | "
            f"{(tier.failed + tier.errors):<5} | "
            f"{tier.skipped:<5} | "
            f"{tier.status_label}"
        )
        print(row)

    print("-" * 80)
    if total_fail > 0:
        overall_status = "FAIL"
    elif total_pass == 0 and total_skip > 0:
        overall_status = "SKIPPED (Awaiting Backend)"
    elif total_total == 0:
        overall_status = "NO TESTS COLLECTED"
    else:
        overall_status = "PASS"

    summary_row = (
        f"{'TOTAL SUITE METRICS':<35} | "
        f"{total_target:<6} | "
        f"{total_total:<5} | "
        f"{total_pass:<5} | "
        f"{total_fail:<5} | "
        f"{total_skip:<5} | "
        f"{overall_status}"
    )
    print(summary_row)
    print("=" * 80)

    if total_fail > 0:
        print("\nFAILURE DETAILS:")
        for fail in collector.failures:
            print("-" * 80)
            print(fail)
        print("=" * 80)
        return int(pytest_exit_code) if pytest_exit_code != 0 else 1

    # Preserve and propagate non-zero pytest exit codes (e.g. code 5 for NO_TESTS_COLLECTED)
    if pytest_exit_code != 0:
        return int(pytest_exit_code)

    # When all tests are skipped awaiting backend, return 1 unless --allow-skip is passed
    if total_pass == 0 and total_skip > 0 and not allow_skip:
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Promob Render Enhancer 4-Tier E2E Test Runner")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], help="Run only a specific tier")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output")
    parser.add_argument("-k", "--filter", type=str, help="Pytest keyword expression filter")
    parser.add_argument("-m", "--marker", type=str, help="Pytest marker expression filter (e.g. 'not gpu and not relighting')")
    parser.add_argument("--include-gpu", action="store_true", help="Include @pytest.mark.gpu and @pytest.mark.relighting tests")
    parser.add_argument("--cpu-only", action="store_true", help="Run only non-GPU / non-relighting baseline tests")
    parser.add_argument("--allow-skip", action="store_true", help="Exit 0 even if all tests are skipped awaiting backend")
    args = parser.parse_args()

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    pytest_args = [tests_dir]

    if args.tier == 1:
        pytest_args = [
            os.path.join(tests_dir, "test_tier1_features.py"),
            os.path.join(tests_dir, "test_tier1_generative_pbr.py"),
            os.path.join(tests_dir, "test_tier1_relighting_contracts.py"),
        ]
    elif args.tier == 2:
        pytest_args = [
            os.path.join(tests_dir, "test_tier2_boundaries.py"),
            os.path.join(tests_dir, "test_tier2_generative_boundaries.py"),
            os.path.join(tests_dir, "test_tier2_relighting_boundaries.py"),
        ]
    elif args.tier == 3:
        pytest_args = [
            os.path.join(tests_dir, "test_tier3_combinations.py"),
            os.path.join(tests_dir, "test_tier3_generative_combinations.py"),
        ]
    elif args.tier == 4:
        pytest_args = [
            os.path.join(tests_dir, "test_tier4_workloads.py"),
            os.path.join(tests_dir, "test_tier4_generative_workloads.py"),
        ]

    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    if args.filter:
        pytest_args.extend(["-k", args.filter])

    if args.marker:
        pytest_args.extend(["-m", args.marker])
    elif args.cpu_only and not args.include_gpu:
        pytest_args.extend(["-m", "not gpu and not relighting"])

    collector = TierMetricsCollector()
    start_time = time.time()
    pytest_exit_code = pytest.main(pytest_args, plugins=[collector])
    elapsed = time.time() - start_time

    return print_summary_table(
        collector,
        elapsed,
        allow_skip=args.allow_skip,
        pytest_exit_code=int(pytest_exit_code)
    )


if __name__ == "__main__":
    sys.exit(main())
