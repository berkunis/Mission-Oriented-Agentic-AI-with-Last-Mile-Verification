"""
Benchmark Runner
================

Performance benchmarking for NL-to-SQL agent evaluation.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nl_to_sql.agent import NLToSQLAgent
from nl_to_sql.llm.mock import MockLLM
from nl_to_sql.models import AgentResult

# Import tracking if available
try:
    from mlops.tracking import ExperimentTracker, ExperimentMetrics
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False


@dataclass
class TestCase:
    """A single test case for evaluation."""

    id: str
    query: str
    category: str
    difficulty: str
    expected_tables: list[str] = field(default_factory=list)
    expected_operations: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""

    test_case: TestCase
    success: bool
    sql: str | None
    attempts: int
    processing_time_ms: float
    verification_passed: bool
    error_message: str | None = None


@dataclass
class BenchmarkReport:
    """Summary report of a benchmark run."""

    run_id: str
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    success_rate: float
    avg_processing_time_ms: float
    avg_attempts: float
    results_by_category: dict[str, dict]
    results_by_difficulty: dict[str, dict]
    individual_results: list[BenchmarkResult]


class BenchmarkRunner:
    """
    Runs evaluation benchmarks on the NL-to-SQL agent.

    Features:
    - Load test cases from JSON
    - Run against configurable agent
    - Track metrics with MLflow
    - Generate detailed reports
    """

    def __init__(
        self,
        agent: NLToSQLAgent | None = None,
        tracker: "ExperimentTracker | None" = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            agent: Agent to benchmark (defaults to MockLLM agent)
            tracker: Optional experiment tracker
        """
        self.agent = agent or self._create_default_agent()
        self.tracker = tracker

    def _create_default_agent(self) -> NLToSQLAgent:
        """Create a default agent with mock responses for benchmarking."""
        mock_responses = {
            "customers": ["SELECT * FROM customers"],
            "products": ["SELECT * FROM products"],
            "premium": ["SELECT * FROM customers WHERE tier = 'premium'"],
            "orders": ["SELECT * FROM orders"],
            "amount greater": ["SELECT * FROM orders WHERE amount > 100"],
            "total revenue": ["SELECT SUM(amount) FROM orders"],
            "how many customers": ["SELECT COUNT(*) FROM customers"],
            "average order": ["SELECT AVG(amount) FROM orders"],
            "by customer": [
                "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id"
            ],
            "by category": [
                "SELECT category, COUNT(*) FROM products GROUP BY category"
            ],
            "ordered by name": ["SELECT * FROM customers ORDER BY name"],
            "top 5": [
                "SELECT c.name, SUM(o.amount) as total FROM customers c "
                "JOIN orders o ON c.id = o.customer_id "
                "GROUP BY c.name ORDER BY total DESC LIMIT 5"
            ],
            "with their order": [
                "SELECT c.name, o.amount FROM customers c "
                "JOIN orders o ON c.id = o.customer_id"
            ],
            "spent more than": [
                "SELECT c.name, SUM(o.amount) as total FROM customers c "
                "JOIN orders o ON c.id = o.customer_id "
                "GROUP BY c.name HAVING total > 500"
            ],
            "most popular": [
                "SELECT p.category, COUNT(*) as count FROM products p "
                "JOIN orders o ON p.id = o.product_id "
                "GROUP BY p.category ORDER BY count DESC LIMIT 1"
            ],
            "delete": ["SELECT * FROM customers WHERE status = 'inactive'"],
            "first 10": ["SELECT * FROM customers LIMIT 10"],
        }
        llm = MockLLM(responses=mock_responses)
        return NLToSQLAgent(llm=llm, max_retries=3)

    def load_test_cases(self, filepath: str | Path) -> list[TestCase]:
        """
        Load test cases from a JSON file.

        Args:
            filepath: Path to the test cases JSON file

        Returns:
            List of TestCase objects
        """
        with open(filepath) as f:
            data = json.load(f)

        test_cases = []
        for tc in data["test_cases"]:
            test_cases.append(
                TestCase(
                    id=tc["id"],
                    query=tc["query"],
                    category=tc["category"],
                    difficulty=tc["difficulty"],
                    expected_tables=tc.get("expected_tables", []),
                    expected_operations=tc.get("expected_operations", []),
                    notes=tc.get("notes", ""),
                )
            )

        return test_cases

    def run_single(self, test_case: TestCase) -> BenchmarkResult:
        """
        Run a single test case.

        Args:
            test_case: Test case to run

        Returns:
            BenchmarkResult with outcome details
        """
        start_time = time.perf_counter()

        try:
            result = self.agent.process(test_case.query)
            processing_time = (time.perf_counter() - start_time) * 1000

            return BenchmarkResult(
                test_case=test_case,
                success=result.success,
                sql=result.sql,
                attempts=result.attempts,
                processing_time_ms=processing_time,
                verification_passed=result.success,
                error_message=None if result.success else result.final_message,
            )

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                test_case=test_case,
                success=False,
                sql=None,
                attempts=0,
                processing_time_ms=processing_time,
                verification_passed=False,
                error_message=str(e),
            )

    def run_benchmark(
        self,
        test_cases: list[TestCase] | None = None,
        test_file: str | Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BenchmarkReport:
        """
        Run a full benchmark suite.

        Args:
            test_cases: List of test cases (or load from file)
            test_file: Path to test cases file
            progress_callback: Optional callback for progress updates

        Returns:
            BenchmarkReport with full results
        """
        # Load test cases
        if test_cases is None:
            if test_file is None:
                test_file = Path(__file__).parent.parent / "data" / "sample_queries.json"
            test_cases = self.load_test_cases(test_file)

        # Track in MLflow if available
        if self.tracker:
            self.tracker.log_agent_config(self.agent)

        # Run tests
        results: list[BenchmarkResult] = []
        total = len(test_cases)

        for i, tc in enumerate(test_cases):
            result = self.run_single(tc)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

            # Log individual result
            if self.tracker:
                self.tracker.log_result(
                    self.agent.audit_trail[-1] if self.agent.audit_trail else None,
                    result.processing_time_ms,
                )

        # Generate report
        report = self._generate_report(results)

        # Log summary metrics
        if self.tracker and TRACKING_AVAILABLE:
            batch_results = [
                (
                    AgentResult(
                        success=r.success,
                        sql=r.sql,
                        original_query=r.test_case.query,
                        attempts=r.attempts,
                        audit_trail=[],
                        final_message="",
                    ),
                    r.processing_time_ms,
                )
                for r in results
            ]
            self.tracker.log_batch_results(batch_results)

        return report

    def _generate_report(self, results: list[BenchmarkResult]) -> BenchmarkReport:
        """Generate a benchmark report from results."""
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Calculate overall metrics
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed
        success_rate = passed / total if total > 0 else 0.0

        total_time = sum(r.processing_time_ms for r in results)
        avg_time = total_time / total if total > 0 else 0.0

        total_attempts = sum(r.attempts for r in results)
        avg_attempts = total_attempts / total if total > 0 else 0.0

        # Group by category
        by_category: dict[str, dict] = {}
        for r in results:
            cat = r.test_case.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "passed": 0, "failed": 0}
            by_category[cat]["total"] += 1
            if r.success:
                by_category[cat]["passed"] += 1
            else:
                by_category[cat]["failed"] += 1

        for cat in by_category:
            stats = by_category[cat]
            stats["success_rate"] = stats["passed"] / stats["total"]

        # Group by difficulty
        by_difficulty: dict[str, dict] = {}
        for r in results:
            diff = r.test_case.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "passed": 0, "failed": 0}
            by_difficulty[diff]["total"] += 1
            if r.success:
                by_difficulty[diff]["passed"] += 1
            else:
                by_difficulty[diff]["failed"] += 1

        for diff in by_difficulty:
            stats = by_difficulty[diff]
            stats["success_rate"] = stats["passed"] / stats["total"]

        return BenchmarkReport(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            total_tests=total,
            passed=passed,
            failed=failed,
            success_rate=success_rate,
            avg_processing_time_ms=avg_time,
            avg_attempts=avg_attempts,
            results_by_category=by_category,
            results_by_difficulty=by_difficulty,
            individual_results=results,
        )

    def print_report(self, report: BenchmarkReport) -> None:
        """Print a formatted benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)
        print(f"Run ID: {report.run_id}")
        print(f"Timestamp: {report.timestamp}")
        print()

        print("OVERALL RESULTS")
        print("-" * 40)
        print(f"Total Tests:      {report.total_tests}")
        print(f"Passed:           {report.passed}")
        print(f"Failed:           {report.failed}")
        print(f"Success Rate:     {report.success_rate:.1%}")
        print(f"Avg Time:         {report.avg_processing_time_ms:.2f}ms")
        print(f"Avg Attempts:     {report.avg_attempts:.2f}")
        print()

        print("BY CATEGORY")
        print("-" * 40)
        for cat, stats in sorted(report.results_by_category.items()):
            print(
                f"  {cat:20} {stats['passed']}/{stats['total']} "
                f"({stats['success_rate']:.0%})"
            )
        print()

        print("BY DIFFICULTY")
        print("-" * 40)
        for diff, stats in sorted(report.results_by_difficulty.items()):
            print(
                f"  {diff:20} {stats['passed']}/{stats['total']} "
                f"({stats['success_rate']:.0%})"
            )
        print()

        # Show failures
        failures = [r for r in report.individual_results if not r.success]
        if failures:
            print("FAILURES")
            print("-" * 40)
            for f in failures:
                print(f"  [{f.test_case.id}] {f.test_case.query[:40]}...")
                print(f"    Error: {f.error_message}")
            print()


def main():
    """Run benchmark from command line."""
    print("Running NL-to-SQL Agent Benchmark...")

    # Create tracker if available
    tracker = None
    if TRACKING_AVAILABLE:
        tracker = ExperimentTracker(experiment_name="nl-to-sql-benchmark")

    # Create runner
    runner = BenchmarkRunner(tracker=tracker)

    # Run with progress
    def progress(current, total):
        print(f"  Progress: {current}/{total}", end="\r")

    if TRACKING_AVAILABLE and tracker:
        with tracker.start_run(run_name="benchmark_run"):
            report = runner.run_benchmark(progress_callback=progress)
    else:
        report = runner.run_benchmark(progress_callback=progress)

    print()  # Clear progress line
    runner.print_report(report)


if __name__ == "__main__":
    main()
