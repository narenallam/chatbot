#!/usr/bin/env python3
"""
Beautiful Test Report Viewer using Rich
Displays comprehensive test results in detailed tables with colors and formatting
Usage: cd backend && python scripts/display_test_report.py [report_file]
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.tree import Tree
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.syntax import Syntax

console = Console()


class TestReportViewer:
    def __init__(self, report_file: Optional[str] = None):
        self.report_file = report_file or self.find_latest_report()
        self.report_data = self.load_report()

    def find_latest_report(self) -> str:
        """Find the latest test report file"""
        logs_dir = Path("./logs")
        if not logs_dir.exists():
            console.print("[red]❌ No logs directory found![/red]")
            sys.exit(1)

        report_files = list(logs_dir.glob("test_report_*.json"))
        if not report_files:
            console.print("[red]❌ No test report files found![/red]")
            sys.exit(1)

        # Get the latest report
        latest = sorted(report_files, key=os.path.getmtime)[-1]
        return str(latest)

    def load_report(self) -> Dict[str, Any]:
        """Load and parse the test report JSON"""
        try:
            with open(self.report_file, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]❌ Error loading report: {e}[/red]")
            sys.exit(1)

    def display_header(self):
        """Display the report header with run information"""
        run_info = self.report_data.get("test_run_info", {})
        summary = self.report_data.get("test_summary", {})

        # Create header panel
        header_text = Text()
        header_text.append("🧪 COMPREHENSIVE TEST REPORT", style="bold blue")
        header_text.append("\n")
        header_text.append(
            f"📅 Run Date: {run_info.get('timestamp', 'Unknown')}", style="cyan"
        )
        header_text.append(
            f"\n⏱️  Duration: {run_info.get('duration_seconds', 0):.2f} seconds",
            style="cyan",
        )
        header_text.append(
            f"\n📁 Backend: {run_info.get('backend_directory', 'Unknown')}",
            style="cyan",
        )
        header_text.append(
            f"\n📊 Test Data: {run_info.get('test_data_directory', 'Unknown')}",
            style="cyan",
        )

        header_panel = Panel(
            header_text,
            title="[bold green]Test Execution Info[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )

        # Create summary panel
        total_tests = summary.get("total_tests", 0)
        passed_tests = summary.get("passed_tests", 0)
        failed_tests = summary.get("failed_tests", 0)
        success_rate = summary.get("success_rate", 0)

        summary_text = Text()
        summary_text.append(f"📊 Total Tests: {total_tests}", style="bold")
        summary_text.append(f"\n✅ Passed: {passed_tests}", style="bold green")
        summary_text.append(f"\n❌ Failed: {failed_tests}", style="bold red")
        summary_text.append(
            f"\n📈 Success Rate: {success_rate:.1f}%", style="bold cyan"
        )

        # Color code the success rate
        if success_rate >= 80:
            rate_style = "bold green"
        elif success_rate >= 60:
            rate_style = "bold yellow"
        else:
            rate_style = "bold red"

        summary_text.stylize(rate_style, 93, len(summary_text))  # Style the percentage

        summary_panel = Panel(
            summary_text,
            title="[bold blue]Test Summary[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )

        # Display both panels side by side
        console.print(Columns([header_panel, summary_panel], equal=True))
        console.print()

    def display_test_results_table(self):
        """Display detailed test results in a table"""
        test_results = self.report_data.get("test_results", {})

        # Create the main test results table
        table = Table(
            title="🧪 Detailed Test Results",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
            title_style="bold cyan",
        )

        table.add_column("#", style="dim", width=3)
        table.add_column("Test Name", style="bold", min_width=25)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Details", min_width=30)
        table.add_column("Error/Info", min_width=25)

        for i, (test_name, result) in enumerate(test_results.items(), 1):
            success = result.get("success", False)
            duration = result.get("duration", 0)
            details = result.get("details", {})
            error = result.get("error", "")

            # Status formatting
            if success:
                status = "[green]✅ PASS[/green]"
            else:
                status = "[red]❌ FAIL[/red]"

            # Duration formatting
            duration_str = f"{duration:.3f}s"
            if duration > 5:
                duration_str = f"[red]{duration_str}[/red]"
            elif duration > 2:
                duration_str = f"[yellow]{duration_str}[/yellow]"
            else:
                duration_str = f"[green]{duration_str}[/green]"

            # Extract key details
            detail_info = []
            if isinstance(details, dict):
                if details.get("file_count"):
                    detail_info.append(f"Files: {details['file_count']}")
                if details.get("successful_conversions"):
                    detail_info.append(f"Success: {details['successful_conversions']}")
                if details.get("chunk_count"):
                    detail_info.append(f"Chunks: {details['chunk_count']}")
                if details.get("throughput_mb_per_s"):
                    detail_info.append(
                        f"Speed: {details['throughput_mb_per_s']:.1f} MB/s"
                    )
                if details.get("documents_added"):
                    detail_info.append(f"Added: {details['documents_added']}")
                if details.get("storage_cleared"):
                    detail_info.append(
                        "Storage: Cleared"
                        if details["storage_cleared"]
                        else "Storage: Error"
                    )

            detail_str = " | ".join(detail_info) if detail_info else "N/A"

            # Error formatting
            error_str = error[:50] + "..." if len(error) > 50 else error
            if error:
                error_str = f"[red]{error_str}[/red]"
            else:
                error_str = "[green]None[/green]"

            table.add_row(
                str(i),
                test_name.replace("_", " ").title(),
                status,
                duration_str,
                detail_str,
                error_str,
            )

        console.print(table)
        console.print()

    def display_performance_benchmarks(self):
        """Display performance benchmark results"""
        benchmarks = self.report_data.get("performance_benchmarks", {})

        if not benchmarks:
            console.print("[yellow]⚠️  No performance benchmark data available[/yellow]")
            return

        # Create performance table
        perf_table = Table(
            title="🚀 Performance Benchmarks",
            show_header=True,
            header_style="bold green",
            box=box.ROUNDED,
            title_style="bold green",
        )

        perf_table.add_column("File", style="bold", min_width=20)
        perf_table.add_column("Size (MB)", justify="right", width=12)
        perf_table.add_column("Avg Time (s)", justify="right", width=12)
        perf_table.add_column("Min Time (s)", justify="right", width=12)
        perf_table.add_column("Max Time (s)", justify="right", width=12)
        perf_table.add_column("Throughput (MB/s)", justify="right", width=15)
        perf_table.add_column("Chunks", justify="right", width=8)

        for filename, perf_data in benchmarks.items():
            file_size = perf_data.get("file_size_mb", 0)
            avg_time = perf_data.get("avg_processing_time", 0)
            min_time = perf_data.get("min_processing_time", 0)
            max_time = perf_data.get("max_processing_time", 0)
            throughput = perf_data.get("throughput_mb_per_s", 0)
            chunks = perf_data.get("chunk_count", 0)

            # Color code throughput
            if throughput > 30:
                throughput_str = f"[green]{throughput:.2f}[/green]"
            elif throughput > 10:
                throughput_str = f"[yellow]{throughput:.2f}[/yellow]"
            else:
                throughput_str = f"[red]{throughput:.2f}[/red]"

            perf_table.add_row(
                filename,
                f"{file_size:.2f}",
                f"{avg_time:.3f}",
                f"{min_time:.3f}",
                f"{max_time:.3f}",
                throughput_str,
                str(chunks),
            )

        console.print(perf_table)
        console.print()

    def display_system_info(self):
        """Display system information"""
        system_info = self.report_data.get("system_info", {})

        if not system_info:
            return

        info_text = Text()
        info_text.append(
            f"🐍 Python: {system_info.get('python_version', 'Unknown')}", style="cyan"
        )
        info_text.append(
            f"\n💻 Platform: {system_info.get('platform', 'Unknown')}", style="cyan"
        )
        info_text.append(
            f"\n📁 Working Dir: {system_info.get('working_directory', 'Unknown')}",
            style="cyan",
        )

        info_panel = Panel(
            info_text,
            title="[bold yellow]System Information[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )

        console.print(info_panel)
        console.print()

    def display_detailed_test_breakdown(self):
        """Display a tree view of detailed test information"""
        test_results = self.report_data.get("test_results", {})

        tree = Tree("🧪 [bold blue]Detailed Test Breakdown[/bold blue]")

        for test_name, result in test_results.items():
            success = result.get("success", False)
            details = result.get("details", {})

            # Create test node
            test_style = "green" if success else "red"
            test_node = tree.add(
                f"[{test_style}]{test_name.replace('_', ' ').title()}[/{test_style}]"
            )

            # Add details
            test_node.add(f"Status: {'✅ PASSED' if success else '❌ FAILED'}")
            test_node.add(f"Duration: {result.get('duration', 0):.3f}s")
            test_node.add(f"Timestamp: {result.get('timestamp', 'Unknown')}")

            if result.get("error"):
                test_node.add(f"[red]Error: {result['error']}[/red]")

            # Add specific details based on test type
            if isinstance(details, dict):
                details_node = test_node.add("[cyan]Details[/cyan]")
                for key, value in details.items():
                    if isinstance(value, (int, float)):
                        if key.endswith("_mb"):
                            details_node.add(f"{key}: {value:.2f} MB")
                        elif key.endswith("_seconds") or key.endswith("_time"):
                            details_node.add(f"{key}: {value:.3f}s")
                        else:
                            details_node.add(f"{key}: {value}")
                    elif isinstance(value, dict):
                        sub_node = details_node.add(f"[yellow]{key}[/yellow]")
                        for sub_key, sub_value in value.items():
                            sub_node.add(f"{sub_key}: {sub_value}")
                    else:
                        details_node.add(f"{key}: {value}")

        console.print(tree)
        console.print()

    def display_statistics_summary(self):
        """Display various statistics and metrics"""
        test_results = self.report_data.get("test_results", {})
        benchmarks = self.report_data.get("performance_benchmarks", {})

        # Calculate statistics
        total_duration = sum(
            result.get("duration", 0) for result in test_results.values()
        )
        avg_duration = total_duration / len(test_results) if test_results else 0

        # Performance stats
        if benchmarks:
            total_throughput = sum(
                b.get("throughput_mb_per_s", 0) for b in benchmarks.values()
            )
            avg_throughput = total_throughput / len(benchmarks)
            max_throughput = max(
                b.get("throughput_mb_per_s", 0) for b in benchmarks.values()
            )
            total_file_size = sum(b.get("file_size_mb", 0) for b in benchmarks.values())
        else:
            avg_throughput = max_throughput = total_file_size = 0

        # Create statistics table
        stats_table = Table(
            title="📊 Test Statistics",
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            title_style="bold cyan",
        )

        stats_table.add_column("Metric", style="bold", min_width=25)
        stats_table.add_column("Value", justify="right", min_width=15)
        stats_table.add_column("Unit", style="dim", min_width=10)

        stats_table.add_row("Total Test Duration", f"{total_duration:.3f}", "seconds")
        stats_table.add_row("Average Test Duration", f"{avg_duration:.3f}", "seconds")
        stats_table.add_row("Total Files Benchmarked", str(len(benchmarks)), "files")
        stats_table.add_row("Total File Size Tested", f"{total_file_size:.2f}", "MB")

        if benchmarks:
            stats_table.add_row("Average Throughput", f"{avg_throughput:.2f}", "MB/s")
            stats_table.add_row("Maximum Throughput", f"{max_throughput:.2f}", "MB/s")

        console.print(stats_table)
        console.print()

    def display_report(self):
        """Display the complete formatted report"""
        console.print()
        console.rule(
            "[bold blue]🧪 COMPREHENSIVE TEST REPORT VIEWER 🧪[/bold blue]",
            style="blue",
        )
        console.print()

        # Report file info
        file_info = f"📄 Report File: [cyan]{self.report_file}[/cyan]"
        console.print(file_info)
        console.print()

        # Main sections
        self.display_header()
        self.display_test_results_table()
        self.display_performance_benchmarks()
        self.display_statistics_summary()
        self.display_system_info()
        self.display_detailed_test_breakdown()

        console.rule("[bold green]✨ End of Report ✨[/bold green]", style="green")
        console.print()


def main():
    """Main function"""
    # Parse command line arguments
    report_file = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        viewer = TestReportViewer(report_file)
        viewer.display_report()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Report viewing interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]❌ Error displaying report: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
