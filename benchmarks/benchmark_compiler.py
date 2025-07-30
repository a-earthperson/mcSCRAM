#!/usr/bin/env python3
"""
Benchmark script for running scram-cli with different compilation passes
on a set of XML input files.
"""

import argparse
import glob
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import json
import os


def run_scram_cli(xml_file: str, compilation_pass: int, use_no_kn: bool = False, timeout: int = 300) -> Dict[str, Any]:
    """
    Run scram-cli on a single XML file with specified compilation pass.
    
    Args:
        xml_file: Path to the XML input file
        compilation_pass: Compilation pass value (0-5)
        use_no_kn: Whether to use the --no-kn flag
        timeout: Timeout in seconds for the subprocess
        
    Returns:
        Dictionary with results including success status, timing, and any errors
    """
    result = {
        'file': xml_file,
        'compilation_pass': compilation_pass,
        'no_kn': use_no_kn,
        'success': False,
        'duration': None,
        'error': None,
        'stdout': None,
        'stderr': None
    }
    
    # Build the command
    cmd = [
        './scram-cli',
        '--preprocessor',
        '--ccf',
    ]
    
    if use_no_kn:
        cmd.append('--no-kn')
    
    cmd.extend([
        '--compilation-passes', str(compilation_pass),
        xml_file
    ])
    
    # Record start time
    start_time = time.time()
    
    try:
        print(cmd)
        # Run the command with timeout
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Record duration
        result['duration'] = time.time() - start_time
        result['success'] = completed.returncode == 0
        result['stdout'] = completed.stdout
        result['stderr'] = completed.stderr
        
        if completed.returncode != 0:
            result['error'] = f"Process exited with code {completed.returncode}"
            
    except subprocess.TimeoutExpired as e:
        result['duration'] = timeout
        result['error'] = f"Timeout after {timeout} seconds"
        result['stdout'] = e.stdout.decode() if e.stdout else None
        result['stderr'] = e.stderr.decode() if e.stderr else None
        
    except Exception as e:
        result['duration'] = time.time() - start_time
        result['error'] = f"Exception: {type(e).__name__}: {str(e)}"
    
    return result


def benchmark_compilation_passes(
    files_or_pattern: Any,
    min_pass: int = 0,
    max_pass: int = 5,
    timeout: int = 300,
    output_file: str = None,
    run_variants: str = 'both'
) -> List[Dict[str, Any]]:
    """
    Run scram-cli with different compilation passes on XML files.
    
    Args:
        files_or_pattern: Either a glob pattern string or a list of file paths
        min_pass: Minimum compilation pass value (default: 0)
        max_pass: Maximum compilation pass value (default: 5)
        timeout: Timeout per run in seconds (default: 300)
        output_file: Optional JSON file to save results
        run_variants: Which variants to run: 'both', 'default', or 'no-kn' (default: 'both')
        
    Returns:
        List of result dictionaries
    """
    # Handle both pattern string and list of files
    if isinstance(files_or_pattern, str):
        # It's a pattern, use glob
        xml_files = glob.glob(files_or_pattern, recursive=True)
        if not xml_files:
            print(f"No files found matching pattern: {files_or_pattern}")
            return []
    elif isinstance(files_or_pattern, list):
        # It's a list of files
        xml_files = files_or_pattern
    else:
        print("Error: files_or_pattern must be either a string pattern or a list of files")
        return []
    
    print(f"Found {len(xml_files)} XML files")
    
    # Check if scram-cli exists
    if not os.path.exists('./scram-cli'):
        print("Error: scram-cli executable not found in current directory")
        return []
    
    results = []
    
    # Determine which variants to run
    variants_to_run = []
    if run_variants in ['both', 'default']:
        variants_to_run.append(('default', False))
    if run_variants in ['both', 'no-kn']:
        variants_to_run.append(('--no-kn', True))
    
    if not variants_to_run:
        print(f"Error: Invalid run_variants value: {run_variants}. Use 'both', 'default', or 'no-kn'")
        return []
    
    # Calculate total runs
    total_runs = len(xml_files) * (max_pass - min_pass + 1) * len(variants_to_run)
    completed_runs = 0
    
    # Iterate through all files and compilation passes
    for xml_file in xml_files:
        print(f"\nProcessing: {xml_file}")
        
        for compilation_pass in range(min_pass, max_pass + 1):
            for variant_name, use_no_kn in variants_to_run:
                completed_runs += 1
                print(f"  Pass {compilation_pass} ({variant_name}) ({completed_runs}/{total_runs})...", end='', flush=True)
                
                result = run_scram_cli(xml_file, compilation_pass, use_no_kn=use_no_kn, timeout=timeout)
                results.append(result)
                
                if result['success']:
                    print(f" ✓ ({result['duration']:.2f}s)")
                elif result['error'] and 'Timeout' in result['error']:
                    print(f" ⏱ (timeout)")
                else:
                    print(f" ✗ ({result['error']})")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    timeouts = sum(1 for r in results if r['error'] and 'Timeout' in r['error'])
    failures = len(results) - successful - timeouts
    
    print(f"\nSummary:")
    print(f"  Total runs: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Failures: {failures}")
    
    return results


def print_performance_table(results: List[Dict[str, Any]]):
    """Print a performance comparison table."""
    if not results:
        return
    
    # Group results by file and flag variant
    from collections import defaultdict
    by_file = defaultdict(lambda: defaultdict(dict))
    
    for result in results:
        if result['success']:
            variant = 'no_kn' if result.get('no_kn', False) else 'default'
            by_file[result['file']][variant][result['compilation_pass']] = result['duration']
    
    print("\nPerformance Comparison (seconds):")
    print("=" * 120)
    
    # Print header
    header = f"{'File':<35} | "
    for p in range(6):
        header += f"{'Pass ' + str(p):^16} | "
    print(header)
    
    sub_header = f"{' ':<35} | "
    for p in range(6):
        sub_header += f"{'Default':>7} {'--no-kn':>8} | "
    print(sub_header)
    print("-" * 120)
    
    for file_path, variants in sorted(by_file.items()):
        file_name = Path(file_path).name[:34]
        row = f"{file_name:<35} | "
        
        for p in range(6):
            # Default variant
            if 'default' in variants and p in variants['default']:
                row += f"{variants['default'][p]:>7.2f}"
            else:
                row += f"{'--':>7}"
            
            row += " "
            
            # --no-kn variant
            if 'no_kn' in variants and p in variants['no_kn']:
                row += f"{variants['no_kn'][p]:>7.2f}"
            else:
                row += f"{'--':>7}"
            
            row += " | "
        
        print(row)
    
    # Print summary statistics
    print("\n" + "=" * 120)
    print("\nSummary Statistics:")
    
    # Calculate speedup/slowdown
    speedups = []
    for file_path, variants in by_file.items():
        if 'default' in variants and 'no_kn' in variants:
            for p in range(6):
                if p in variants['default'] and p in variants['no_kn']:
                    default_time = variants['default'][p]
                    no_kn_time = variants['no_kn'][p]
                    speedup = (default_time - no_kn_time) / default_time * 100
                    speedups.append(speedup)
    
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"Average performance change with --no-kn: {avg_speedup:+.1f}%")
        if avg_speedup > 0:
            print("(Positive means --no-kn is faster)")
        else:
            print("(Negative means --no-kn is slower)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark scram-cli with different compilation passes"
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='XML files or wildcard pattern (e.g., "input/**/*.xml")'
    )
    parser.add_argument(
        '--min-pass',
        type=int,
        default=0,
        help='Minimum compilation pass value (default: 0)'
    )
    parser.add_argument(
        '--max-pass',
        type=int,
        default=5,
        help='Maximum compilation pass value (default: 5)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout per run in seconds (default: 300)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--table',
        action='store_true',
        help='Print performance comparison table'
    )
    parser.add_argument(
        '--variants',
        choices=['both', 'default', 'no-kn'],
        default='both',
        help='Which variants to run (default: both)'
    )
    
    args = parser.parse_args()
    
    # Determine if we have a pattern or list of files
    if len(args.files) == 1:
        # Could be either a single file or a pattern
        files_or_pattern = args.files[0]
    else:
        # Multiple files provided
        files_or_pattern = args.files
    
    # Run the benchmark
    results = benchmark_compilation_passes(
        files_or_pattern,
        args.min_pass,
        args.max_pass,
        args.timeout,
        args.output,
        args.variants
    )
    
    # Print performance table if requested
    if args.table:
        print_performance_table(results)


if __name__ == '__main__':
    main()