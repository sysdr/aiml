#!/usr/bin/env python3
"""
Validate that all metrics displayed in the demo are non-zero
"""
import sys
import re
from pathlib import Path

def validate_metrics(output_file=None):
    """Validate metrics from demo output"""
    if output_file and Path(output_file).exists():
        with open(output_file, 'r') as f:
            output = f.read()
    else:
        # Run the demo and capture output
        import subprocess
        result = subprocess.run(
            ['python', 'lesson_code.py'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        output = result.stdout
    
    print("🔍 Validating Dashboard Metrics...\n")
    
    # Extract all numeric metrics
    metrics_found = []
    errors = []
    
    # Pattern to match correlation, covariance, and other metrics
    patterns = {
        'Correlation': r'correlation:\s*([\d.+-]+)',
        'Covariance': r'covariance:\s*([\d.+-]+)',
        'Average Correlation': r'Average correlation:\s*([\d.]+)',
        'Maximum Correlation': r'Maximum correlation:\s*([\d.]+)',
        'Minimum Correlation': r'Minimum correlation:\s*([\d.]+)',
    }
    
    # Find all metrics
    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, output, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match)
                metrics_found.append((metric_name, value))
                if value == 0.0 and metric_name not in ['Minimum Correlation']:  # Min corr can be 0
                    errors.append(f"❌ {metric_name} is zero: {value}")
            except ValueError:
                pass
    
    # Also check for correlation values in the report
    corr_matches = re.findall(r'([\d.+-]+)\s*$', output, re.MULTILINE)
    for match in corr_matches:
        try:
            value = float(match)
            if -1 <= value <= 1 and value != 0.0:
                metrics_found.append(('Correlation Value', value))
        except ValueError:
            pass
    
    # Check for specific metrics that should exist
    required_metric_types = ['Correlation', 'Average Correlation', 'Maximum Correlation']
    found_metric_types = set([m[0] for m in metrics_found])
    
    print(f"📊 Found {len(metrics_found)} metric values")
    print("\nSample of metrics found:")
    for metric_name, value in metrics_found[:15]:  # Show first 15
        status = "✅" if value != 0.0 or 'Minimum' in metric_name else "❌"
        print(f"  {status} {metric_name}: {value}")
    
    if len(metrics_found) > 15:
        print(f"  ... and {len(metrics_found) - 15} more")
    
    # Validate all metrics
    all_valid = True
    zero_count = 0
    
    for metric_name, value in metrics_found:
        if value == 0.0 and 'Minimum' not in metric_name:
            zero_count += 1
            all_valid = False
    
    print(f"\n📈 Validation Summary:")
    print(f"  Total metrics found: {len(metrics_found)}")
    print(f"  Zero values (excluding min): {zero_count}")
    print(f"  Non-zero values: {len(metrics_found) - zero_count}")
    
    if all_valid and len(metrics_found) > 0:
        print("\n✅ SUCCESS: All metrics are non-zero!")
        print("✅ Dashboard metrics are properly updated and displayed")
        return True
    elif len(metrics_found) == 0:
        print("\n❌ ERROR: No metrics found in output!")
        print("Make sure lesson_code.py runs successfully and generates output.")
        return False
    else:
        print(f"\n❌ ERROR: Found {zero_count} zero values!")
        for error in errors[:10]:
            print(f"  {error}")
        return False

if __name__ == "__main__":
    success = validate_metrics()
    sys.exit(0 if success else 1)

