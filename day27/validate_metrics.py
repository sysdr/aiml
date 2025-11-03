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
    
    # Pattern to match Mean, Std Dev, Variance, CV values
    patterns = {
        'Mean': r'Mean:\s*([\d.]+)',
        'Std Dev': r'Std Dev:\s*([\d.]+)',
        'Variance': r'Variance:\s*([\d.]+)',
        'CV': r'CV[%]?:\s*([\d.]+)',
        'Sample Variance': r'Sample Variance:\s*([\d.]+)',
        'Sample Std Dev': r'Sample Std Dev:\s*([\d.]+)',
    }
    
    # Find all metrics
    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, output, re.IGNORECASE)
        for match in matches:
            value = float(match)
            metrics_found.append((metric_name, value))
            if value == 0.0:
                errors.append(f"❌ {metric_name} is zero: {value}")
            elif value < 0:
                errors.append(f"⚠️  {metric_name} is negative: {value}")
    
    # Check for specific metrics that should exist
    required_metrics = ['Mean', 'Std Dev', 'Variance', 'CV']
    found_metric_types = set([m[0] for m in metrics_found])
    
    print(f"📊 Found {len(metrics_found)} metric values")
    print("\nSample of metrics found:")
    for metric_name, value in metrics_found[:10]:  # Show first 10
        status = "✅" if value != 0.0 else "❌"
        print(f"  {status} {metric_name}: {value}")
    
    if len(metrics_found) > 10:
        print(f"  ... and {len(metrics_found) - 10} more")
    
    # Validate all metrics
    all_valid = True
    zero_count = 0
    
    for metric_name, value in metrics_found:
        if value == 0.0:
            zero_count += 1
            all_valid = False
    
    print(f"\n📈 Validation Summary:")
    print(f"  Total metrics found: {len(metrics_found)}")
    print(f"  Zero values: {zero_count}")
    print(f"  Non-zero values: {len(metrics_found) - zero_count}")
    
    if all_valid and len(metrics_found) > 0:
        print("\n✅ SUCCESS: All metrics are non-zero!")
        print("✅ Dashboard metrics are properly updated and displayed")
        return True
    elif len(metrics_found) == 0:
        print("\n❌ ERROR: No metrics found in output!")
        return False
    else:
        print(f"\n❌ ERROR: Found {zero_count} zero values!")
        for error in errors[:10]:
            print(f"  {error}")
        return False

if __name__ == "__main__":
    success = validate_metrics()
    sys.exit(0 if success else 1)

