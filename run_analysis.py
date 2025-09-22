#!/usr/bin/env python3
"""
HydroSense - Main Analysis Runner

This script runs the complete flow prediction analysis from the project root directory.
It executes all models and generates outputs in the proper directory structure.
"""

import sys
import os
import subprocess
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    
    try:
        # Use o mesmo executável Python que está executando este script
        venv_python = sys.executable
        # Use o mesmo executável Python que está executando este script
        venv_python = sys.executable
        result = subprocess.run([str(venv_python), script_path],
                                check=True,
                                cwd='src',
                                text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {str(e)}")
        return False
    
    return True

def main():
    """Main analysis runner"""
    print("🚀 Starting HydroSense Flow Prediction Analysis")
    print("=" * 60)
    
    # Ensure output directories exist
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    # List of scripts to run in order
    scripts = [
        ('flow_prediction_app.py', 'Model 1: Flow Prediction Training'),
        ('evaluate_model1.py', 'Model 1: Performance Evaluation'),
        ('evaluate_model2.py', 'Model 2: Error Prediction & Evaluation'),
        ('create_plots.py', 'Visualization: Time Series Plots with Confidence Intervals')
    ]
    
    # Run each script
    success_count = 0
    for script, description in scripts:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n⚠️  Stopping analysis due to error in {script}")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully completed: {success_count}/{len(scripts)} steps")
    
    if success_count == len(scripts):
        print("\n🎉 All analysis steps completed successfully!")
        print("\n📁 Output files generated:")
        print("   📊 outputs/model1_performance_metrics.csv")
        print("   📊 outputs/model2_error_prediction_metrics.csv") 
        print("   📈 outputs/flow_prediction_station_58030000_subbasin_24.png")
        print("   📈 outputs/flow_prediction_station_58060000_subbasin_36.png")
        print("\n📖 Documentation:")
        print("   📄 docs/README_results.md")
        print("\n🚀 Ready for deployment and business use!")
    else:
        print(f"\n⚠️  Analysis incomplete. Please check error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
    
