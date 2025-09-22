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
        # Define o caminho para o executável Python do ambiente virtual de produção
        project_root = Path(__file__).parent
        venv_python = project_root / '.venv-prod' / 'bin' / 'python'
        
        # Garante que o executável do venv exista
        if not venv_python.exists():
            print(f"❌ Erro: Ambiente virtual de produção não encontrado em '{venv_python}'")
            print("Por favor, execute o script 'setup_ambientes.sh' para criar os ambientes.")
            return False

        result = subprocess.run([str(venv_python), script_path],
                                check=True,
                                cwd='src',
                                text=True,
                                capture_output=True)
        
        print(result.stdout)
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}: Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
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
    else:
        print(f"\n⚠️  Analysis incomplete. Please check error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
    
