#!/usr/bin/env python3
"""
HydroSense Test Runner

Comprehensive test runner for the HydroSense project.
Supports different test categories and provides detailed reporting.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
def ensure_venv_test():
    """Reinvoca o script com o Python do .venv-test se necessário."""
    # Caminho absoluto do projeto (onde está este script)
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root.joinpath('.venv-test', 'bin', 'python')
    venv_python_abs = venv_python.resolve()
    # Se já está rodando no venv-test, não faz nada
    if Path(sys.executable).resolve() == venv_python_abs:
        return
    # Se não está, reinvoca
    if not venv_python_abs.exists():
        print(f"❌ Python do ambiente .venv-test não encontrado em {venv_python_abs}. Rode setup_ambientes.sh.")
        sys.exit(1)
    print(f"🔄 Reinvocando no ambiente de testes: {venv_python_abs}")
    os.execv(str(venv_python_abs), [str(venv_python_abs)] + sys.argv)

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    try:
        venv_python = Path('.venv-test/bin/python').resolve()
        if not venv_python.exists():
            raise RuntimeError("Python do ambiente .venv-test não encontrado. Rode setup_ambientes.sh.")
        # Se o comando começa com sys.executable, substitui pelo python do venv-test
        if command[0] == sys.executable or command[0].endswith('python'):
            command = [str(venv_python)] + command[1:]
        result = subprocess.run(command, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"\u2705 {description} completed successfully")
            return True
        else:
            print(f"\u274c {description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\u274c Error running {description}: {str(e)}")
        return False

def check_dependencies():
    """Check if required testing dependencies are installed."""
    print("🔍 Checking test dependencies...")
    
    required_packages = ['pytest', 'pandas', 'numpy', 'sklearn', 'matplotlib', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages before running tests.")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def run_unit_tests():
    """Run unit tests."""
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root.joinpath('.venv-test', 'bin', 'python').resolve()
    pytest_bin = project_root.joinpath('.venv-test', 'bin', 'pytest').resolve()
    command = [str(pytest_bin), 'tests/unit/', '-v', '--tb=short']
    return run_command(command, "Unit Tests")

def run_integration_tests():
    """Run integration tests."""
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root.joinpath('.venv-test', 'bin', 'python').resolve()
    pytest_bin = project_root.joinpath('.venv-test', 'bin', 'pytest').resolve()
    command = [str(pytest_bin), 'tests/integration/', '-v', '--tb=short']
    return run_command(command, "Integration Tests")

def run_specific_test_category(category):
    """Run tests for a specific category."""
    category_map = {
        'model1': ['-m', 'model1'],
        'model2': ['-m', 'model2'],
        'plotting': ['-m', 'plotting'],
        'evaluation': ['-m', 'evaluation'],
        'pipeline': ['-m', 'pipeline'],
        'fast': ['-m', 'fast'],
        'slow': ['-m', 'slow']
    }
    
    if category not in category_map:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {', '.join(category_map.keys())}")
        return False
    
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root.joinpath('.venv-test', 'bin', 'python').resolve()
    pytest_bin = project_root.joinpath('.venv-test', 'bin', 'pytest').resolve()
    command = [str(pytest_bin), 'tests/', '-v'] + category_map[category]
    return run_command(command, f"Tests for category: {category}")

def run_all_tests():
    """Run all tests."""
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root.joinpath('.venv-test', 'bin', 'python').resolve()
    pytest_bin = project_root.joinpath('.venv-test', 'bin', 'pytest').resolve()
    command = [str(pytest_bin), 'tests/', '-v', '--tb=short']
    return run_command(command, "All Tests")

def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    print("📊 Running tests with coverage analysis...")
    
    # Check if pytest-cov is available
    try:
        import pytest_cov
        coverage_available = True
    except ImportError:
        print("⚠️  pytest-cov not installed. Installing...")
        install_cmd = [sys.executable, '-m', 'pip', 'install', 'pytest-cov']
        if not run_command(install_cmd, "Installing pytest-cov"):
            print("❌ Failed to install pytest-cov. Running tests without coverage.")
            coverage_available = False
        else:
            coverage_available = True
    
    if coverage_available:
        project_root = Path(__file__).parent.absolute()
        venv_python = project_root.joinpath('.venv-test', 'bin', 'python').resolve()
        pytest_bin = project_root.joinpath('.venv-test', 'bin', 'pytest').resolve()
        command = [
            str(pytest_bin), 'tests/', '-v',
            '--cov=src', '--cov-report=html', '--cov-report=term-missing',
            '--cov-report=xml'
        ]
        success = run_command(command, "Tests with Coverage")
        
        if success:
            print("\n📊 Coverage Report Generated:")
            print("   📄 HTML Report: htmlcov/index.html")
            print("   📄 XML Report: coverage.xml")
            print("   📄 Terminal Report: displayed above")
        
        return success
    else:
        return run_all_tests()

def run_performance_tests():
    """Run performance-focused tests."""
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root.joinpath('.venv-test', 'bin', 'python').resolve()
    pytest_bin = project_root.joinpath('.venv-test', 'bin', 'pytest').resolve()
    command = [
        str(pytest_bin), 'tests/', '-v',
        '-k', 'performance or scalability or memory',
        '--durations=0'
    ]
    return run_command(command, "Performance Tests")

def validate_test_structure():
    """Validate test directory structure."""
    print("🔍 Validating test structure...")
    
    required_dirs = [
        'tests',
        'tests/unit',
        'tests/integration',
        'tests/fixtures'
    ]
    
    required_files = [
        'tests/__init__.py',
        'tests/conftest.py',
        'tests/unit/__init__.py',
        'tests/integration/__init__.py'
    ]
    
    all_valid = True
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            all_valid = False
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            all_valid = False
    
    # Check test files
    test_files = list(Path('tests').rglob('test_*.py'))
    print(f"📁 Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"   📄 {test_file}")
    
    if len(test_files) == 0:
        print("❌ No test files found!")
        all_valid = False
    
    return all_valid

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="HydroSense Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                 # Run all tests
  python run_tests.py --unit                # Run only unit tests
  python run_tests.py --integration         # Run only integration tests
  python run_tests.py --category model1     # Run Model 1 tests
  python run_tests.py --coverage            # Run tests with coverage
  python run_tests.py --performance         # Run performance tests
  python run_tests.py --validate            # Validate test structure
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--category', type=str, help='Run tests for specific category')
    parser.add_argument('--coverage', action='store_true', help='Run tests with coverage reporting')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--validate', action='store_true', help='Validate test structure')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')
    
    args = parser.parse_args()
    
    print("🚀 HydroSense Test Runner")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        if args.check_deps:
            return 1
        print("\n⚠️  Continuing with tests despite missing dependencies...")
    
    if args.check_deps:
        return 0
    
    # Validate test structure
    if args.validate or not any(vars(args).values()):
        if not validate_test_structure():
            print("❌ Test structure validation failed!")
            return 1
        
        if args.validate:
            return 0
    
    success_count = 0
    total_count = 0
    
    # Run requested tests
    if args.unit:
        total_count += 1
        if run_unit_tests():
            success_count += 1
    
    if args.integration:
        total_count += 1
        if run_integration_tests():
            success_count += 1
    
    if args.category:
        total_count += 1
        if run_specific_test_category(args.category):
            success_count += 1
    
    if args.coverage:
        total_count += 1
        if run_tests_with_coverage():
            success_count += 1
    
    if args.performance:
        total_count += 1
        if run_performance_tests():
            success_count += 1
    
    if args.all or (not any([args.unit, args.integration, args.category, args.coverage, args.performance])):
        total_count += 1
        if run_all_tests():
            success_count += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful test runs: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 All tests completed successfully!")
        print("\n🚀 Your HydroSense application is ready for production!")
        return 0
    else:
        print(f"\n⚠️  {total_count - success_count} test run(s) failed.")
        print("Please review the error messages above and fix any issues.")
        return 1

if __name__ == "__main__":
    ensure_venv_test()
    exit_code = main()
    sys.exit(exit_code)
    
