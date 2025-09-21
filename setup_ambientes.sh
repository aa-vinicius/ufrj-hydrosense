#!/bin/bash
# Script para criar e ativar ambientes virtuais para testes e produção

sudo apt-get update && sudo apt-get install -y python3 python3-pip

sudo apt-get install -y python3-venv

set -e

# Ambiente de testes (mock)
echo "Criando ambiente virtual para TESTES (mock)..."
python3 -m venv .venv-test
source .venv-test/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt

echo "Ambiente de testes criado e dependências instaladas."
deactivate

# Ambiente de produção (dados reais)
echo "Criando ambiente virtual para PRODUÇÃO..."
python3 -m venv .venv-prod
source .venv-prod/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Ambiente de produção criado e dependências instaladas."
deactivate

echo "Ambientes prontos! Para ativar, use:"
echo "  source .venv-test/bin/activate   # Para rodar testes (mock)"
echo "  source .venv-prod/bin/activate   # Para rodar scripts de produção"
