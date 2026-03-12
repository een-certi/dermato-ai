#!/bin/bash

# DERMATO AI - Script de Inicialização Rápida
echo "=========================================="
echo "🩺 Iniciando DERMATO AI - Sistema de Triagem"
echo "=========================================="

# 1. Ativa o ambiente virtual
echo "📦 Carregando ambiente virtual..."
source .venv/bin/activate

# 2. Inicia o servidor de auditoria (MLflow) em background
echo "📊 Iniciando MLOps / Servidor de Auditoria..."
mlflow ui --port 5000 > /dev/null 2>&1 &
MLFLOW_PID=$!

# 3. Abre a interface Médica
echo "🌐 Iniciando Painel Clínico (Streamlit)..."
streamlit run src/app.py

# 4. Quando o usuário fechar o Streamlit (Ctrl+C), limpa os processos
echo "Desligando o sistema de forma segura..."
kill $MLFLOW_PID
echo "✅ Sistema encerrado."