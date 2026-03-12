# 🏥 DERMATO AI - Sistema de Triagem Oncológica Auditável

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Optimized-EE4C2C)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2)
![Optuna](https://img.shields.io/badge/HPO-Optuna-4B8BBE)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B)

Um pipeline *End-to-End* de Machine Learning Operations (MLOps) para classificação de lesões de pele (foco em Melanoma). O sistema processa imagens dermatoscópicas, realiza inferência usando arquitetura otimizada para CPU (MobileNetV3) e fornece **Explicabilidade Visual (Grad-CAM)** para apoio à decisão médica.

---

## 🚀 Arquitetura e Diferenciais (MLOps)

Este projeto foi construído focando em boas práticas de governança, rastreabilidade e performance:

1. **Ingestão em Memória (Zero-I/O Bottleneck):** Leitura direta do dataset HAM10000 compactado (`.zip`) para a RAM, redimensionando e salvando em um banco otimizado **HDF5**, reduzindo drasticamente o tempo de leitura do disco.
2. **Otimização Bayesiana (TPE):** Utilização do Optuna com *Tree-structured Parzen Estimator* para encontrar os melhores hiperparâmetros matematicamente, abandonando abordagens de tentativa e erro.
3. **Rastreabilidade de Experimentos:** Integração total com o **MLflow** para registrar parâmetros, métricas e artefatos de cada ciclo de treinamento.
4. **Governança e XAI (Explainable AI):** Implementação manual do algoritmo **Grad-CAM**. A IA não atua como "caixa preta": ela gera um mapa de calor apontando as estruturas morfológicas que guiaram o diagnóstico, permitindo auditoria clínica.
5. **CI/CD Pipeline Local:** Testes automatizados usando `pytest` para garantir a integridade da arquitetura da rede neural e do banco de dados antes do deploy.

---

## 📂 Estrutura do Projeto

```text
DERMATO/
│
├── data/
│   ├── raw/                 # Dados brutos originais (Ignorados pelo Git)
│   └── processed/           # Banco de dados dataset.h5 otimizado
│
├── models/
│   └── checkpoints/         # Pesos do modelo treinado (.pth)
│
├── src/
│   ├── dataset.py           # Pipeline de ETL (ZIP -> HDF5)
│   ├── optimize.py          # Busca de Hiperparâmetros (Optuna/TPE)
│   ├── train.py             # Treinamento do modelo candidato final
│   ├── explain.py           # Motor de Auditoria Visual (Grad-CAM)
│   └── app.py               # Interface Clínica Web (Streamlit)
│
├── tests/                   # Bateria de testes automatizados (Pytest)
│
├── start_dermato.sh         # Script de inicialização One-Click
├── requirements.txt         # Dependências de produção
└── README.md                # Documentação


⚖️ Aviso Legal e Ético

Este aplicação é uma Prova de Conceito (PoC) para demonstração de Arquitetura de Software e MLOps. Não é um dispositivo médico certificado. Qualquer decisão clínica deve ser tomada exclusivamente por um médico dermatologista utilizando equipamentos devidamente homologados pela ANVISA/FDA.