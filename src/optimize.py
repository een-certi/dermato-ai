# src/optimize.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
import h5py
import numpy as np
import optuna
import mlflow
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Reutilizando a classe do Dataset que já criamos
from train import HAM10000Dataset

def objective(trial):
    """
    Esta é a função que o TPE vai tentar otimizar (maximizar a acurácia).
    O objeto 'trial' é o cérebro do Optuna escolhendo os valores.
    """
    # 1. O TPE sugere os Hiperparâmetros de forma inteligente
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    
    # 2. Carregando os Dados (Usando um subconjunto pequeno para o TPE rodar rápido no i7)
    load_dotenv()
    hdf5_path = os.getenv("PROCESSED_DATA_PATH", "./data/processed") + "/dataset.h5"
    dataset = HAM10000Dataset(hdf5_path)
    
    # Usamos apenas 800 imagens para a fase de busca (pesquisa rápida)
    subset = Subset(dataset, list(range(800)))
    
    # Dividindo em Treino (80%) e Validação (20%) para o TPE saber se a IA decorou ou aprendeu
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_data, val_data = torch.utils.data.random_split(subset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # 3. Montando o Modelo
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 7)
    
    criterion = nn.CrossEntropyLoss()
    
    # Aplicando o otimizador que o TPE escolheu
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # 4. Treinamento ultra-rápido (Apenas 2 épocas por tentativa para economizar a CPU)
    EPOCHS = 2
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 5. Avaliação (O que o TPE realmente quer saber: Qual a Acurácia?)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    
    # Registrando essa tentativa específica no MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("val_accuracy", accuracy)

    return accuracy

def run_tpe_optimization():
    mlflow.set_experiment("DERMATO_Otimizacao_TPE")
    
    with mlflow.start_run(run_name="Estudo_Optuna_TPE"):
        logging.info("🧠 A iniciar o Estudo de Hiperparâmetros com TPE...")
        
        # 1. CRIAR UMA BASE DE DADOS LOCAL PARA O PAINEL VISUAL LER
        nome_base_dados = "sqlite:///optuna_estudo.db"
        
        # Criando o "Estudo" e guardando-o no disco
        study = optuna.create_study(
            study_name="dermato_mobileNet_tpe",
            storage=nome_base_dados,
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(),
            load_if_exists=True # Se parar e voltar a correr, ele continua de onde ficou!
        )
        
        # 2. Iniciar a busca (Vamos testar 10 combinações para ver o gráfico a ganhar forma)
        study.optimize(objective, n_trials=10)
        
        logging.info("🏆 Otimização Concluída!")
        logging.info(f"Melhor Acurácia Encontrada: {study.best_value:.4f}")
        logging.info(f"Melhores Hiperparâmetros: {study.best_params}")
        
        mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_accuracy", study.best_value)

if __name__ == "__main__":
    run_tpe_optimization()