# src/train.py
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
import mlflow
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class HAM10000Dataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        
        logging.info("Carregando o Banco HDF5 para a RAM do seu i7...")
        with h5py.File(self.hdf5_path, 'r') as h5f:
            self.images = h5f['images'][:]
            self.labels = h5f['labels'][:]
        logging.info("Dados carregados!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def run_production_training():
    load_dotenv()
    hdf5_path = os.getenv("PROCESSED_DATA_PATH", "./data/processed") + "/dataset.h5"
    
    # ==========================================
    # OS HIPERPARÂMETROS DE OURO ENCONTRADOS PELO TPE
    # ==========================================
    EPOCHS = 5  # 5 Épocas é um bom balanço para a CPU
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0003383422168586194
    OPTIMIZER_NAME = "Adam"
    NUM_CLASSES = 7

    dataset = HAM10000Dataset(hdf5_path)
    
    # ---> MODO APRESENTAÇÃO RÁPIDA (Usando 2000 imagens para não demorar horas na sua CPU)
    # Se quiser treinar com todas as 10.015 imagens (Modo Real), comente as duas linhas abaixo
    # e mude o subset nas linhas de divisão para 'dataset'
    indices_reduzidos = list(range(2000)) 
    dataset_reduzido = Subset(dataset, indices_reduzidos)
    
    # Dividindo: 80% Treino e 20% Validação (Auditoria de aprendizado)
    tamanho_treino = int(0.8 * len(dataset_reduzido))
    tamanho_val = len(dataset_reduzido) - tamanho_treino
    train_data, val_data = random_split(dataset_reduzido, [tamanho_treino, tamanho_val])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    logging.info("Montando o modelo MobileNetV3-Large...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mlflow.set_experiment("DERMATO_Producao")
    
    with mlflow.start_run(run_name="Treinamento_Modelo_Candidato_Final"):
        mlflow.log_params({
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, 
            "learning_rate": LEARNING_RATE, "optimizer": OPTIMIZER_NAME,
            "architecture": "MobileNetV3_Large", "data_size": len(dataset_reduzido)
        })

        logging.info("🚀 Iniciando o Treinamento de Produção...")
        
        for epoch in range(EPOCHS):
            # Fase de Treinamento
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Fase de Validação (O teste final da época)
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            logging.info(f"Época {epoch+1}/{EPOCHS} | Val Acc: {val_acc:.2f}% | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)

        # Salvando o modelo definitivo
        os.makedirs("./models/registry", exist_ok=True)
        caminho_modelo_final = "./models/checkpoints/mobilenet_v3_dermato.pth"
        torch.save(model.state_dict(), caminho_modelo_final)
        logging.info(f"💾 Cérebro Definitivo salvo em: {caminho_modelo_final}")

if __name__ == "__main__":
    run_production_training()