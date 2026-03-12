# tests/test_model.py
import torch
import torch.nn as nn
from torchvision import models
import os

def test_arquitetura_modelo():
    """Garante que a rede neural está configurada para as 7 doenças."""
    # 1. Monta o modelo
    modelo = models.mobilenet_v3_large(weights=None)
    in_features = modelo.classifier[3].in_features
    modelo.classifier[3] = nn.Linear(in_features, 7)
    
    # 2. Cria uma imagem falsa (ruído) apenas para testar a matemática
    # Formato: (Batch=1, Canais=3(RGB), Altura=224, Largura=224)
    imagem_falsa = torch.randn(1, 3, 224, 224)
    
    # 3. Passa a imagem pelo modelo
    saida = modelo(imagem_falsa)
    
    # 4. As Asserções (O coração do teste)
    assert saida.shape == (1, 7), f"Erro: O modelo devia dar 7 saídas, mas deu {saida.shape[1]}"
    assert not torch.isnan(saida).any(), "Erro: O modelo gerou valores nulos (NaN)!"

def test_pesos_existem():
    """Verifica se o treino realmente guardou o ficheiro .pth na pasta certa."""
    caminho_pesos = "./models/checkpoints/mobilenet_v3_dermato.pth"
    assert os.path.exists(caminho_pesos), "ALERTA CRÍTICO: O ficheiro de pesos .pth sumiu ou não foi gerado!"