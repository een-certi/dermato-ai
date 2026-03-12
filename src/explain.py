# src/explain.py
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

class AuditorIA_GradCAM:
    """
    Motor de Auditoria Visual (Grad-CAM).
    Extrai os gradientes e as ativações da última camada da Rede Neural
    para entender onde a IA 'olhou' antes de tomar a decisão.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # "Escutas" (Hooks) na rede neural para capturar a matemática interna
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        # 1. Modo de avaliação (desliga o treinamento)
        self.model.eval()
        
        # 2. Faz a previsão (Forward Pass)
        output = self.model(input_tensor)
        
        if target_class is None:
            # Se não passarmos uma classe, ele audita a classe que a IA escolheu
            target_class = torch.argmax(output).item()

        # 3. Zera os gradientes anteriores e calcula a importância (Backward Pass)
        self.model.zero_grad()
        target = output[0][target_class]
        target.backward()

        # 4. Tira a média dos gradientes (Peso de importância de cada filtro)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # 5. Multiplica a importância (gradientes) pelo que a IA viu (ativações)
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # 6. Achata tudo em um único mapa 2D
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # 7. Remove os valores negativos (ReLU) - só queremos o que ajudou na decisão
        heatmap = F.relu(heatmap)
        
        # 8. Normaliza entre 0 e 1 para podermos pintar a imagem depois
        heatmap /= torch.max(heatmap)

        return heatmap.detach().cpu().numpy()

def criar_imagem_auditada(caminho_imagem, heatmap, alpha=0.5):
    """
    Pega a foto original da pele e 'carimba' o mapa de calor por cima.
    """
    # Lê a imagem original usando OpenCV
    img = cv2.imread(caminho_imagem)
    img = cv2.resize(img, (224, 224))
    
    # Redimensiona o mapa de calor para o tamanho da imagem (224x224)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Converte o mapa (0 a 1) para cores RGB de mapa de calor (0 a 255, Azul -> Vermelho)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Sobrepõe as duas imagens
    imagem_sobreposta = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    
    # Converte de BGR (padrão do OpenCV) para RGB (padrão de monitores/web)
    imagem_sobreposta = cv2.cvtColor(imagem_sobreposta, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(imagem_sobreposta)