import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configuração da Página
st.set_page_config(page_title="Dermato AI - Diagnóstico Auditável", layout="wide")

# Mapeamento das classes do dataset HAM10000
CLASSES = {
    0: "Queratose Actínica (Cancerígena)",
    1: "Carcinoma Basocelular",
    2: "Lesões Benignas tipo Queratose",
    3: "Dermatofibroma",
    4: "Melanoma (Altamente Maligno)",
    5: "Nevos Melanocíticos (Pintas Comuns)",
    6: "Lesões Vasculares"
}

# --- FUNÇÃO DE LOG PARA DATA DRIFT (LGPD-Compliant) ---
def log_drift_embedding(model, image_tensor, prediction):
    """
    Extrai o vetor de características (Embedding) da MobileNetV3.
    Salva apenas a estatística matemática, garantindo a privacidade do paciente.
    """
    try:
        model.eval()
        with torch.no_grad():
            # Extraímos as características antes da última camada de classificação
            features = model.features(image_tensor)
            pooled = model.avgpool(features)
            embedding = torch.flatten(pooled, 1).numpy()[0]
        
        # Criamos o registro com Timestamp e o Vetor de 1.280 dimensões
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "previsao": prediction
        }
        # Adiciona cada dimensão do embedding como uma coluna
        for i, val in enumerate(embedding):
            log_data[f"feat_{i}"] = val
            
        # Salva em CSV (Append mode)
        os.makedirs("data/production_logs", exist_ok=True)
        log_path = "data/production_logs/drift_embeddings.csv"
        
        df_new = pd.DataFrame([log_data])
        if not os.path.isfile(log_path):
            df_new.to_csv(log_path, index=False)
        else:
            df_new.to_csv(log_path, mode='a', header=False, index=False)
            
    except Exception as e:
        st.warning(f"Aviso de Sistema: Erro ao registrar logs de auditoria técnica ({e})")

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_trained_model():
    # Usamos MobileNetV3 Large para balanço entre performance e precisão em CPU
    model = models.mobilenet_v3_large(weights=None)
    
    # Ajustamos a cabeça de saída para as nossas 7 classes
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(CLASSES))
    
    # Carregamos os pesos treinados
    model_path = "models/checkpoints/mobilenet_v3_dermato.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

# --- INTERFACE STREAMLIT ---
def main():
    st.title("🏥 Dermato AI: Sistema de Triagem e MLOps")
    st.markdown("""
        Este sistema utiliza Deep Learning para auxiliar na triagem de lesões dermatológicas. 
        **Privacidade:** As imagens enviadas são processadas em memória e descartadas imediatamente. 
        Apenas metadados matemáticos são retidos para monitoramento de drift.
    """)

    model = load_trained_model()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Upload da Imagem")
        uploaded_file = st.file_uploader("Selecione uma imagem dermatoscópica...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Imagem Original", use_container_width=True)

    with col2:
        st.subheader("🔍 Análise de Inteligência Artificial")
        
        if uploaded_file:
            # Pré-processamento (idêntico ao treino)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            input_tensor = preprocess(image).unsqueeze(0)

            with st.spinner("Analisando padrões morfológicos..."):
                # Inferência
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    prob, class_id = torch.max(probabilities, 0)
                
                prediction = CLASSES[class_id.item()]
                confidencia = prob.item() * 100

                # --- DISPARA LOG DE DRIFT ---
                log_drift_embedding(model, input_tensor, prediction)

                # Exibição de Resultados
                st.metric(label="Diagnóstico Provável", value=prediction)
                st.progress(confidencia / 100)
                st.write(f"**Grau de Confiança:** {confidencia:.2f}%")

                if class_id.item() in [0, 1, 4]:
                    st.error("⚠️ Atenção: Esta lesão apresenta características de alta prioridade clínica. Recomenda-se biópsia.")
                else:
                    st.info("ℹ️ Nota: Lesão com características de acompanhamento de rotina.")

    st.divider()
    st.caption("Aviso Legal: Esta ferramenta é uma Prova de Conceito (PoC) para MLOps e não substitui o diagnóstico médico.")

if __name__ == "__main__":
    main()