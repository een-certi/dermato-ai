import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import os
import cv2

# Importamos a função de explicabilidade que já tínhamos criado
from explain import generate_gradcam

# Configuração da Página
st.set_page_config(page_title="Dermato AI - Diagnóstico Auditável", layout="wide")

CLASSES = {
    0: "Queratose Actínica", 1: "Carcinoma Basocelular", 2: "Lesões Benignas",
    3: "Dermatofibroma", 4: "Melanoma", 5: "Nevos (Pintas)", 6: "Lesões Vasculares"
}

# --- LOG DE DATA DRIFT (LGPD) ---
def log_drift_embedding(model, image_tensor, prediction):
    try:
        model.eval()
        with torch.no_grad():
            features = model.features(image_tensor)
            pooled = model.avgpool(features)
            embedding = torch.flatten(pooled, 1).numpy()[0]
        
        log_data = {"timestamp": datetime.now().isoformat(), "previsao": prediction}
        for i, val in enumerate(embedding):
            log_data[f"feat_{i}"] = val
            
        os.makedirs("data/production_logs", exist_ok=True)
        log_path = "data/production_logs/drift_embeddings.csv"
        df_new = pd.DataFrame([log_data])
        
        if not os.path.isfile(log_path):
            df_new.to_csv(log_path, index=False)
        else:
            df_new.to_csv(log_path, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Erro no log de drift: {e}")

# --- CARREGAMENTO DO MODELO ---
@st.cache_resource
def load_trained_model():
    model = models.mobilenet_v3_large(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(CLASSES))
    
    model_path = "models/checkpoints/mobilenet_v3_dermato.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- INTERFACE ---
def main():
    st.title("🏥 Dermato AI - Triagem com Auditoria Visual")
    
    model = load_trained_model()
    
    uploaded_file = st.sidebar.file_uploader("Carregar imagem da lesão", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        
        # Pré-processamento
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(img).unsqueeze(0)

        # Execução do Pipeline (IA + XAI + DRIFT)
        with st.status("Analizando tecidos e gerando auditoria visual...", expanded=True) as status:
            # 1. Inferência
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(prob, 0)
            res = CLASSES[idx.item()]
            
            # 2. XAI (Grad-CAM)
            heatmap = generate_gradcam(model, input_tensor, target_layer=model.features[-1])
            
            # 3. Log de Drift (Furtivo)
            log_drift_embedding(model, input_tensor, res)
            
            status.update(label="Análise concluída!", state="complete", expanded=False)

        # Exibição Lado a Lado
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagem Original")
            st.image(img, use_container_width=True)
            
        with col2:
            st.subheader("Auditoria Visual (Grad-CAM)")
            st.image(heatmap, use_container_width=True)
            st.caption("As áreas em vermelho indicam onde a IA focou para o diagnóstico.")

        # Resultados Clínicos
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Diagnóstico", res)
        c2.metric("Confiança", f"{conf.item()*100:.2f}%")

        if idx.item() in [0, 4]:
            st.error("⚠️ ALERTA: Lesão com alta probabilidade de malignidade. Encaminhar para biópsia urgente.")
        else:
            st.success("✅ Nota: Características morfológicas sugestivas de benignidade.")

    else:
        st.info("Aguardando upload de imagem dermatoscópica no menu lateral.")

if __name__ == "__main__":
    main()