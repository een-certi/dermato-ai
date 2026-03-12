# src/app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import time

# Garante que o Python encontre o explain.py na mesma pasta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from explain import AuditorIA_GradCAM, criar_imagem_auditada

# 1. CONFIGURAÇÃO DA PÁGINA E CSS CUSTOMIZADO
st.set_page_config(page_title="DERMATO AI | Triagem", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Esconde o menu sanduíche e o rodapé do Streamlit para um visual mais limpo */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Melhora o visual das métricas */
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    </style>
""", unsafe_allow_html=True)

# 2. DICIONÁRIO CLÍNICO (Com Níveis de Risco)
CLASSES_DOENCA = {
    0: {'nome': 'Nevo Melanocítico (Pinta comum)', 'risco': 'Baixo', 'cor': 'verde'},
    1: {'nome': 'Melanoma', 'risco': 'Alto', 'cor': 'vermelho'},
    2: {'nome': 'Ceratose Benigna', 'risco': 'Baixo', 'cor': 'verde'},
    3: {'nome': 'Carcinoma Basocelular', 'risco': 'Moderado', 'cor': 'amarelo'},
    4: {'nome': 'Ceratose Actínica', 'risco': 'Moderado', 'cor': 'amarelo'},
    5: {'nome': 'Lesão Vascular', 'risco': 'Baixo', 'cor': 'verde'},
    6: {'nome': 'Dermatofibroma', 'risco': 'Baixo', 'cor': 'verde'}
}

@st.cache_resource
def carregar_modelo_ia():
    modelo = models.mobilenet_v3_large(weights=None)
    in_features = modelo.classifier[3].in_features
    modelo.classifier[3] = nn.Linear(in_features, 7)
    
    caminho_pesos = "./models/checkpoints/mobilenet_v3_dermato.pth"
    if os.path.exists(caminho_pesos):
        modelo.load_state_dict(torch.load(caminho_pesos, map_location=torch.device('cpu')))
        modelo.eval()
        return modelo
    return None

# 3. BARRA LATERAL (Governança e MLOps)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063188.png", width=80) # Ícone médico genérico
    st.title("Governança MLOps")
    st.markdown("---")
    st.info("💻 **Infraestrutura:** Servidor Local (i7 CPU)")
    st.success("🛡️ **Auditoria:** Grad-CAM Ativo")
    st.text("📦 Modelo: MobileNetV3 (v1.0)")
    st.text("🗄️ Cache HDF5: Otimizado")
    st.markdown("---")
    st.caption("Sistema de Apoio à Decisão Clínica. Não substitui o diagnóstico médico especializado.")

# 4. ÁREA PRINCIPAL
st.title("🏥 DERMATO AI - Triagem Oncológica")
st.markdown("Faça o upload da imagem dermatoscópica. O sistema processará o risco e fornecerá a auditoria visual.")

modelo = carregar_modelo_ia()

if modelo is None:
    st.error("❌ **Cérebro da IA não encontrado!** Verifique se o treinamento foi concluído em `models/checkpoints/`.")
    st.stop()

# Caixa de Upload elegante
arquivo_upload = st.file_uploader("📥 Arraste a foto da lesão aqui (JPG/PNG)", type=["jpg", "jpeg", "png"])

if arquivo_upload is not None:
    imagem_pil = Image.open(arquivo_upload).convert('RGB')
    
    transformacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    imagem_tensor = transformacao(imagem_pil).unsqueeze(0)
    
    st.markdown("---") # Linha divisória
    
    # Efeito de carregamento mais profissional
    with st.status("Analizando tecidos e gerando auditoria visual...", expanded=True) as status:
        st.write("Extraindo características morfológicas...")
        time.sleep(0.5) # Pausa rápida para a interface respirar
        
        # Inferência
        st.write("Calculando probabilidades...")
        saida = modelo(imagem_tensor)
        probabilidades = torch.nn.functional.softmax(saida[0], dim=0)
        classe_predita = torch.argmax(probabilidades).item()
        confianca = probabilidades[classe_predita].item() * 100
        dados_doenca = CLASSES_DOENCA[classe_predita]
        
        # Auditoria (Grad-CAM)
        st.write("Mapeando áreas de atenção (Grad-CAM)...")
        camada_alvo = modelo.features[-1]
        auditor = AuditorIA_GradCAM(modelo, camada_alvo)
        mapa_calor = auditor.generate_heatmap(imagem_tensor, classe_predita)
        
        caminho_temp = "temp_upload.jpg"
        imagem_pil.save(caminho_temp)
        imagem_auditada = criar_imagem_auditada(caminho_temp, mapa_calor)
        os.remove(caminho_temp)
        
        status.update(label="Análise concluída com sucesso!", state="complete", expanded=False)

    # 5. EXIBIÇÃO DE RESULTADOS (Layout em 2 colunas principais)
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        # Usando Abas (Tabs) para não poluir a tela
        aba_original, aba_auditoria = st.tabs(["📸 Imagem Original", "🔍 Auditoria de IA"])
        with aba_original:
            st.image(imagem_pil, use_container_width=True)
        with aba_auditoria:
            st.image(imagem_auditada, use_container_width=True)
            st.caption("🔴 **Zonas quentes:** Áreas da pele que mais influenciaram a IA.")

    with col2:
        st.subheader("📋 Relatório Clínico")
        
        # Sistema de Cores por Risco Clínico
        if dados_doenca['risco'] == 'Alto':
            st.error(f"🚨 **Diagnóstico Sugerido:** {dados_doenca['nome']}")
        elif dados_doenca['risco'] == 'Moderado':
            st.warning(f"⚠️ **Diagnóstico Sugerido:** {dados_doenca['nome']}")
        else:
            st.success(f"✅ **Diagnóstico Sugerido:** {dados_doenca['nome']}")
            
        # Métricas lado a lado
        m1, m2 = st.columns(2)
        with m1:
            st.metric(label="Nível de Risco", value=dados_doenca['risco'])
        with m2:
            st.metric(label="Confiança do Modelo", value=f"{confianca:.1f}%")
        
        st.markdown("### Justificativa de Governança")
        st.info(
            "O mapa de calor (Grad-CAM) na aba de 'Auditoria' comprova a rastreabilidade da decisão. "
            "Se as zonas vermelhas corresponderem à lesão, o modelo atuou conforme o esperado. "
            "Se apontarem para bordas ou ruídos, sugere-se revisão médica imediata."
        )