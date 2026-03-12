# src/dataset.py
import os
import zipfile
import io
import logging
import pandas as pd
import numpy as np
import h5py
from PIL import Image
from dotenv import load_dotenv

# Configuração de Auditoria (Logs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dicionário padrão do HAM10000 (Mapeamento de Doenças)
LESION_MAP = {
    'nv': 0,    # Melanocytic nevi (Pinta comum)
    'mel': 1,   # Melanoma (Câncer perigoso)
    'bkl': 2,   # Benign keratosis-like lesions
    'bcc': 3,   # Basal cell carcinoma
    'akiec': 4, # Actinic keratoses
    'vasc': 5,  # Vascular lesions
    'df': 6     # Dermatofibroma
}

def process_pipeline():
    load_dotenv()
    
    # 1. Ajuste de Caminhos
    # Lê direto da sua pasta Downloads do Ubuntu
    downloads_zip_path = os.path.expanduser("~/Downloads/archive.zip")
    
    # Salva o arquivo final otimizado dentro da pasta do projeto DERMATO
    processed_dir = os.getenv("PROCESSED_DATA_PATH", "./data/processed")
    hdf5_path = os.path.join(processed_dir, "dataset.h5")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Verifica se já foi processado antes
    if os.path.exists(hdf5_path):
        logging.info("✅ Arquivo HDF5 já existe. Pipeline de dados pulado.")
        return

    # Verifica se o arquivo realmente está nos Downloads
    if not os.path.exists(downloads_zip_path):
        logging.error(f"❌ Arquivo não encontrado em: {downloads_zip_path}")
        logging.info("Verifique se o nome do arquivo é 'archive.zip' e se está na pasta Downloads.")
        return

    logging.info(f"🚀 Lendo {downloads_zip_path} direto da RAM (Zero Extração para o disco)...")
    
    with zipfile.ZipFile(downloads_zip_path, 'r') as z:
        # 2. Encontrar o CSV de metadados dentro do ZIP
        csv_filename = [f for f in z.namelist() if f.endswith('.csv') and 'metadata' in f.lower()]
        if not csv_filename:
            logging.error("❌ Arquivo CSV de metadados não encontrado dentro do ZIP!")
            return
            
        with z.open(csv_filename[0]) as csv_file:
            df = pd.read_csv(csv_file)
            
        label_dict = dict(zip(df['image_id'], df['dx']))
        
        # Filtrar apenas as imagens
        image_files = [f for f in z.namelist() if f.endswith('.jpg')]
        total_images = len(image_files)
        
        # 3. Criar o Banco HDF5 Otimizado
        with h5py.File(hdf5_path, 'w') as h5f:
            img_dataset = h5f.create_dataset("images", shape=(total_images, 224, 224, 3), dtype=np.uint8)
            label_dataset = h5f.create_dataset("labels", shape=(total_images,), dtype=np.int8)
            
            logging.info(f"📦 Convertendo {total_images} imagens para HDF5 (Usando poder do seu i7)...")
            
            processed_count = 0
            # 4. Processamento Streaming (Imagem por Imagem na RAM)
            for file_name in image_files:
                img_id = os.path.basename(file_name).replace('.jpg', '')
                
                if img_id not in label_dict:
                    continue # Pula se a imagem não tiver diagnóstico no CSV
                
                label_num = LESION_MAP[label_dict[img_id]]
                
                # Ler bytes -> Decodificar imagem -> Redimensionar -> Converter para Matriz
                img_bytes = z.read(file_name)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize((224, 224))
                
                # Salvar no HDF5
                img_dataset[processed_count] = np.array(img)
                label_dataset[processed_count] = label_num
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    logging.info(f"🔄 Processado {processed_count}/{total_images} imagens...")

    logging.info(f"🎯 Sucesso! Dataset HDF5 super rápido criado em: {hdf5_path}")
    logging.info("💡 Nota: Seu arquivo 'archive.zip' original foi mantido intacto nos Downloads.")

if __name__ == "__main__":
    process_pipeline()