# tests/test_data.py
import os
import h5py

def test_integridade_hdf5():
    """Garante que a base de dados em cache não está corrompida e tem a estrutura correta."""
    caminho_hdf5 = "./data/processed/dataset.h5"
    
    # 1. Verifica se o ficheiro existe
    assert os.path.exists(caminho_hdf5), "A base de dados dataset.h5 não foi encontrada!"
    
    # 2. Abre a base e verifica os compartimentos internos
    with h5py.File(caminho_hdf5, 'r') as h5f:
        chaves = list(h5f.keys())
        
        # Tem de ter as imagens e os rótulos (labels)
        assert 'images' in chaves, "A base de dados não contém a coluna de imagens!"
        assert 'labels' in chaves, "A base de dados não contém a coluna de diagnósticos (labels)!"
        
        # Pega a primeira imagem para ver se tem o tamanho certo
        primeira_imagem = h5f['images'][0]
        assert primeira_imagem.shape == (224, 224, 3), "As imagens no HDF5 não estão em 224x224 RGB!"