# extrair_features_hog.py

import sys
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import io, color, exposure

# --- Configuração ---
# O nome do arquivo de imagem que você deve adicionar ao repositório
IMAGE_FILENAME = 'carmel.jpg'
RESULT_FILENAME = 'resultado_hog.png'
# --------------------

def extrair_e_visualizar_hog(image_path):
    """
    Carrega uma imagem, extrai as features HOG e salva uma visualização.
    """
    try:
        # Carregar a imagem
        image = io.imread(image_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{image_path}' não encontrado.")
        print("Por favor, baixe uma imagem de sua escolha (ex: de sites como Pexels ou Unsplash)")
        print(f"e salve-a no mesmo diretório deste script com o nome '{IMAGE_FILENAME}'.")
        return

    # Converter para escala de cinza
    # O HOG é geralmente calculado em imagens de um único canal (escala de cinza).
    image_gray = color.rgb2gray(image)
    
    print(f"Imagem original carregada: {image.shape}")
    print(f"Imagem convertida para escala de cinza: {image_gray.shape}")

    # Calcular as features HOG
    # Parâmetros baseados na documentação da scikit-image:
    # orientations: Número de "bins" (caixas) para o histograma de gradientes.
    # pixels_per_cell: O tamanho (em pixels) de cada célula.
    # cells_per_block: O número de células em cada bloco (para normalização).
    # visualize=True: Retorna também a imagem HOG para visualização.
    fd, hog_image = hog(image_gray, 
                        orientations=9, 
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), 
                        visualize=True,
                        channel_axis=None) # 'channel_axis=None' pois já está em cinza

    # Plotar os resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # Imagem Original
    ax1.axis('off')
    ax1.imshow(image) # Usamos a imagem colorida original para melhor comparação
    ax1.set_title('Imagem Original')

    # Imagem HOG
    # Re-escalar a imagem HOG para melhor visualização, pois os valores são pequenos
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Visualização HOG')
    
    plt.suptitle('Extração de Features HOG (Histogram of Oriented Gradients)')
    plt.tight_layout()
    
    # Salvar a imagem de resultado
    plt.savefig(RESULT_FILENAME)
    print(f"\nImagem de resultado salva como '{RESULT_FILENAME}'")
    
    # Mostrar o gráfico na tela
    plt.show()

    print(f"Dimensões do vetor de features HOG (fd): {fd.shape}")
    print(f"Número total de features HOG extraídas: {fd.size}")

if __name__ == "__main__":
    extrair_e_visualizar_hog(IMAGE_FILENAME)