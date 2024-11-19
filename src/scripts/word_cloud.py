from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import config


def generate_word_cloud():
    """
    Génère et sauvegarde un Word Cloud basé sur les occurrences de mots dans inputs_documents.
    """
    # Calcul des fréquences
    word_frequencies = np.sum(config.INPUTS_DOCUMENTS, axis=0)

    # Création d'un dictionnaire de mots avec leurs fréquences
    word_freq_dict = {word: freq for word, freq in zip(config.VOCAB, word_frequencies)}

    # Generate word_cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)

    # Affichage et sauvegarde du Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    plt.savefig(f"../../plots/{config.ALGORITHM}/{config.PREDICTION_FILENAME}_wordcloud.svg", format='svg')
