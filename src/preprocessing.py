import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import config

sw_nltk = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def remove_stopwords():
    """
    Filter the document to remove stopwords
    """

    # Remove stopwords
    clean_index = [word for word in range(len(config.VOCAB)) if config.VOCAB[word].lower() in sw_nltk]

    # Filter the inputs
    return np.delete(config.INPUTS_DOCUMENTS, clean_index, axis=1), np.delete(config.TEST_DOCUMENTS, clean_index,
                                                                              axis=1)


def steeming():
    """
    Steeming des mots du vocabulaire
    """
    steemer = PorterStemmer()
    steemed_vocab = [steemer.stem(word.lower()) for word in config.VOCAB]
    filtered_vocab = [word for word in steemed_vocab if word not in sw_nltk]

    # Garder les indices des premiers mots uniques
    unique_vocab, unique_indices = np.unique(filtered_vocab, return_index=True)

    # Réduire les dimensions des matrices d'inputs et de test
    inputs_reduced = config.INPUTS_DOCUMENTS[:, unique_indices]
    test_reduced = config.TEST_DOCUMENTS[:, unique_indices]

    return inputs_reduced, test_reduced


def lemmatise():
    """
    Lemmatisation des mots du vocabulaire
    """
    lemmatized_vocab = [lemmatizer.lemmatize(word.lower()) for word in config.VOCAB]
    filtered_vocab = [word for word in lemmatized_vocab if word not in sw_nltk]

    # # Garder les indices des premiers mots uniques
    unique_vocab, unique_indices = np.unique(filtered_vocab, return_index=True)

    # Réduire les dimensions des matrices d'inputs et de test
    inputs_reduced = config.INPUTS_DOCUMENTS[:, unique_indices]
    test_reduced = config.TEST_DOCUMENTS[:, unique_indices]

    return inputs_reduced, test_reduced


def tf_idf():
    """
    Calculate TF-IDF
    """
    # Calcul de TF-IDF pour les documents d'input
    tf_inputs = config.INPUTS_DOCUMENTS / np.sum(config.INPUTS_DOCUMENTS, axis=1, keepdims=True)
    N_inputs = config.INPUTS_DOCUMENTS.shape[0]
    df_inputs = np.sum(config.INPUTS_DOCUMENTS > 0, axis=0)
    idf_inputs = np.log(N_inputs / (df_inputs + 1))
    tf_idf_inputs = tf_inputs * idf_inputs

    # Calcul de TF-IDF pour les documents de test
    tf_tests = config.TEST_DOCUMENTS / np.sum(config.TEST_DOCUMENTS, axis=1, keepdims=True)
    N_tests = config.TEST_DOCUMENTS.shape[0]
    idf_inputs = np.log(N_tests / (df_inputs + 1))
    tf_idf_tests = tf_tests * idf_inputs

    return tf_idf_inputs, tf_idf_tests


def remove_low_high_frequency(low_threshold, high_threshold, X_train, X_test):
    # Count each word
    occurrence_0 = np.sum(X_train[config.LABELS_DOCUMENTS == 0], axis=0)
    occurrence_1 = np.sum(X_train[config.LABELS_DOCUMENTS == 1], axis=0)

    # Différence des occurrences entre chaque classe
    difference = np.abs(occurrence_0 - occurrence_1)

    # Supprime les mots en dessous du seuil d'apparition
    delete_words_low = [x for x in range(len(difference)) if difference[x] < low_threshold]

    X_train = np.delete(X_train, delete_words_low, axis=1)
    X_test = np.delete(X_test, delete_words_low, axis=1)

    if high_threshold == 0:
        return X_train, X_test

    occurrence_0 = np.sum(X_train[config.LABELS_DOCUMENTS == 0], axis=0)
    occurrence_1 = np.sum(X_train[config.LABELS_DOCUMENTS == 1], axis=0)

    # Différence des occurrences entre chaque classe
    difference = np.abs(occurrence_0 - occurrence_1)

    sorted_index = np.argsort(difference)

    delete_words_high = sorted_index[-high_threshold:]

    # Supprime les mots en dessous du seuil d'apparition
    return np.delete(X_train, delete_words_high, axis=1), np.delete(X_test, delete_words_high, axis=1)
