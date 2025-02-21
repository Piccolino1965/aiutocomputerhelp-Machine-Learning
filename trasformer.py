###########################################################
# trasformer
# Giovanni Popolizio 2025 www.aiutocomputerhelp.it
###########################################################
import numpy as np
import matplotlib.pyplot as plt
# Definizione del meccanismo di attenzione
def attenzione(q, k, v):
    """
    Implementa un meccanismo di attenzione semplice.
    q: query
    k: key
    v: value
    """
    scores = np.dot(q, k.T) / np.sqrt(k.shape[-1])  # Scaled dot-product
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Softmax
    return np.dot(weights, v), weights
# Embedding semplice
def embedding(tokens, vocab_size, embedding_dim):
    """
    Trasforma una sequenza di token in embedding.
    """
    np.random.seed(42)
    embeddings = np.random.rand(vocab_size, embedding_dim)
    return np.array([embeddings[token] for token in tokens])
# Transformer Encoder Layer
def transformer_encoder(tokens, vocab_size, embedding_dim, num_heads):
    """
    Implementa un layer encoder di un Transformer.
    """
    # Creazione degli embedding
    token_embeds = embedding(tokens, vocab_size, embedding_dim)
    # Multi-head Attention simulato (1 head per semplicità)
    attn_output, attn_weights = attenzione(token_embeds, token_embeds, token_embeds)
    # Passaggio attraverso una feedforward
    feedforward = np.maximum(0, np.dot(attn_output, np.random.rand(embedding_dim, embedding_dim)))
    return feedforward, attn_weights
# Visualizzazione delle matrici di attenzione
def visualizza_matrice_attn(attn_weights, sentence, vocab):
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_weights, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(sentence)), [vocab[token] for token in sentence], rotation=45)
    plt.yticks(range(len(sentence)), [vocab[token] for token in sentence])
    plt.title("Matrice di Attenzione")
    plt.show()
# Esempio di dataset
dataset = ["amo il machine learning", "i transformer sono potenti", "deep learning è da studiare", "la privacy vale tanto","Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura ché la diritta via era smarrita"]
# Creazione di un vocabolario
vocab = {word: idx for idx, word in enumerate(set(" ".join(dataset).split()))}
vocab_inverso = {idx: word for word, idx in vocab.items()}
vocab_size = len(vocab)
# Converti frasi in token
tokens = [[vocab[word] for word in sentence.split()] for sentence in dataset]
# Configurazione dei parametri
embedding_dim = 16
num_heads = 1
# Applicazione del Transformer Encoder
for idx, sentence in enumerate(tokens):
    output, attn_weights = transformer_encoder(sentence, vocab_size, embedding_dim, num_heads)
    print(f"Output del Transformer Encoder per la frase {idx + 1}:\n", output)
    # Visualizzazione della matrice di attenzione
    visualizza_matrice_attn(attn_weights, sentence, vocab_inverso)
