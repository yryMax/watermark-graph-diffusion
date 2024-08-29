
import numpy as np
import torch
from scipy.stats import binom
import networkx as nx
import matplotlib.pyplot as plt
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

p = 0.0844
prob = np.load('../model/node_dist.npy')
node_dist =  torch.distributions.Categorical(torch.tensor(prob))


def ciphertext_from_message(message, key=None, nonce=None, n=None):
    if key is None:
        key = os.urandom(32)
    if nonce is None:
        nonce = os.urandom(16)

    if isinstance(key, str):
        key = bytes.fromhex(key)
    if isinstance(nonce, str):
        nonce = bytes.fromhex(nonce)
    if n is None:
        n = node_dist.sample()
    bytes_length = n // 8

    # cut message to if it's too long
    if len(message) > bytes_length:
        message = message[:bytes_length]

    message_bytes = message.encode() + b'\0' * (bytes_length - len(message.encode()))
    cipher = Cipher(algorithms.ChaCha20(key, nonce), mode=None, backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(message_bytes) + encryptor.finalize()
    ciphertext_bits = np.unpackbits(np.frombuffer(ciphertext, dtype=np.uint8))
    #  print(ciphertext_bits)
    bit_to_randomize = n - len(ciphertext_bits)
    assert bit_to_randomize >= 0 and bit_to_randomize < 8
    ciphertext_bits = np.concatenate([ciphertext_bits, np.random.randint(0, 2, (bit_to_randomize,))])
    assert len(ciphertext_bits) == n
    return ciphertext_bits, key, nonce


def binom_pesudo_sampler(ciphertext_bits , p):
    n = len(ciphertext_bits)
    rv = binom(n-2, p)
    samples = []
    for i in range(n):
        u = (ciphertext_bits[i] + np.random.uniform(0, 1))/2
        assert u >= 0 and u <= 1
        samples.append(rv.ppf(u))
    return samples


def ER_pesudo_sampler(ciphertext_bits, p):
    n = len(ciphertext_bits)

    offset_dist = binom(1, p)
    while 1:
        d = binom_pesudo_sampler(ciphertext_bits, p)
        offset = offset_dist.rvs()
        degrees = [d[i] + offset for i in range(n)]
        if sum(degrees) % 2 == 0:
            while 1:
                try:
                    #    print(degrees)
                    G = nx.random_degree_sequence_graph(degrees)
                    degrees_from_graph = [G.degree(node) for node in G.nodes()]
                    assert degrees == degrees_from_graph
                    return G, offset
                except nx.exception.NetworkXError:
                    print('no graph')

def watermark_embedding(message, n = None, key=None, nonce=None):
    c,k,n = ciphertext_from_message(message, key, nonce, n)
    G,offset = ER_pesudo_sampler(c, p)
    return G, k, n, offset


def watermark_detection(G, offset, key, nonce, message):

    n = len(G.nodes())
    ciphertext_bits_true ,_,_ = ciphertext_from_message(message, key, nonce, n)
    rv = binom(n-2, p)
    degrees = [G.degree(node) - offset for node in G.nodes()]
    ciphertext_bits_recovered = []
    for i in range(n):
        u = rv.cdf(degrees[i])
        if u <=0.5:
            ciphertext_bits_recovered.append(0)
        else:
            ciphertext_bits_recovered.append(1)
    assert len(ciphertext_bits_true) == len(ciphertext_bits_recovered)
    usefull_bits = (n // 8) * 8
    ciphertext_bits_true = ciphertext_bits_true[:usefull_bits]
    ciphertext_bits_recovered = ciphertext_bits_recovered[:usefull_bits]
    return np.sum(np.abs(ciphertext_bits_true - ciphertext_bits_recovered)) / usefull_bits


if __name__ == '__main__':
    message = "A"
    G, key, nonce, offset = watermark_embedding(message)
    print(watermark_detection(G, offset, key, nonce, message))

