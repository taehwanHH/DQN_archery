import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()

trellis = eng.poly2trellis(matlab.double([5, 4]), matlab.double([[23, 35, 0], [0, 5, 13]]))  # R = 2 / 3


def convenc(msg_bits, trellis):
    # Initialize the shift registers to zero
    sr = np.zeros(int(trellis['numInputSymbols'] - 1), dtype=int)

    # Append zeros to message bits to flush the shift registers
    msg_bits_padded = np.append(msg_bits, np.zeros(int(trellis['numInputSymbols']) - 1))

    # Convolutional encoding
    tx_bits = np.zeros(int(len(msg_bits) * trellis['numOutputSymbols']), dtype=int)
    k = 0
    for i in range(len(msg_bits)):
        s_current = np.dot(sr, 2 ** np.arange(len(sr)))
        tx_bits[k:k + int(trellis['numOutputSymbols'])] = trellis['outputs'][s_current, :]
        sr = np.roll(sr, 1)
        sr[0] = msg_bits_padded[i]
        k += int(trellis['numOutputSymbols'])

    return tx_bits


tx_bits = np.random.randint(2, size=(10, 1))  # data bits for one block
tx_bits_enc = convenc(tx_bits, trellis)
print(tx_bits)
print(tx_bits_enc)
