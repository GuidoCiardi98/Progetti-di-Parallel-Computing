# GUIDO CIARDI 7090798 FILTRO DI BLOOM SEQUENZIALE

import time
import numpy as np


n = 500000   # Lunghezza dell'array di bit

# Definizione delle funzioni hash:

# Nota: Sono state definite per convenzione 7 funzioni hash (da h0 ad h7) le quali,
# presa una stringa in input, restituiscono un valore compreso tra 0 ed (n - 1)

def h0(string):
    return (len(string) * 48765276) % n


def h1(string):
    sumUnicode = 0

    for i in string:
        sumUnicode = sumUnicode + ord(i)

    return sumUnicode % n


def h2(string):
    productUnicode = 1

    for i in string:
        productUnicode = productUnicode * ord(i)

    return productUnicode % n


def h3(string):
    mixUnicode = 1

    for i in string:
        mixUnicode = (mixUnicode * ord(i)) + (22543 * len(i))

    return mixUnicode % n


def h4(string):
    return ((len(string) + ord(string[0])) * 9876367) % n


def h5(string):
    result = []

    for i in range(0, len(string)):
        result.append(0)

    for j in range(0, len(string)):  # Conversione di ciascun carattere della stringa in Unicode (in result)
        result[j] = ord(string[j])

    vanderMatrix = np.vander(result, n)  # Creazione di una matrice di Vandermonde basata sugli unicode dentro result
    sumElements = 0
    for j in range(0, len(vanderMatrix)):  # Tutti gli elementi della matrice vengono sommati
        for i in range(0, len(vanderMatrix[0])):
            sumElements = sumElements + (vanderMatrix[j][i]) % 10
    return sumElements % n


def h6(string):
    return pow(ord(string[0]), 5) % n


# Funzione di controllo dei match con la parola attualmente considerata:
def matchingCheck(actualString, bloomBitVector):

    if bloomBitVector[h0(actualString)] == 0:  # Casi in cui non si ha un match per la parola attuale
        pass
    elif bloomBitVector[h1(actualString)] == 0:
        pass
    elif bloomBitVector[h2(actualString)] == 0:
        pass
    elif bloomBitVector[h3(actualString)] == 0:
        pass
    elif bloomBitVector[h4(actualString)] == 0:
        pass
    elif bloomBitVector[h5(actualString)] == 0:
        pass
    elif bloomBitVector[h6(actualString)] == 0:
        pass
    else:  # Caso di match trovato per la parola attuale
        print(actualString)


# Inizio codice del programma:

if __name__ == '__main__':

    set_S = []  # Lista di stringhe da considerare nel filtraggio
    extended_set = []  # Lista di stringhe da filtrare
    bloomBitVector = []  # Vettore di bit del filtro

    start_time_prog = time.time()

    # Riempimento del set_S (insieme di m = 100 valori da considerare):
    with open('most_uses_english_words.txt') as file:
        for line in file:
            line = line.replace('\n', '')
            set_S.append(line)

    # Riempimento del extended_set (insieme di valori da filtrare per vedere quali termini sono anche in S):
    with open('parole_da_filtrare.txt') as file:
        for line in file:
            line = line.replace('\n', '')
            extended_set.append(line)

    # Nota: 14 parole di extended_set sono contenute anche in set_S

    start_time = time.time()

    # Inizializzazione del vettore(filtro) di Bloom:
    for i in range(n):
        bloomBitVector.append(0)



    # Si setta ad 1 i bit di bloomBitVector calcolati mediante l'uso delle 7 funzioni hash,
    # considerando ciascuna parola contenuta nel set iniziale di parole considerato (set_S):
    h5time = 0
    for i in range(0, len(set_S)):
        bloomBitVector[h0(set_S[i])] = 1
        bloomBitVector[h1(set_S[i])] = 1
        bloomBitVector[h2(set_S[i])] = 1
        bloomBitVector[h3(set_S[i])] = 1
        bloomBitVector[h4(set_S[i])] = 1
        start_time_h5 = time.time()
        bloomBitVector[h5(set_S[i])] = 1
        end_time_h5 = time.time()
        bloomBitVector[h6(set_S[i])] = 1


        h5time = h5time + (end_time_h5 - start_time_h5)

    end_time = time.time()



    print("Tempo di esecuzione h5: ", h5time)
    print("Tempo di inizializzazione: ", end_time - start_time)

    # Fase di filtraggio per i nuovi elementi e stampe finali per le corrispondenze trovate:

    print("Corrispondenze trovate: ")


    start_time_matching = time.time()
    for i in range(0, len(extended_set)):
        matchingCheck(extended_set[i], bloomBitVector)
    end_time_matching = time.time()

    print("Tempo di matching: ", end_time_matching - start_time_matching)

    end_time_prog = time.time()

    print("Tempo complessivo di esecuzione: ", end_time_prog - start_time_prog)
