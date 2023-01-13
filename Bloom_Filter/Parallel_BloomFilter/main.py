# GUIDO CIARDI 7090798 FILTRO DI BLOOM PARALLELO

import time
from joblib import Parallel, delayed
import numpy as np

n = 500000   # Lunghezza dell'array di bit

numThreads = 10  # Numero di processi

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
    # Definiamo parallelamente la lista result che contiene i valori unicode dei caratteri di string
    result = Parallel(n_jobs=numThreads)(delayed(ord)(string[i]) for i in range(0, len(string)))

    vanderMatrix = np.vander(result, n)  # Creazione di una matrice di Vandermonde basata sugli unicode dentro result
    sumElements = 0
    for j in range(0, len(vanderMatrix)):   # Tutti gli elementi della matrice vengono sommati
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



    start_time = time.time()

    # Inizializzazione del vettore(filtro) di Bloom:
    for i in range(n):
        bloomBitVector.append(0)


    # Nota: 14 parole di extended_set sono contenute anche in set_S

    # Definizione componenti di parallelizzazione:
    parallel_set0 = Parallel(n_jobs=numThreads)(delayed(h0)(set_S[i]) for i in range(0, len(set_S)))
    parallel_set1 = Parallel(n_jobs=numThreads)(delayed(h1)(set_S[i]) for i in range(0, len(set_S)))
    parallel_set2 = Parallel(n_jobs=numThreads)(delayed(h2)(set_S[i]) for i in range(0, len(set_S)))
    parallel_set3 = Parallel(n_jobs=numThreads)(delayed(h3)(set_S[i]) for i in range(0, len(set_S)))
    parallel_set4 = Parallel(n_jobs=numThreads)(delayed(h4)(set_S[i]) for i in range(0, len(set_S)))
    start_time_h5 = time.time()
    parallel_set5 = Parallel(n_jobs=numThreads)(delayed(h5)(set_S[i]) for i in range(0, len(set_S)))
    end_time_h5 = time.time()
    parallel_set6 = Parallel(n_jobs=numThreads)(delayed(h6)(set_S[i]) for i in range(0, len(set_S)))

    # Si setta ad 1 i bit di bloomBitVector calcolati mediante l'uso delle 7 funzioni hash,
    # considerando ciascuna parola contenuta nel set iniziale di parole considerato (set_S):
    for i in range(0, len(set_S)):
        bloomBitVector[parallel_set0[i]] = 1
        bloomBitVector[parallel_set1[i]] = 1
        bloomBitVector[parallel_set2[i]] = 1
        bloomBitVector[parallel_set3[i]] = 1
        bloomBitVector[parallel_set4[i]] = 1
        bloomBitVector[parallel_set5[i]] = 1
        bloomBitVector[parallel_set6[i]] = 1

    end_time = time.time()

    # Oppure prova: h5time = (end_time_h5b - start_time_h5b)
    print("Tempo di esecuzione h5: ", end_time_h5 - start_time_h5)
    print("Tempo di inizializzazione: ", end_time - start_time)

    # Fase di filtraggio parallelizzata per i nuovi elementi:

    print("Corrispondenze trovate: ")

    start_time_matching = time.time()
    Parallel(n_jobs=numThreads)(delayed(matchingCheck)(extended_set[i], bloomBitVector) for i in range(0, len(extended_set)))
    end_time_matching = time.time()
    print("Tempo di matching: ", end_time_matching - start_time_matching)

    end_time_prog = time.time()

    print("Tempo complessivo di esecuzione: ", end_time_prog - start_time_prog)
