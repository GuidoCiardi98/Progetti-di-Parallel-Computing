#include <stdio.h>
#include <math.h>
#include <omp.h>

#define numPoints 10000 // Numero di punti coinvolti nel clustering
#define numClusters 200 // Numero di clusters da utilizzare
#define nThreads 8 // Numero di threads utilizzati

typedef struct{
    float x;
    float y;
    int clusterId;
}point;


point point_Initialization(float x, float y);
void k_means(point points[], point clustesr[]);
int euclideanDistance(point a, point b);
int nearestCentroid(point actualPoint, point clusters[]);


int main() {


    // Generazione di punti e clusters iniziali:
    point points[numPoints]; // Vettore dei punti
    point clusters[numClusters]; // Vettore dei clusters

    // Dichiarazione variabili relative alla misurazione dei tempi
    double start_Time = 0;
    double end_Time = 0;
    double execution_Time = 0;


    start_Time = omp_get_wtime(); // Inizio della misurazione del tempo di esecuzione

#pragma omp parallel num_threads(nThreads)
    {
#pragma omp for schedule(auto) nowait
        for(int i = 0; i < numPoints; i++){
            points[i] = point_Initialization(i, numPoints - i);
            //points[i] = point_Initialization(i, pow(i,2));  //Per visualizzare eventualmente anche clusterID nei centroidi, qualora si volesse
        }
        // Si utilizza nowait perchè i due cicli sono indipendenti e non serve la barriera dopo il primo for
        // La schedule(auto) decide se schedulare/distribuire i valori tra i thread in modo statico o dinamico a seconda delle esigenze

#pragma omp for schedule(auto)
        for(int i = 0; i < numClusters; i++){
            clusters[i] = point_Initialization((i * numClusters) % (numPoints + 1),  (numPoints + i * numClusters) % (numPoints - 2));
            clusters[i].clusterId = i;
        }
    }


    k_means(points, clusters);


    // STAMPA DEI RISULTATI

    printf("\nESECUZIONE DELL'ALGORITMO K-MEANS SEQUENZIALE\n\n\n");
    printf("Utilizzando K = %d clusters vengono definiti i seguenti punti:\n\n", numClusters);

    for(int i = 0; i < numClusters; i++){
        printf("\n\nCluster numero: %d\n", i);

        printf("Lista dei punti associati al centroide (%f, %f):\n", clusters[i].x, clusters[i].y);

#pragma omp parallel for schedule(auto) num_threads(nThreads)
        for(int point = 0; point < numPoints; point++) {
            if(points[point].clusterId == i){
                printf("(%f, %f)  ", points[point].x, points[point].y);
            }
        }
    }


    end_Time = omp_get_wtime(); // Misurazione del tempo finale di esecuzione

    execution_Time = end_Time - start_Time; // Tempo complessivo di esecuzione

    printf("\n\nTempo di esecuzione dell'algoritmo: %f", execution_Time);

    return 0;

}


point point_Initialization(float x, float y){
    point p = {x, y, -1};
    return p;
}


void k_means(point points[], point clusters[]){
    int centroidIndex = 0; // Valore del centroide più vicinio al punto attuale (si usa per la selezione del cluster)

    float newX;
    float newY;
    int numPointsOfCluster = 0;
    int convergence = 0; // Flag di verifica per la convergenza dell'algoritmo
    int iterazioni = 0;


    while(convergence == 0){ // Condizione di convergenza

        convergence = 1;


        // Calcolo del cluster di appartenenza per ciascuno dei punti a disposizione
#pragma omp parallel num_threads(nThreads)
        {
#pragma omp for schedule(auto) private(centroidIndex)
            for (int i = 0; i < numPoints; i++) { // Scansione dei punti

                centroidIndex = nearestCentroid(points[i], clusters);

                points[i].clusterId = centroidIndex; // Il punto "i" viene associato al cluster "centroidIndex"

            }


            // Aggiornamento delle coordinate dei centroidi
#pragma omp for schedule(auto) private(newX, newY)
            for(int centroid = 0; centroid < numClusters; centroid++){
                newX = 0;
                newY = 0;
                numPointsOfCluster = 0;


                for(int i = 0; i < numPoints; i++){
                    if(points[i].clusterId == centroid){
                        newX = newX + points[i].x;
                        newY = newY + points[i].y;
                        numPointsOfCluster++;
                    }

                }

                if(numPointsOfCluster != 0){
                    newX = newX / numPointsOfCluster;
                    newY = newY / numPointsOfCluster;
                }
                else{
                    newX = clusters[centroid].x;
                    newY = clusters[centroid].y;
                }

                if(newX != clusters[centroid].x || newY != clusters[centroid].y){
                    clusters[centroid].x = newX;
                    clusters[centroid].y = newY;
                    convergence = 0;
                }
            }
        }


    }


}



int euclideanDistance(point a, point b){
    return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
}

int nearestCentroid(point actualPoint, point clusters[]){
    double distance;
    int centroidIndex;

    // Confronto del punto col primo centroide per inizializzare la distanza minima
    double minDistance = euclideanDistance(actualPoint, clusters[0]);
    // minDistance viene usata per mantenersi la distanza minore ed individuare il cluster di
    // appartenenza di un determinato punto


    for (int centroid = 1; centroid < numClusters; centroid++) { // Scansione dei centroidi
        distance = euclideanDistance(actualPoint, clusters[centroid]); // Calcolo della distanza Euclidea

        if (distance < minDistance) {
            minDistance = distance;
            centroidIndex = centroid;
        }

    }

    return centroidIndex;
}