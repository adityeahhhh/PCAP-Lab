#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

void toggle_case(char *word) {
    for (int i = 0; word[i] != '\0'; i++) {
        if (islower(word[i])) {
            word[i] = toupper(word[i]);
        } else if (isupper(word[i])) {
            word[i] = tolower(word[i]);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int MAX_WORD_LENGTH = 100;
    char word[MAX_WORD_LENGTH];
    MPI_Status status;

    if (rank == 0) {
        printf("Enter a word to send to process 1: ");
        scanf("%s", word);

        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Rank 0: Sent word '%s' to rank 1\n", word);

        MPI_Recv(word, MAX_WORD_LENGTH, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &status);
        printf("Rank 0: Received toggled word '%s' from rank 1\n", word);

    } else if (rank == 1) {

        MPI_Recv(word, MAX_WORD_LENGTH, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank 1: Received word '%s' from rank 0\n", word);

        toggle_case(word);

        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        printf("Rank 1: Sent toggled word '%s' back to rank 0\n", word);
    }

    MPI_Finalize();
    return 0;
}
