/* DO NOT MODIFY THIS FILE */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "dataset.h"
#include "output.h"
#include "find_frequent_pairs.h"

#define bad_call(exec) { print_usage(exec); return 1; }

void
print_usage(const char *exec)
{
    printf("Usage: %s (FILE|-) THRESHOLD [METHOD] [NUM_RUNS]\n", exec);
    printf("       FILE can be a filename or stdin, i.e., `-`.\n");
    printf("       THRESHOLD defines the minimal count of a work pair.\n");
    printf("       METHOD can be naive_indexes,\n");
    printf("                     naive_bitmaps,\n");
    printf("                     quick_indexes,\n");
    printf("                     quick_bitmaps.\n");
    printf("       You will implement the `quick_*` methods.\n");
    printf("       NUM_RUNS lets you run the program multiple times for accurate\n");
    printf("       timing.\n");
    printf("\n");
    printf("       EXAMPLE: %s file.dtm.gz 10 quick_bitmaps 5\n", exec);
    printf("       EXAMPLE: gzip -dc file.dtm.gz | %s - 10\n", exec);
}

int
main(int argc, char **args)
{
    if (argc < 3) bad_call(args[0]);

    char *file = args[1];
    int threshold;
    if (sscanf(args[2], "%d", &threshold) != 1) bad_call(args[0]);
    if (threshold < 0)
    {
        fputs("invalid negative threshold", stderr);
        exit(EXIT_FAILURE);
    }

    clock_t begin, end;
    double time_spent;

    puts("loading dataset...");

    begin = clock();
    dataset ds;
    if (load_dataset(file, &ds))
    {
        fputs("loading dataset failed", stderr);
        return EXIT_FAILURE;
    }
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("dataset loaded in %f seconds\n", time_spent);

    void (* find_pairs)(const dataset *, output_pairs *, int) = NULL;
    if (argc >= 4)
    {
        if (strcmp(args[3], "quick_bitmaps") == 0)
        {
            puts("selecting `find_pairs_quick_bitmaps`");
            find_pairs = &find_pairs_quick_bitmaps;
        }
        else if (strcmp(args[3], "quick_indexes") == 0)
        {
            puts("selecting `find_pairs_quick_indexes`");
            find_pairs = &find_pairs_quick_indexes;
        }
        else if (strcmp(args[3], "naive_bitmaps") == 0)
        {
            puts("selecting `find_pairs_naive_bitmaps`");
            find_pairs = &find_pairs_naive_bitmaps;
        }
    }
    if (find_pairs == NULL)
    {
        puts("selecting `find_pairs_naive_indexes`");
        find_pairs = &find_pairs_naive_indexes;
    }

    int num_runs = 1;
    if (argc >= 5)
    {
        sscanf(args[4], "%d", &num_runs);
    }
    if (num_runs <= 0 || num_runs > 1000)
    {
        fputs("overriding invalid negative NUM_RUNS with 1", stderr);
        num_runs = 1;
    }

    printf("running %d time(s)\n", num_runs);

    double time_sum = 0.0;
    for (int run = 1; run <= num_runs; ++run)
    {
        output_pairs op;
        if (alloc_output_pairs(&op) != 0)
        {
            fputs("failed allocating output_pairs", stderr);
            free_dataset(&ds);
            return EXIT_FAILURE;
        }

        printf("#run%d#\n", run);
        begin = clock();
        find_pairs(&ds, &op, threshold);
        end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        time_sum += time_spent;

        print_output_pairs(&ds, &op);
        printf("#time# %f\n", time_spent);
        printf("#npairs# %ld\n", op.len);
        free_output_pairs(&op);
    }

    printf("done in %.3f seconds (average %.4f seconds per run)\n", time_sum,
            time_sum / (double)num_runs);

    free_dataset(&ds);
    return 0;
}
