/* DO NOT MODIFY THIS FILE */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "dataset.h"
#include "output.h"

#define OUTPUT_PAIRS_INITIAL_CAP 4096

int
alloc_output_pairs(output_pairs *op)
{
    op->pairs = NULL;
    op->cap = 0;
    op->len = 0;

    op->pairs = malloc(sizeof(output_pair) * OUTPUT_PAIRS_INITIAL_CAP);
    if (op->pairs == NULL)
    {
        puts("error: alloc failed (output_pairs)");
        return 1;
    }

    op->cap = OUTPUT_PAIRS_INITIAL_CAP;
    op->len = 0;

    return 0;
}

void
free_output_pairs(output_pairs *op)
{
    if (op->pairs != NULL)
        free(op->pairs);
}

void
push_output_pair(output_pairs *op, size_t word_index1, size_t word_index2, int count)
{
    if (op->len >= op->cap)
    {
        output_pair *new_pairs = realloc(op->pairs, 2 * op->cap * sizeof(output_pair));
        if (new_pairs == NULL)
        {
            puts("error: realloc failed (output_pairs)");
            exit(EXIT_FAILURE);
        }
        op->pairs = new_pairs;
        op->cap *= 2;
    }

    output_pair p =  { word_index1, word_index2, count };
    op->pairs[op->len++] = p;
}

int
output_pair_cmp(const void *a, const void *b)
{
    const output_pair *p1 = (const output_pair *) a;
    const output_pair *p2 = (const output_pair *) b;
    if (p1->count > p2->count)
        return -1;
    if (p1->count < p2->count)
        return 1;
    if (p1->word_index1 > p2->word_index1)
        return 1;
    if (p1->word_index1 < p2->word_index1)
        return -1;
    if (p1->word_index2 > p2->word_index2)
        return 1;
    if (p1->word_index2 < p2->word_index2)
        return -1;
    return 0;
}

void
print_output_pairs(dataset *ds, output_pairs *op)
{
    // rearrange words such that the two words in the pair are sorted alphabetically
    for (size_t i = 0; i < op->len; ++i)
    {
        output_pair *p = &op->pairs[i];
        char *word1 = get_word(ds, p->word_index1);
        char *word2 = get_word(ds, p->word_index2);
        int cmp = strcmp(word1, word2);
        if (cmp > 0)
        {
            size_t tmp = p->word_index1;
            p->word_index1 = p->word_index2;
            p->word_index2 = tmp;
        }
        else if (cmp == 0)
        {
            fprintf(stderr, "error: pair with same word `%s`\n", word1);
            exit(EXIT_FAILURE);
        }
    }

    // sort word pairs
    qsort(op->pairs, op->len, sizeof(output_pair), &output_pair_cmp);

    for (size_t i = 0; i < op->len; ++i)
    {
        output_pair *p = &op->pairs[i];
        char *word1 = get_word(ds, p->word_index1);
        char *word2 = get_word(ds, p->word_index2);

        printf("#pair# %s %s %d\n", word1, word2, p->count);
    }
}
