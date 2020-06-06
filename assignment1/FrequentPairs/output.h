#ifndef OUTPUT_H
#define OUTPUT_H

/* DO NOT MODIFY THIS FILE */

typedef struct {
    size_t word_index1;
    size_t word_index2;
    int count;
} output_pair;

typedef struct {
    output_pair *pairs;
    size_t cap;
    size_t len;
} output_pairs;

int
alloc_output_pairs(output_pairs *op);

void
free_output_pairs(output_pairs *op);

/** Use this function to output a word pair and its occurrence count. */
void
push_output_pair(output_pairs *op, size_t word_index1, size_t word_index2, int count);

void
print_output_pairs(dataset *ds, output_pairs *op);

#endif /* OUTPUT_H */
