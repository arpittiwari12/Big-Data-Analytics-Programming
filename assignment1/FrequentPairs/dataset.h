#ifndef DATASET_H
#define DATASET_H

/* DO NOT MODIFY THIS FILE */

#include <stdint.h>

void *
aligned_malloc(int align, size_t size); 

void
aligned_free(void *p);

typedef struct {
    size_t len;
    size_t cap;
    uint64_t *indexes;
} index_list;

int
alloc_index_list(index_list *il);

void
free_index_list(index_list *il);

int
push_to_index_list(index_list *il, uint64_t value);


typedef struct {
    size_t num_documents;   // the number of documents in the corpus
    size_t vocab_size;      // the number of words in the vocabulary
    char *vocab_buffer;     // use `get_word`
    size_t *vocab_offsets;  // use `get_word`
    uint8_t *doc_term_bitmaps; // see function `get_term_bitmap`
    index_list *doc_term_indexes; // see function `get_term_indexes`
} dataset;

/* Useful function for debugging. Prints `nbits` bits starting at pointer `x`. */
void
print_bits(void const *x, int nbits);

/** Load a dataset from file. */
int
load_dataset(char *file, dataset *ds);

/** Free the memory used by `ds`. */
void
free_dataset(dataset *ds);

/** Print the entire vocabulary. */
void
print_vocabulary(const dataset *ds);

/** Get the `voc_index`th vocabulary word as a string.  */
char *
get_word(const dataset *ds, size_t voc_index);

/**
 * Get a pointer to a bit-level structure storing whether a particular word
 * (i.e., the word returned by `get_word(voc_index)`) is present in each
 * document.
 *
 * More specifically, bit `i` is 1 if document `i` contains word `voc_index`.
 * It is 0 otherwise.
 *
 * Properties of this column vector that may be useful:
 *  - columns are zero-padded to multiples of 32 bytes.
 *  - columns start at addressed that are 32-byte aligned.
 */
uint8_t *
get_term_bitmap(const dataset *ds, size_t voc_index);

/** Get the length in number of bytes of a vocabulary bit-column. */
size_t
get_term_bitmap_len(const dataset *ds);

/**
 * Get the number of documents in which a word appears.
 *
 * Access the words using: `index_list->indexes[i]`, with `i` in
 * `0..index_list->len`.
 */
const index_list *
get_term_indexes(const dataset *ds, size_t voc_index);


#endif /* DATASET_H */
