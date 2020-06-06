/* DO NOT MODIFY THIS FILE */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h> // ;-)
#include "dataset.h"

#define INDEX_LIST_DEFAULT_CAP 256
#define INITIAL_VOCAB_BUFFER_SIZE 1024*1024
#define return_err_if(cond, msg) \
    if (cond) { fprintf(stderr, "error: %s\n", msg); return 1; }
#define return_err_free_if(cond) \
    if (cond) { free_dataset(ds); fclose(fh); return 1; }


void * 
aligned_malloc(int align, size_t size) 
{
    /* alignment could not be less then zero */
    if (align < 0) {
        return NULL;
    }

    /* Allocate necessary memory area client request - size parameter -
     * plus area to store the address of the memory returned by standard
     * malloc().
     */
    void *ptr;
    
    /* accomodating atleast one aligned memory area by adding align-1
    /  size of (void *) is added since we intent to keep the below address
    /   in p. This will be retrieved later during free
    */
    void *p = malloc(size + align - 1 + sizeof(void*));

    if (p != NULL) {

        /* Address of the aligned memory according to the align parameter*/
        ptr = (void*) (((size_t)p + sizeof(void*) + align -1) & ~(align-1));

        /* store the address of the malloc() above the beginning of our total memory area.
         */
        *((void**)((size_t)ptr - sizeof(void*))) = p;

        /* Return the address of aligned memory */
        return ptr;
    }
    return NULL;
}

void 
aligned_free(void *p) 
{
    /* Get the address of the memory, stored at the
     * start of our total memory area.
     */
    void *ptr = *((void**)((size_t)p - sizeof(void*)));
    free(ptr);
    return;
}

int
alloc_index_list(index_list *il)
{
    il->len = 0;
    il->cap = 0;
    il->indexes = NULL;

    il->indexes = malloc(INDEX_LIST_DEFAULT_CAP * sizeof(uint64_t));
    return_err_if(il->indexes == NULL, "malloc failed (index_list)");

    il->cap = INDEX_LIST_DEFAULT_CAP;
    return 0;
}

void
free_index_list(index_list *il)
{
    if (il->indexes != NULL)
        free(il->indexes);
    il->indexes = NULL;
}

int
push_to_index_list(index_list *il, uint64_t value)
{
    if (il->len >= il->cap) // realloc
    {
        //printf("realloc for index_list: %lu -> %lu\n", il->cap, il->cap * 2);
        uint64_t *new_indexes = realloc(il->indexes, 2 * il->cap * sizeof(uint64_t));
        return_err_if(new_indexes == NULL, "malloc failed (index_list)");
        il->indexes = new_indexes;
        il->cap *= 2;
    }

    il->indexes[il->len++] = value;
    return 0;
}

void
print_bits(void const *x, int nbits)
{
    unsigned char const *xx = (unsigned char const *) x;
    for (int i = 0; i < nbits; ++i)
    {
        //int j = (nbits / 8) - 1 - (i / 8); // little endian
        int j = i / 8;
        int k = i % 8;
        int b = (xx[j] >> (7-k)) & 0x1;
        if (i > 0 && k == 0) putchar(' ');
        //if (k == 0) printf("(%d)", xx[j]);
        if (b) putchar('1'); else putchar('0');
    }
    putchar('\n');
}

int
load_vocabulary(FILE *fh, dataset *ds)
{
    // allocate character buffer
    size_t buffer_size = INITIAL_VOCAB_BUFFER_SIZE;
    ds->vocab_buffer = calloc(buffer_size, sizeof(char));
    return_err_if(ds->vocab_buffer == NULL, "malloc failed (load_vocabulary, 1)");

    // allocate offsets array
    ds->vocab_offsets = calloc(ds->vocab_size, sizeof(size_t));
    return_err_if(ds->vocab_offsets == NULL, "malloc failed (load_vocabulary, 2)");

    char *buffer = ds->vocab_buffer;
    for (size_t voc_index = 0; voc_index < ds->vocab_size; ++voc_index)
    {
        return_err_if(feof(fh), "early eof (load_vocabulary)");

        size_t buffer_used = buffer - ds->vocab_buffer;
        if (buffer_size - buffer_used <= 100) // realloc vocab buffer
        {
            size_t new_buffer_size = buffer_size * 2;
            printf("info: reallocated buffer (use %ld/%ld -> %ld)\n",
                    buffer_used, buffer_size, new_buffer_size);
            char *new_buffer = realloc(ds->vocab_buffer, new_buffer_size);
            return_err_if(new_buffer == NULL, "realloc failed (load_vocabulary)");
            buffer_size = new_buffer_size;
            ds->vocab_buffer = new_buffer;
            buffer = ds->vocab_buffer + buffer_used;
        }

        fscanf(fh, "%1023s", buffer);
        int len = strlen(buffer);
        //printf("word %ld: %s (%d)\n", voc_index, buffer, len);
        ds->vocab_offsets[voc_index] = buffer_used;
        buffer += (len + 1); // \0 byte
    }

    return 0;
}

int
load_doc_term_matrix(FILE *fh, dataset *ds)
{
    // This constructs two structures at once: the bitmaps and the index lists

    size_t column_size = get_term_bitmap_len(ds);
    size_t nbytes = column_size * ds->vocab_size;

    // Initialize bitmaps
    ds->doc_term_bitmaps = aligned_malloc(32, nbytes); // 256-bit aligned
    return_err_if(ds->doc_term_bitmaps == NULL, "alloc fail (doc_term_bitmaps)");

    { // Initialize bitmaps to zero
        __m256i *ptr = (__m256i *) ds->doc_term_bitmaps;
        for (size_t i = 0; i < nbytes / 32; ++i)
            ptr[i] = _mm256_setzero_si256();
    }

    // Initialize index lists
    ds->doc_term_indexes = calloc(ds->vocab_size, sizeof(index_list));
    return_err_if(ds->doc_term_indexes == NULL, "alloc fail (doc_term_indexes)");
    for (size_t voc_i = 0; voc_i < ds->vocab_size; ++voc_i)
    {
        return_err_if(alloc_index_list(&ds->doc_term_indexes[voc_i]) != 0,
                "alloc fail (doc_term_indexes index_list)");
    }


    uint8_t *ptr = (uint8_t *) ds->doc_term_bitmaps;
    for (size_t doc_i = 0; doc_i < ds->num_documents; ++doc_i)
    {
        printf("\rreading document %ld / %ld", doc_i, ds->num_documents);

        char c = (char) fgetc(fh); // we expect a newline here
        return_err_if(c != '\n', "invalid file format, expected newline)");

        for (size_t voc_i = 0; voc_i < ds->vocab_size; ++voc_i)
        {
            return_err_if(feof(fh), "early eof");

            char c = (char) fgetc(fh);
            return_err_if(c != '0' && c != '1', "invalid file format, 0/1 expected");

            if (c == '1')
            {
                // Set 1 bit in bitmap of word `voc_i`
                size_t byte_i = voc_i * column_size + (doc_i / 8);
                size_t bit_i = 7 - (doc_i % 8);
                ptr[byte_i] |= 0x1 << bit_i;

                // Push index of document to index list of `voc_i`
                push_to_index_list(&ds->doc_term_indexes[voc_i], doc_i);
            }
        }
    }

    //printf("debug print:\n");
    //for (size_t voc_i = 0; voc_i < ds->vocab_size; ++voc_i)
    //{
    //    print_bits(ptr + column_size * voc_i, column_size * 8);
    //}

    return 0;
}

int
load_dataset(char *file, dataset *ds)
{
    ds->num_documents = 0;
    ds->vocab_size = 0;
    ds->vocab_buffer = NULL;
    ds->vocab_offsets = NULL;
    ds->doc_term_bitmaps = NULL;
    ds->doc_term_indexes = NULL;

    FILE *fh;
    if (strcmp(file, "-") == 0)
    {
        fh = stdin;
    }
    else
    {
        fh = fopen(file, "rb");
    }
    if (fh == NULL) return 1;

    // read number of documents and size of vocabulary
    fscanf(fh, "%zu\n", &ds->num_documents);
    fscanf(fh, "%zu\n", &ds->vocab_size);

    return_err_free_if(ds->num_documents <= 0 || ds->vocab_size <= 0);

    //printf("debug: num_documents = %ld\n", ds->num_documents);
    //printf("debug: vocab_size = %ld\n", ds->vocab_size);

    return_err_free_if(load_vocabulary(fh, ds) != 0);
    return_err_free_if(load_doc_term_matrix(fh, ds) != 0);

    fclose(fh);
    return 0; // all OK! structure in `ds`
}

void
free_dataset(dataset *ds)
{
    if (ds->vocab_offsets != NULL)
        free(ds->vocab_offsets);
    if (ds->vocab_buffer != NULL)
        free(ds->vocab_buffer);
    if (ds->doc_term_bitmaps != NULL)
        aligned_free(ds->doc_term_bitmaps);

    if (ds->doc_term_indexes != NULL)
    {
        for (size_t i = 0; i < ds->vocab_size; ++i)
            free_index_list(&ds->doc_term_indexes[i]);
        free(ds->doc_term_indexes);
    }

    ds->vocab_offsets = NULL;
    ds->vocab_buffer = NULL;
    ds->doc_term_bitmaps = NULL;
    ds->doc_term_indexes = NULL;
}

void
print_vocabulary(const dataset *ds)
{
    // really only to check whether the memory layout is as it should be
    size_t i = 0;
    size_t n0 = 0;
    while (n0 < ds->vocab_size)
    {
        char c = ds->vocab_buffer[i++];
        if (c == '\0')
        {
            n0++;
            printf("\n");
        }
        else
        {
            printf("%c", c);
        }
    }
}

char *
get_word(const dataset *ds, size_t voc_index)
{
    if (voc_index >= ds->vocab_size)
    {
        printf("error: out of bounds %ld/%ld\n", voc_index, ds->vocab_size);
        exit(EXIT_FAILURE);
    }

    size_t offset = ds->vocab_offsets[voc_index];
    return ds->vocab_buffer + offset;
}

uint8_t *
get_term_bitmap(const dataset *ds, size_t voc_index)
{
    if (voc_index >= ds->vocab_size)
    {
        printf("error: voc_index out of bounds %ld/%ld\n", voc_index, ds->vocab_size);
        exit(EXIT_FAILURE);
    }

    size_t column_size = get_term_bitmap_len(ds);
    return ds->doc_term_bitmaps + (column_size * voc_index);
}

size_t
get_term_bitmap_len(const dataset *ds)
{ 
    // num of bytes ceiled to 32
    size_t column_size = ((ds->num_documents / 256) + 1) * 32;
    return column_size;
}

const index_list *
get_term_indexes(const dataset *ds, size_t voc_index)
{
    return &ds->doc_term_indexes[voc_index];
}
