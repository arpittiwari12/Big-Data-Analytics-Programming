#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#include "dataset.h"
#include "output.h"
#include "find_frequent_pairs.h"

int
document_has_word(const dataset *ds, size_t doc_index, size_t voc_index)
{
    // Auxiliary function for `find_pairs_naive_bitmaps`

    uint8_t *column_ptr = get_term_bitmap(ds, voc_index);
    if (doc_index >= ds->num_documents)
    {
        printf("error: doc_index out of bounds %ld/%ld\n", doc_index, ds->num_documents);
        exit(1);
    }
    size_t byte_i = doc_index / 8;
    size_t bit_i = 7 - (doc_index % 8);
    uint8_t b = column_ptr[byte_i];
    return ((b >> bit_i) & 0x1) ? 1 : 0;
}

int
find_frequency(int n, uint64_t *a , uint64_t *b, uint64_t *c)
{
    int i = 0 , cnt = 0, n8 = (n/4) * 4;
    for (; i < n8 ; i += 4)
        {
            __m256i x = _mm256_loadu_si256 (( __m256i *)&a[i]);
            __m256i y = _mm256_loadu_si256 (( __m256i *)&b[i]);
            __m256i z = _mm256_and_si256 (x , y);
            _mm256_storeu_si256 (( __m256i *)&c[i] , z);
            for (int j=i; j<4+i; ++j)
            {
                cnt = cnt + _mm_popcnt_u64(c[j]);
            }
        }
    for (; i < n; ++i)
    {
        c[i] = a[i] & b[i];
        cnt = cnt + _mm_popcnt_u64(c[i]);
    }
    return cnt;
}

void
find_pairs_naive_bitmaps(const dataset *ds, output_pairs *op, int threshold)
{
    // This is an example implementation. You don't need to change this, you
    // should implement `find_pairs_quick_*`.

    for (size_t t1 = 0; t1 < ds->vocab_size; ++t1)
    {
        for (size_t t2 = t1+1; t2 < ds->vocab_size; ++t2)
        {
            int count = 0;
            for (size_t d = 0; d < ds->num_documents; ++d)
            {
                int term1_appears_in_doc = document_has_word(ds, d, t1);
                int term2_appears_in_doc = document_has_word(ds, d, t2);
                if (term1_appears_in_doc && term2_appears_in_doc)
                {
                    ++count;
                }
            }
            if (count >= threshold)
                push_output_pair(op, t1, t2, count);
        }
    }
}

void
find_pairs_naive_indexes(const dataset *ds, output_pairs *op, int threshold)
{
    // This is an example implementation. You don't need to change this, you
    // should implement `find_pairs_quick_*`.

    for (size_t t1 = 0; t1 < ds->vocab_size; ++t1)
    {
        const index_list *il1 = get_term_indexes(ds, t1);
        for (size_t t2 = t1+1; t2 < ds->vocab_size; ++t2)
        {
            const index_list *il2 = get_term_indexes(ds, t2);
            int count = 0;
            size_t i1 = 0, i2 = 0;
            for (; i1 < il1->len && i2 < il2->len;)
            {
                size_t x1 = il1->indexes[i1], x2 = il2->indexes[i2];
                if (x1 == x2) { ++count; ++i1; ++i2; }
                else if (x1 < x2) { ++i1; }
                else { ++i2; }
            }
            if (count >= threshold)
                push_output_pair(op, t1, t2, count);
        }
    }
}

void
find_pairs_quick_bitmaps(const dataset *ds, output_pairs *op, int threshold)
{
    // TODO implement a quick `find_pairs_quick_bitmaps` procedure using
    // `get_term_bitmap`.
    for (size_t t1=0; t1 < ds->vocab_size; ++t1)
    {
        for (size_t t2=t1+1; t2 < ds->vocab_size; ++t2)
        {
            size_t column_size = get_term_bitmap_len(ds);
            uint64_t *c = aligned_malloc(32,column_size);
            uint64_t *column_ptr1 = (uint64_t *) get_term_bitmap(ds, t1);
            uint64_t *column_ptr2 = (uint64_t *) get_term_bitmap(ds, t2);
            int cnt = find_frequency(column_size/8, column_ptr1, column_ptr2,c);
            if (cnt >= threshold)
                push_output_pair(op, t1, t2, cnt);
            aligned_free(c);
        }
    }
}

int
set_intersection(uint64_t * a, uint64_t * b, size_t len_a, size_t len_b)
{
    int lookup_table[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
    size_t i = 0, j = 0, count =0;
    size_t al = (len_a / 4) * 4, bl = (len_b / 4) * 4;
    for(;i < al && j < bl;)
    {
        __m256i x = _mm256_loadu_si256 (( __m256i *)&a[i]);
        __m256i y = _mm256_loadu_si256 (( __m256i *)&b[j]);
        int a_last = _mm256_extract_epi64(x,3);
        int b_last = _mm256_extract_epi64(y,3);
        if (a_last > b_last)
        {j = j + 4;}
        else if (a_last == b_last)
        {i = i + 4; j = j + 4;}
        else {i = i + 4;}
        __m256i mask1 = _mm256_cmpeq_epi64(x, y);
        y = _mm256_permute4x64_epi64(y, _MM_SHUFFLE(0,3,2,1));
        __m256i mask2 = _mm256_cmpeq_epi64(x, y);
        y = _mm256_permute4x64_epi64(y, _MM_SHUFFLE(0,3,2,1));
        __m256i mask3 = _mm256_cmpeq_epi64(x, y);
        y = _mm256_permute4x64_epi64(y, _MM_SHUFFLE(0,3,2,1));
        __m256i mask4 = _mm256_cmpeq_epi64(x, y);
        __m256i mask = _mm256_or_si256(_mm256_or_si256(mask1, mask2),_mm256_or_si256(mask3, mask4));
        int maskf = _mm256_movemask_pd((__m256d)mask);
        count = count + lookup_table[maskf];
    }
    for (;i < len_a && j < len_b;)
    {
        if (a[i] == b[j]) { ++count; ++i; ++j; }
        else if (a[i] < b[j]) { ++i; }
        else { ++j; }
    }
    return count;
}

void
find_pairs_quick_indexes(const dataset *ds, output_pairs *op, int threshold)
{
    // TODO implement a quick `find_pairs_quick_indexes` procedure using
    // `get_term_indexes`.
    for (size_t t1 = 0; t1 < ds->vocab_size; ++t1)
    {
        const index_list *il1 = get_term_indexes(ds, t1);
        if (il1->len >= threshold)
        {
            for (size_t t2 = t1+1; t2 < ds->vocab_size; ++t2)
            {
                const index_list *il2 = get_term_indexes(ds, t2);
                if (il2->len >= threshold) // If length of any index list (il1, il2) is less than threshold, it cant be pushed to output
                {
                    int count = set_intersection(il1->indexes,il2->indexes,il1->len,il2->len);
                    if (count >= threshold)
                        push_output_pair(op, t1, t2, count); 
                }         
            }
        }    
    }
}
