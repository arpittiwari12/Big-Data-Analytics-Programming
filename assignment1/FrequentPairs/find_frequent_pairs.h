#ifndef COUNT_PAIRS_H
#define COUNT_PAIRS_H

/*
 * DO NOT MODIFY THIS INTERFACE.
 *
 * You should implement `find_pairs_quick_bitmaps` and
 * `find_pairs_quick_bitmaps`.
 */

/**
 * Output the number of times a word pair occurs together in a document. Use
 * the `push_output_pair` function to output a word pair and its occurrence
 * count. Important notes:
 *      - Skip all pairs that occur less than `threshold` times. (Do report the
 *        pairs that appear `threshold` times).
 *      - Words in a pair should be different.
 *      - The order in which you output the pairs does not matter.
 *
 * This is a naive implementation to check your results.
 */
void
find_pairs_naive_bitmaps(const dataset *ds, output_pairs *op, int threshold);

/**
 * See docs of `find_pairs_naive_bitmaps`.
 *
 * This is a naive implementation to check your results.
 */
void
find_pairs_naive_indexes(const dataset *ds, output_pairs *op, int threshold);

/**
 * See docs of `find_pairs_naive_bitmaps`.
 *
 * Implement this. Use the bitmaps returned by `get_term_bitmap` (dataset.h)
 * to compute the most frequent pairs. `find_pairs_naive_bitmaps` shows how
 * these bitmaps can be used.
 */
void
find_pairs_quick_bitmaps(const dataset *ds, output_pairs *op, int threshold);

/**
 * See docs of `find_pairs_naive_bitmaps`.
 *
 * Implement this. Use the index lists returned by `get_term_indexes` (dataset.h)
 * to compute the most frequent pairs. `find_pairs_naive_indexes` shows how
 * these index lists can be used.
 */
void
find_pairs_quick_indexes(const dataset *ds, output_pairs *op, int threshold);

#endif /* COUNT_PAIRS_H */
