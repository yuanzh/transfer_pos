#!/bin/sh

# Project embeddings with CCA or pseudo-inverse
# Argument:
# 1. Target language embedding file
# 2. Source language embedding file
# 3. Word pair file
# 4. Number of word pairs
# 5. Output file for projection matrix
# 6. Projected embedding file for target language
# 7. Projected embedding file for source language
# 8. Running id
#
# Example usage
# sh project_vectors_to_lang2.sh Spanish.vec English.vec es-en.pair 10 projection Spanish.proj English.proj 0

# Exit if any of the below listed command fails.
set -e

id=$8

# Get the aligned vectors from lang1 for English (lang2).
echo "Aligning vectors..."
echo temp_${id}
python alignVectors.py -w1 $1 -w2 $2 -a $3 -k $4 -o temp_${id}
# alignVectors.py writes two files: temp_orig_subset.txt, temp_new_aligned.txt.
# Give them better names.
mv temp_${id}_orig_subset.txt temp_${id}_aligned_lang2_embeddings
mv temp_${id}_new_aligned.txt temp_${id}_aligned_lang1_embeddings

# Find a "shared" vector space such that the projected words from lang1 vector
# space and the projected of aligned words in lang2 vector space are maximally
# correlated. Assume linear transformations. Finally, project all lang1 word
# vectors from the lang1 vector space to lang2 vector space (via the shared 
# space).
echo "Projecting vectors..."
matlab -nodesktop -nosplash -nojvm -nodisplay -r "project_vectors_to_lang2('$1','$2', 'temp_${id}_aligned_lang1_embeddings', 'temp_${id}_aligned_lang2_embeddings', '$5', 'temp_${id}_lang1_words_in_lang2_space', 'temp_${id}_lang2_words_in_lang2_space'); exit"
#matlab -nodesktop -nosplash -nojvm -nodisplay -r "project_vectors_to_lang2_unit('$1','$2', 'temp_${id}_aligned_lang1_embeddings', 'temp_${id}_aligned_lang2_embeddings', '$5', 'temp_${id}_lang1_words_in_lang2_space', 'temp_${id}_lang2_words_in_lang2_space'); exit"

# Do some post-processing for English
echo "Some post-processing..."
cut -f1 -d" " $1  > temp_${id}_lang1_words
cut -f1 -d" " $2  > temp_${id}_lang2_words
python paste.py -1 temp_${id}_lang1_words -2 temp_${id}_lang1_words_in_lang2_space -o $6 -d" "
python paste.py -1 temp_${id}_lang2_words -2 temp_${id}_lang2_words_in_lang2_space -o $7 -d" "
rm -f temp_${id}_*

echo "The linear projection matrix between lang1->lang2 spaces can be found at " $5
echo "Embeddings of lang1 words, projected to lang2 space, can be found at " $6
echo "Embeddings of lang2 words, normalized in the lang2 space, can be found at " $7
