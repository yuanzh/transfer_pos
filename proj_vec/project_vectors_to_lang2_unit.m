function project_vectors_to_lang2_unit(lang1_words_in_lang1_space_filename, lang2_words_in_lang2_space_filename, aligned_lang1_words_in_lang1_space_filename, aligned_lang2_words_in_lang2_space_filename, lang1_space_to_lang2_space_filename, lang1_words_in_lang2_space_outfile, lang2_words_in_lang2_space_outfile)

% first column is words, hence not being read
lang1_words_in_lang1_space = dlmread(lang1_words_in_lang1_space_filename, ' ', 0, 1);
lang2_words_in_lang2_space = dlmread(lang2_words_in_lang2_space_filename, ' ', 0, 1);
aligned_lang1_words_in_lang1_space = dlmread(aligned_lang1_words_in_lang1_space_filename, ' ', 0, 1);
aligned_lang2_words_in_lang2_space = dlmread(aligned_lang2_words_in_lang2_space_filename, ' ', 0, 1);

lang1_words_in_lang1_space_cols = size(lang1_words_in_lang1_space, 2);
lang2_words_in_lang2_space_cols = size(lang2_words_in_lang2_space, 2);
if norm(lang1_words_in_lang1_space(:, lang1_words_in_lang1_space_cols)) == 0
  lang1_words_in_lang1_space_cols = lang1_words_in_lang1_space_cols - 1;
  lang1_words_in_lang1_space = lang1_words_in_lang1_space(:, 1:lang1_words_in_lang1_space_cols);
end;
if norm(lang2_words_in_lang2_space(:, lang2_words_in_lang2_space_cols)) == 0
  lang2_words_in_lang2_space_cols = lang2_words_in_lang2_space_cols - 1;
  lang2_words_in_lang2_space = lang2_words_in_lang2_space(:, 1:lang2_words_in_lang2_space_cols);
end;

% center the embedding
%lang1_mean = mean(lang1_words_in_lang1_space);
%lang2_mean = mean(lang2_words_in_lang2_space);
%lang1_words_in_lang1_space = lang1_words_in_lang1_space - repmat(lang1_mean, size(lang1_words_in_lang1_space, 1), 1);
%lang2_words_in_lang2_space = lang2_words_in_lang2_space - repmat(lang2_mean, size(lang2_words_in_lang2_space, 1), 1);
%aligned_lang1_words_in_lang1_space = aligned_lang1_words_in_lang1_space - repmat(lang1_mean, size(aligned_lang1_words_in_lang1_space, 1), 1);
%aligned_lang2_words_in_lang2_space = aligned_lang2_words_in_lang2_space - repmat(lang2_mean, size(aligned_lang2_words_in_lang2_space, 1), 1);

% Normalize all the matrices by rows
lang1_words_in_lang1_space = normr(lang1_words_in_lang1_space);
lang2_words_in_lang2_space = normr(lang2_words_in_lang2_space);
aligned_lang1_words_in_lang1_space = normr(aligned_lang1_words_in_lang1_space);
aligned_lang2_words_in_lang2_space = normr(aligned_lang2_words_in_lang2_space);

cnt = size(aligned_lang1_words_in_lang1_space, 1);
dim = size(aligned_lang1_words_in_lang1_space, 2);

addpath '../unit_opt/';
[lang1_space_to_lang2_space, obj] = unitary_project(aligned_lang1_words_in_lang1_space, aligned_lang2_words_in_lang2_space);
obj
lang1_words_in_lang2_space = lang1_words_in_lang1_space * lang1_space_to_lang2_space;
lang1_words_in_lang2_space = normr(lang1_words_in_lang2_space);

%lang1_words_in_lang1_space(2, :)

%lang2_words_in_lang2_space = (lang2_words_in_lang2_space - repmat(mean(aligned_lang2_words_in_lang2_space), lang2_words_count, 1)) * B;
lang2_words_in_lang2_space = normr(lang2_words_in_lang2_space);
%lang2_words_in_lang2_space(2, :)

mean(lang1_words_in_lang2_space)
mean(lang2_words_in_lang2_space)

% Write output files
max_line = 100010
dlmwrite(lang1_space_to_lang2_space_filename, lang1_space_to_lang2_space, ' ');
dlmwrite(lang1_words_in_lang2_space_outfile, lang1_words_in_lang2_space(1:max_line, :), ' ');
dlmwrite(lang2_words_in_lang2_space_outfile, lang2_words_in_lang2_space(1:max_line, :), ' ');

clear;
