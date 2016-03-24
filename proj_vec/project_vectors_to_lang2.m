function project_vectors_to_lang2(lang1_words_in_lang1_space_filename, lang2_words_in_lang2_space_filename, aligned_lang1_words_in_lang1_space_filename, aligned_lang2_words_in_lang2_space_filename, lang1_space_to_lang2_space_filename, lang1_words_in_lang2_space_outfile, lang2_words_in_lang2_space_outfile)

% first column is words, hence not being read
lang1_words_in_lang1_space = dlmread(lang1_words_in_lang1_space_filename, ' ', 0, 1);
lang2_words_in_lang2_space = dlmread(lang2_words_in_lang2_space_filename, ' ', 0, 1);
aligned_lang1_words_in_lang1_space = dlmread(aligned_lang1_words_in_lang1_space_filename, ' ', 0, 1);
aligned_lang2_words_in_lang2_space = dlmread(aligned_lang2_words_in_lang2_space_filename, ' ', 0, 1);

%aligned_lang1_words_in_lang1_space
%aligned_lang2_words_in_lang2_space(1, :)
%lang2_words_in_lang2_space(6, :)
%z1 = lang1_words_in_lang1_space(2, :)
%z2 = lang2_words_in_lang2_space(2, :)

% word2vec embeddings have a trailing space which matlab parses as an additional
% column of all zeros. If the last column is all zeros, remove it.
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

%coss = sum(sum(aligned_lang1_words_in_lang1_space.*aligned_lang2_words_in_lang2_space))
%sqrs = sum(sum((aligned_lang1_words_in_lang1_space - aligned_lang2_words_in_lang2_space).^2))

cnt = size(aligned_lang1_words_in_lang1_space, 1);
dim = size(aligned_lang1_words_in_lang1_space, 2);
if cnt <= dim
%lang1_space_to_lang2_space = aligned_lang1_words_in_lang1_space \ aligned_lang2_words_in_lang2_space;
inv_mat = pinv(aligned_lang1_words_in_lang1_space);
lang1_space_to_lang2_space = inv_mat * aligned_lang2_words_in_lang2_space;
lang1_space_to_lang2_space = lang1_space_to_lang2_space + (eye(dim) - inv_mat * aligned_lang1_words_in_lang1_space) * rand(dim) * rand(dim) * 100;
lang1_words_in_lang2_space = lang1_words_in_lang1_space * lang1_space_to_lang2_space;
%aligned_lang2_words_in_lang2_space
%aligned_lang1_words_in_lang2_space = aligned_lang1_words_in_lang1_space * lang1_space_to_lang2_space
else
% Perform CCA on the subset of the aligned vectors
[A, B, r, U, V] = canoncorr(aligned_lang1_words_in_lang1_space, aligned_lang2_words_in_lang2_space);
%A(1, :)
%B(1, :)
%r
%U(1, :)
%V(1, :)
%aligned_lang1_words_in_lang1_space(1, :)
%lang1_words_in_lang1_space(5, :)

% Project lang1 words from the lang1 space to the lang2 space
lang1_words_count = size(lang1_words_in_lang1_space, 1);
lang2_words_count = size(lang2_words_in_lang2_space, 1);
% TODO: since B is an orthogonal matrix, inv(B) = B'
lang1_space_to_lang2_space = A / B;
%B/B
%size(lang1_space_to_lang2_space)
%size(lang1_words_in_lang1_space)
lang1_words_in_lang2_space = (lang1_words_in_lang1_space - repmat(mean(aligned_lang1_words_in_lang1_space), lang1_words_count, 1)) * lang1_space_to_lang2_space + repmat(mean(aligned_lang2_words_in_lang2_space), lang1_words_count, 1);
%lang2_words_in_lang2_space = (lang2_words_in_lang2_space - repmat(mean(aligned_lang2_words_in_lang2_space), lang2_words_count, 1)) * B;

end
%lang1_words_in_lang2_space = (lang1_words_in_lang1_space - repmat(mean(aligned_lang1_words_in_lang1_space), lang1_words_count, 1)) * A;
%+ repmat(mean(aligned_lang2_words_in_lang2_space), lang1_words_count, 1);

%lang1_words_in_lang2_space = (lang1_words_in_lang1_space - repmat(mean(aligned_lang1_words_in_lang1_space), lang1_words_count, 1)) * A;
%shift2 = (aligned_lang2_words_in_lang2_space - repmat(mean(aligned_lang2_words_in_lang2_space), cnt, 1));
%shift2(1,:)
%lang2_words_in_lang2_space = shift2 * B;
%lang2_words_in_lang2_space(1,:)
lang1_words_in_lang2_space = normr(lang1_words_in_lang2_space);

%lang1_words_in_lang1_space(2, :)

%lang2_words_in_lang2_space = (lang2_words_in_lang2_space - repmat(mean(aligned_lang2_words_in_lang2_space), lang2_words_count, 1)) * B;
lang2_words_in_lang2_space = normr(lang2_words_in_lang2_space);
%lang2_words_in_lang2_space(2, :)

mean(lang1_words_in_lang2_space)
mean(lang2_words_in_lang2_space)

% Write output files
max_line = 100010;
dlmwrite(lang1_space_to_lang2_space_filename, lang1_space_to_lang2_space, ' ');
dlmwrite(lang1_words_in_lang2_space_outfile, lang1_words_in_lang2_space(1:max_line, :), ' ');
dlmwrite(lang2_words_in_lang2_space_outfile, lang2_words_in_lang2_space(1:max_line, :), ' ');

%cnt = size(aligned_lang1_words_in_lang1_space, 1);
%m1 = (aligned_lang1_words_in_lang1_space - repmat(mean(lang1_words_in_lang1_space), cnt, 1)) * A;
%m2 = (aligned_lang2_words_in_lang2_space - repmat(mean(lang2_words_in_lang2_space), cnt, 1)) * B;
%m1 = (aligned_lang1_words_in_lang1_space - repmat(mean(aligned_lang1_words_in_lang1_space), cnt, 1)) * A / B + repmat(mean(aligned_lang2_words_in_lang2_space), cnt, 1);
%m2 = aligned_lang2_words_in_lang2_space;
%m1 = normr(m1);
%m2 = normr(m2);
%coss = sum(sum(m1.*m2))
%sqrs = sum(sum((m1 - m2).^2))

% Delete all matrices from memory
clear;
