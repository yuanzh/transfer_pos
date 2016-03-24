import numpy
import scipy.optimize
import math
import sys
import load_and_save
from scipy.misc import logsumexp
import re
import heapq, operator
import itertools

universal_pos_tags = ['VERB', 'AUX', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'SCONJ', 'DET', 'NUM', 'PART', 'X', 'PUNCT', 'START']

def get_norm_and_normalize(arr):
    s = arr.sum()
    if s == 0.0: raise Exception()
    arr /= s
    return s

def apply_dictionary(arr, dictionary, word):
    if dictionary is not None:
        for j in xrange(n_pos):
            if not dictionary[word, j]:
                arr[j] = 0
    else:
        arr[start_token] = 0

def fill_forward_lattice(sentence_int, trans_probs, emission_probs, dictionary=None):
    n = trans_probs.shape[0]
    lattice = numpy.zeros((len(sentence_int), n))
    scaling_factors = numpy.zeros((len(sentence_int)))

    for i, w in enumerate(sentence_int):
        if i == 0:
            lattice[i, :] = trans_probs[start_token, :] * emission_probs[:, w]
        else:
            lattice[i, :] = lattice[i-1, :][numpy.newaxis, :].dot(trans_probs).ravel() * emission_probs[:, w]

        apply_dictionary(lattice[i], dictionary, sentence_int[i])
        #print i, lattice[i, :]
        scaling_factors[i] = 1.0 / get_norm_and_normalize(lattice[i, :])

    return lattice, scaling_factors

def add_counts_forward_backward(sentence_int, trans_probs, emission_probs, trans_counts, em_counts, dictionary=None):
    n_pos = trans_probs.shape[0]
    len_sent = len(sentence_int)
    forward_lattice, scaling_factors = fill_forward_lattice(sentence_int, trans_probs, emission_probs, dictionary)

    backward_lattice = numpy.zeros((len_sent, n_pos))

    backward_lattice[len_sent-1, :] = trans_probs[:, start_token] # probability of transitioning to end token
    apply_dictionary(backward_lattice[len_sent-1], dictionary, sentence_int[len_sent-1])
    norm = (forward_lattice[len_sent-1] * backward_lattice[len_sent-1]).sum()
    backward_lattice[len_sent-1, :] /= norm

    for i in reversed(xrange(len_sent-1)):
        backward_lattice[i, :] = (trans_probs.dot((backward_lattice[i+1, :] * emission_probs[:, sentence_int[i+1]])[:, numpy.newaxis]) * scaling_factors[i+1]).ravel()
        apply_dictionary(backward_lattice[i], dictionary, sentence_int[i])

    probs = forward_lattice * backward_lattice
    #print probs.sum(axis=1)

    for i in xrange(len_sent):
        # emission counts
        em_counts[:, sentence_int[i]] += probs[i, :]
        # transition counts
        if i == 0:
            trans_counts[start_token, :] += probs[i, :]
        else:
            m = forward_lattice[i-1, :][:, numpy.newaxis].dot((backward_lattice[i, :] * emission_probs[:, sentence_int[i]])[numpy.newaxis, :])
            m *= trans_probs
            m /= m.sum()
            trans_counts += m
        if i == len_sent - 1:
            trans_counts[:, start_token] += probs[i, :]

def viterbi(sentence_int, trans_probs, emission_probs, dictionary=None):
    n = trans_probs.shape[0]
    lattice = numpy.zeros((len(sentence_int), n))
    back_ptr = numpy.zeros((len(sentence_int), n), dtype=int)

    for i, w in enumerate(sentence_int):
        if i == 0:
            lattice[i, :] = trans_probs[start_token, :] * emission_probs[:, w]
            apply_dictionary(lattice[i], dictionary, w)
        else:
            has_tag = False
            for j in xrange(n):
                if j == start_token or (dictionary is not None and not dictionary[w, j]):
                    lattice[i, j] = 0
                    continue

                has_tag = True
                v = trans_probs[:, j] * lattice[i-1, :]
                val_from = v.argmax()
                val = v[val_from] * emission_probs[j, w]
                back_ptr[i, j] = val_from
                lattice[i, j] = val
        norm = get_norm_and_normalize(lattice[i, :])

    end_probs = lattice[-1, :] * trans_probs[:, start_token]  # include the probability of going to end token
    last = end_probs.argmax()
    seq_rev = [last]
    for i in reversed(xrange(1, len(sentence_int))):
        last = back_ptr[i, last]
        seq_rev.append(last)
    seq_rev.reverse()
    return seq_rev

def get_word_vec(language, words):
    print 'Loading word embeddings of', language
    f = open('ud2_data/embeddings/' + language + '.sg.1.20.unit.proj.10')
    data = f.readlines()
    f.close()

    n_words, dim = [int(s) for s in data[0].strip().split()]
    n_words = min(n_words, len(data) - 1)
    n_words = min(n_words, 100000)
    #print n_words, dim
    word_vec = numpy.zeros((len(words), dim), dtype=float)
    have_vec = set([])
    mean = numpy.zeros(dim, dtype=float)
    for i in xrange(n_words):
        line = data[i + 1].strip().split()
        word = line[0].lower()
        if word in words and word not in have_vec:
            have_vec.add(word)
            mean += [float(s) for s in line[1:dim + 1]]
    mean /= len(have_vec)

    have_vec = set([])
    for i in xrange(n_words):
        line = data[i + 1].strip().split()
        word = line[0].lower()
        if word in words and word not in have_vec:
            have_vec.add(word)
            word_id = words.index(word)
            word_vec[word_id, :] = [float(s) for s in line[1:dim + 1]]
            word_vec[word_id, :] /= numpy.linalg.norm(word_vec[word_id])
    return word_vec

def run_test(text_sentences, sentences, sentence_pos, trans_probs, em_probs, proto_index, out_file, dictionary=None):
    out_text_sents = []
    correct = 0
    total = 0
    s = 0
    confusion_matrix = numpy.zeros((n_pos, n_pos))
    for sentence, sent_pos in zip(sentences, sentence_pos):
        pred_pos = viterbi(sentence, trans_probs, em_probs, dictionary)
        w = 0
        out_text_sent = []
        for pred, gold, word in zip(pred_pos, sent_pos, sentence):
            total += 1
            if pred == gold:
                correct += 1
            out_text_sent.append((text_sentences[s][w], pos_tags[pred]))
            w += 1
            confusion_matrix[pred, gold] += 1
        out_text_sents.append(out_text_sent)
        s += 1
    print '------------------'
    print 'accuracy:', correct / float(total)
    print 'many to one:', confusion_matrix.max(axis=1).sum() / float(confusion_matrix.sum())
    print '------------------'
    if out_file is not None:
        load_and_save.write_sentences_to_file(out_text_sents, out_file)

def get_source_data(language):
    text_sentences = load_and_save.read_sentences_from_file('ud2_data/pos/'+language+'.pos')
    sentences, full_words, freq_words, sentences_pos, pos = load_and_save.integer_sentences_yuan(text_sentences, pos=pos_tags, max_words=1000, unk_word=unk_word)
    word_vec = get_word_vec(language, full_words)
    return full_words, text_sentences, sentences, sentences_pos, word_vec

def probs_from_scores(scores, normalize_axis=1):
    probs = numpy.exp(scores)
    probs /= probs.sum(axis=normalize_axis, keepdims=True)
    return probs

def trans_scores_from_weights(weights):
    return weights[trans_feat_start:trans_feat_end].reshape(n_pos, n_pos).copy()

def em_scores_from_weights(weights, word_vec, projection=False, indicator=False, misc=False):
    if projection:
        projection_matrix = weights[projection_matrix_start:projection_matrix_end].reshape(word_vec_dim, word_vec_dim)
        word_vec = word_vec.dot(projection_matrix)
    score = (weights[embedding_feat_start:embedding_feat_end].reshape(n_pos, word_vec_dim)).dot(word_vec.transpose())
    if indicator:
        score += weights[indicator_start:indicator_end].reshape(n_pos, word_vec.shape[0])
    if misc:
        score += weights[misc_start:misc_end].reshape(n_pos, n_misc_feat).dot(misc_feature_map.transpose())
    return score

def get_source_model(reg_coeff, language):
    def ll(weights):
        log_prob = 0.0
        trans_scores = trans_scores_from_weights(weights)
        em_scores = em_scores_from_weights(weights, word_vec)
        trans_probs[:, :] = probs_from_scores(trans_scores)
        em_probs[:, :] = probs_from_scores(em_scores)
        log_prob += (trans_act_counts * numpy.nan_to_num(numpy.log(trans_probs))).sum()
        log_prob += (em_act_counts * numpy.nan_to_num(numpy.log(em_probs))).sum()

        diff = weights - prior_weights
        log_prob -= (reg_coeffs * diff * diff).sum()

        return log_prob

    def gradient(weights):
        grad = numpy.zeros(n_feats)
        trans_gradient = grad[trans_feat_start:trans_feat_end]
        em_gradient = grad[embedding_feat_start:embedding_feat_end]

        trans_gradient += trans_act_counts.reshape(n_pos * n_pos)
        trans_gradient -= (trans_probs * trans_act_counts.sum(axis=1, keepdims=True)).reshape(n_pos * n_pos)
        em_gradient += em_act_counts.dot(word_vec).reshape(n_pos * word_vec_dim)
        em_gradient -= (em_probs.dot(word_vec) * em_act_counts.sum(axis=1, keepdims=True)).reshape(n_pos * word_vec_dim)

        diff = weights - prior_weights
        grad -= 2 * reg_coeffs * diff

        return grad

    n_pos = len(pos_tags)
    n_feats = embedding_feat_end

    word_vec = source_word_vec
    n_words = len(source_words)

    trans_act_counts = numpy.zeros((n_pos, n_pos))
    em_act_counts = numpy.zeros((n_pos, n_words))
    trans_probs = numpy.zeros((n_pos, n_pos))
    em_probs = numpy.zeros((n_pos, n_words))

    for sent, sent_pos in zip(source_sentences, source_sentences_pos):
        for i in xrange(len(sent)):
            p = sent_pos[i]
            if i == 0:
                trans_act_counts[start_token, p] += 1
            else:
                trans_act_counts[sent_pos[i-1], p] += 1

            em_act_counts[p, sent[i]] += 1
        trans_act_counts[sent_pos[-1], start_token] += 1

    total_token = get_total_token(source_sentences)
    trans_act_counts = trans_act_counts * 15 / total_token
    em_act_counts = em_act_counts * 15 / total_token

    init_weights = numpy.zeros(n_feats)
    prior_weights = numpy.zeros(n_feats)
    reg_coeffs = numpy.ones(n_feats) * reg_coeff
    result = scipy.optimize.minimize(lambda w: -ll(w), init_weights, method='L-BFGS-B', jac=lambda w: -gradient(w), options={'ftol':1e-6})
    weights = result.x
    log_prob = result.fun

    trans_weights = weights[trans_feat_start:trans_feat_end]
    em_weights = weights[embedding_feat_start:embedding_feat_end]
    #print trans_weights.min(), trans_weights.max(), em_weights.max(), em_weights.min()

    return weights

def get_word_pairs(pair_str, words1, words2):
    f = open('ud2_data/pairs/' + pair_str + '.pair.10')
    data = f.readlines()
    f.close()
    data = [s.strip().split(' ||| ') for s in data]
    ret = [[x[0].lower(), x[1].lower()] for x in data if x[0].lower() in words1 and x[1].lower() in words2]
    return ret

def get_dictionary(sentences, sentences_pos, source_n_words, target_n_words, word_pairs_index):
    if word_pairs_index is None:
        return None, None
    source_dictionary = numpy.ones((source_n_words, n_pos), dtype=bool)
    target_dictionary = numpy.ones((target_n_words, n_pos), dtype=bool)
    source_word_pairs_index = word_pairs_index[1]
    target_word_pairs_index = word_pairs_index[0]

    sentences_flat = [i for s in zip(sentences, sentences_pos) for i in zip(s[0], s[1])]
    count = numpy.zeros((source_n_words, n_pos))
    for w, p in sentences_flat:
        count[w, p] += 1
    for i in xrange(len(source_word_pairs_index)):
        source_word = source_word_pairs_index[i]
        target_word = target_word_pairs_index[i]
        has_tag = False
        for j in xrange(n_pos):
            if count[source_word, j] < 3:
                source_dictionary[source_word, j] = False
                target_dictionary[target_word, j] = False
            else:
                has_tag = True
        if not has_tag:
            source_dictionary[source_word, :] = True
            target_dictionary[target_word, :] = True

    for w in target_word_pairs_index:
        has_tag = False
        for j in xrange(n_pos): has_tag = has_tag or target_dictionary[w, j]
        if not has_tag: target_dictionary[w, :] = True

    source_dictionary[:, start_token] = False
    target_dictionary[:, start_token] = False

    #print source_dictionary[source_word_pairs_index]
    #print target_dictionary[target_word_pairs_index]
    return source_dictionary, target_dictionary

def get_word_pairs_index(word_pairs, words1, words2):
    index = [[], []]
    for pair in word_pairs:
        index[0].append(words1.index(pair[0]))
        index[1].append(words2.index(pair[1]))
    return index

def output_model(filename, weights):
    f = open(filename, 'w')
    f.write(str(len(weights)) + '\n')
    for w in weights:
        f.write(str(w) + '\n')
    f.write('\n')
    f.close()

def load_model(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    n = int(data[0].strip())
    weights_str = data[1].strip().split()
    weights = numpy.asarray([float(w) for w in weights_str])

    return weights

def get_total_token(sentences):
    total = 0
    for sent in sentences:
        total += len(sent)
    return total

def get_most_possible_tags(sentences, sentences_pos, n_words):
    count = numpy.zeros((n_pos, n_words), dtype=int)
    for sent, sent_pos in zip(sentences, sentences_pos):
        for w, p in zip(sent, sent_pos):
            count[p, w] += 1
    dictionary = {}
    for w in xrange(n_words):
        tag = count[:, w].argmax()
        dictionary[w] = tag
    return dictionary

def get_emission_features(words):
    feature_dict = {}
    # first round, build features
    for word in words[:-1]:
        # suffix
        for l in xrange(2, 4):
            suffix = word[-l:]
            if suffix not in feature_dict:
                feature_dict['suf' + str(l) + '=' + suffix] = len(feature_dict)

    # contains number
    feature_dict['num=True'] = len(feature_dict)

    # second round, build tables
    n_feat = len(feature_dict)
    feature_map = numpy.zeros((len(words), n_feat))
    for i, word in enumerate(words[:-1]):
        # suffix
        for l in xrange(2, 4):
            suffix = word[-l:]
            feature_map[i, feature_dict['suf' + str(l) + '=' + suffix]] = 1

        # contains number
        if re.search('\d', word):
            feature_map[i, feature_dict['num=True']] = 1

    return feature_map

def get_target_model(source_weights):
    def ll(weights):
        log_prob = 0.0
        trans_scores = trans_scores_from_weights(weights)
        em_scores = em_scores_from_weights(weights, target_word_vec, projection=True, indicator=use_indicator, misc=use_misc)
        trans_probs[:, :] = probs_from_scores(trans_scores)
        em_probs[:, :] = probs_from_scores(em_scores)
        log_prob += (trans_counts * numpy.nan_to_num(numpy.log(trans_probs))).sum()
        log_prob += (em_counts * numpy.nan_to_num(numpy.log(em_probs))).sum()

        diff = weights - prior_weights
        log_prob -= (reg_coeffs * diff * diff).sum()

        return log_prob

    def gradient(weights):
        grad = numpy.zeros(n_feats)
        trans_gradient = grad[trans_feat_start:trans_feat_end]
        em_gradient = grad[embedding_feat_start:embedding_feat_end]
        projection_gradient = grad[projection_matrix_start:projection_matrix_end]
        embedding_weights = weights[embedding_feat_start:embedding_feat_end].reshape(n_pos, word_vec_dim)
        projection_matrix = weights[projection_matrix_start:projection_matrix_end].reshape(word_vec_dim, word_vec_dim)
        word_vec = target_word_vec.dot(projection_matrix)

        trans_gradient += trans_counts.reshape(n_pos * n_pos)
        trans_gradient -= (trans_probs * trans_counts.sum(axis=1, keepdims=True)).reshape(n_pos * n_pos)
        em_gradient += em_counts.dot(word_vec).reshape(n_pos * word_vec_dim)
        em_gradient -= (em_probs.dot(word_vec) * em_counts.sum(axis=1, keepdims=True)).reshape(n_pos * word_vec_dim)
        projection_gradient += em_counts.dot(target_word_vec).transpose().dot(embedding_weights).reshape(word_vec_dim * word_vec_dim)
        projection_gradient -= (em_probs.dot(target_word_vec) * em_counts.sum(axis=1, keepdims=True)).transpose().dot(embedding_weights).reshape(word_vec_dim * word_vec_dim)

        # indicator
        if use_indicator:
            indicator_gradient = grad[indicator_start:indicator_end]
            indicator_gradient += em_counts.reshape(n_pos * n_words)
            indicator_gradient -= (em_counts.sum(axis=1, keepdims=True) * em_probs).reshape(n_pos * n_words)

        # misc
        if use_misc:
            misc_gradient = grad[misc_start:misc_end]
            misc_gradient += em_counts.dot(misc_feature_map).reshape(n_pos * n_misc_feat)
            misc_gradient -= (em_counts.sum(axis=1, keepdims=True) * em_probs.dot(misc_feature_map)).reshape(n_pos * n_misc_feat)

        diff = weights - prior_weights
        grad -= 2 * reg_coeffs * diff

        return grad

    n_feats = all_end
    reg_coeffs = numpy.ones(n_feats) * shared_reg_coeff
    init_weights = numpy.zeros(n_feats)
    init_weights[:embedding_feat_end] = source_weights[:]
    init_weights[projection_matrix_start:projection_matrix_end] = numpy.identity(word_vec_dim).reshape(word_vec_dim * word_vec_dim)
    prior_weights = init_weights.copy()
    target_weights = init_weights.copy()

    total_token = get_total_token(sentences)
    trans_scores = trans_scores_from_weights(init_weights)
    trans_probs = probs_from_scores(trans_scores)
    em_scores = em_scores_from_weights(init_weights, target_word_vec, projection=True, indicator=use_indicator, misc=use_misc)
    em_probs = probs_from_scores(em_scores)

    for iter in xrange(iterations):
        print 'Iteration:', iter
        trans_counts = numpy.zeros((n_pos, n_pos))
        em_counts = numpy.zeros((n_pos, n_words))

        for sentence in sentences:
            add_counts_forward_backward(sentence, trans_probs, em_probs, trans_counts, em_counts, target_dictionary)
        trans_counts = trans_counts * 15 / total_token
        em_counts = em_counts * 15 / total_token

        result = scipy.optimize.minimize(lambda w: -ll(w), init_weights, method='L-BFGS-B', jac=lambda w: -gradient(w))#, options={'ftol':5e-5})

        target_weights = result.x.copy()
        init_weights = target_weights.copy()

        trans_scores = trans_scores_from_weights(target_weights)
        trans_probs[:, :] = probs_from_scores(trans_scores)
        em_scores = em_scores_from_weights(target_weights, target_word_vec, projection=True, indicator=use_indicator, misc=use_misc)
        em_probs[:, :] = probs_from_scores(em_scores)
        run_test(test_text_sentences, test_sentences, test_sentences_pos, trans_probs, em_probs, word_pairs_index[0], out_file, dictionary=target_dictionary)

    return target_weights 

default_args = {'reg_coeff': 0.001, 'shared_reg_coeff': '0.01', 'iterations': 10, 'use_indicator': 'True', 'use_misc': 'False'}

args = default_args.copy()
for arg in sys.argv[1:]:
    i = arg.index(':')
    key = arg[:i]
    val = arg[i+1:]
    args[key] = val

(base_reg_coeff, shared_reg_coeff, out_file, source, target, pair_str, iterations, use_indicator, use_misc) = \
(float(args['reg_coeff']), float(args['shared_reg_coeff']), args['out_file'], args['source'], args['target'], args['pair_str'], int(args['iterations']), args['use_indicator'] == 'True', args['use_misc'] == 'True')

print 'regularization coeff', base_reg_coeff
print 'shared_reg_coeff', shared_reg_coeff
print 'source language', source
print 'target language', target
print 'pair_str', pair_str
print 'use_indicator', use_indicator
print 'use_misc', use_misc

word_vec_dim = 20
pos_tags = universal_pos_tags
unk_word = '<UNK>'
start_token = pos_tags.index('START')

language = target
text_sentences = load_and_save.read_sentences_from_file('ud2_data/pos/'+language+'.pos')
sentences, full_words, freq_words, sentences_pos, pos = load_and_save.integer_sentences_yuan(text_sentences, pos=universal_pos_tags, max_words=1000, unk_word=unk_word)
unk_id = full_words.index(unk_word) if unk_word in full_words else -1
target_word_vec = get_word_vec(language, full_words)
test_text_sentences, test_sentences, test_sentences_pos = load_and_save.read_test_sentences_from_file('ud2_data/pos/' + language + '.test.pos', full_words, unk_id, pos_tags)
if use_misc:
    misc_feature_map = get_emission_features(full_words)
    n_misc_feat = misc_feature_map.shape[1]

n_pos = len(pos_tags)
n_words = len(full_words)
trans_feat_start = 0
trans_feat_end = n_pos * n_pos
embedding_feat_start = trans_feat_end
embedding_feat_end = trans_feat_end + n_pos * word_vec_dim
projection_matrix_start = embedding_feat_end
projection_matrix_end = embedding_feat_end + word_vec_dim * word_vec_dim
indicator_start = projection_matrix_end
indicator_end = indicator_start + n_pos * n_words
all_end = indicator_end
if use_misc:
    misc_start = indicator_end
    misc_end = misc_start + n_pos * n_misc_feat
    all_end = misc_end

annotated_language = source
name = annotated_language
source_words, source_text_sentences, source_sentences, source_sentences_pos, source_word_vec = get_source_data(annotated_language)

word_pairs = get_word_pairs(pair_str, full_words, source_words)
word_pairs_index = get_word_pairs_index(word_pairs, full_words, source_words)
source_dictionary, target_dictionary = get_dictionary(source_sentences, source_sentences_pos, len(source_words), n_words, word_pairs_index)

print 'Train direct transfer model on source langauge'
weights = get_source_model(base_reg_coeff, annotated_language)

trans_scores = trans_scores_from_weights(weights)
trans_probs = probs_from_scores(trans_scores)
em_scores = em_scores_from_weights(weights, source_word_vec)
em_probs = probs_from_scores(em_scores)
print 'Direct transfer model test on source language'
run_test(source_text_sentences, source_sentences, source_sentences_pos, trans_probs, em_probs, word_pairs_index[1], None, dictionary=source_dictionary)
em_scores = em_scores_from_weights(weights, target_word_vec)
em_probs = probs_from_scores(em_scores)
print 'Direct transfer model test on target language'
run_test(test_text_sentences, test_sentences, test_sentences_pos, trans_probs, em_probs, word_pairs_index[0], out_file + '.direct', dictionary=target_dictionary)

print 'Train unsupervised model on target language'
weights = get_target_model(weights)
output_model('model.' + source + '.' + target, weights)
