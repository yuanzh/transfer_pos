import xml.etree
from collections import Counter
import sys
try:
    import nltk
except ImportError:
    pass

def to_unicode(obj, encoding='utf-8'):
    if isinstance(obj, basestring):
        if not isinstance(obj, unicode):
            obj = unicode(obj, encoding)
    return obj

def read_pos_map(filename):
    unimap_file = open(filename, 'r')
    pos_map = {}
    for line in unimap_file:
        split = line.split()
        pos_map[split[0]] = split[1]
    return pos_map

def do_pos_map(pos, map, lang):
    if pos in map:
        result = map[pos]
    else:
        matched = False
        if pos == '':
            result = 'X'
            matched = True
        for p in map:
            if pos.startswith(p):
                result = map[p]
                matched = True

        if not matched:
            result = 'X'
            print 'warning: using X for POS', pos, lang
    return result

def read_conll_sentences(conll_filename, unimap_filename, append_specific_pos=False):
    pos_map = read_pos_map(unimap_filename)
    conll_file = open(conll_filename, 'r')
    sentences = []
    curr_sent = []
    for line in conll_file:
        split = line.strip().split()
        if len(split) == 0:
            if len(curr_sent) > 0:
                sentences.append(curr_sent)
            curr_sent = []
            continue
        word = to_unicode(split[1].lower())
        pos = do_pos_map(split[4], pos_map, conll_filename)
        if append_specific_pos:
            pos += '-' + split[4]

        curr_sent.append((word, pos))
    if len(curr_sent) > 0:
        sentences.append(curr_sent)
    return sentences

def write_sentences_to_file(sentences, filename):
    f = open(filename, 'w')
    for sentence in sentences:
        for w, p in sentence:
            #f.write(w.encode('UTF-8'))
            f.write(w)
            f.write('\t')
            f.write(p)
            f.write('\t')
        f.write('\n')

def read_sentences_from_file(filename, exclude_pos = False):
    f = open(filename, 'r')
    result = []
    for line in f:
        s = line.strip().split()
        words = [w for i, w in enumerate(s) if i % 2 == 0]
        pos = [w for i, w in enumerate(s) if i % 2 == 1]
        sent = [(w, p) for w, p in zip(words, pos) if p != '.'] if exclude_pos else zip(words, pos)
        result.append(sent)
    return result

def read_xml_sentences(filename):
    #TODO handle mk format; also handle the <hi> tags in sk

    # T is article->DET; I is interjection->X; X is residual->X; Y is abbreviation->X
    pos_map = {'N': 'NOUN', 'V': 'VERB', 'A': 'ADJ', 'P': 'PRON', 'D': 'DET', 'T': 'DET', 'R': 'ADV', 'S': 'ADP', 'C': 'CONJ', 'M': 'NUM', 'I': 'X', 'X': 'X', 'Y': 'X', 'Q': 'PRT'}

    e = xml.etree.ElementTree.parse(filename).getroot()
    prefix = '{http://www.tei-c.org/ns/1.0}'
    text_node = e.find(prefix+'text')
    result = []
    for sentence in text_node.iter(prefix+'s'):
        s = []
        for word in sentence:  # sentence.findall(prefix+'w'):
            if word.tag == prefix+'w':
                w = to_unicode(word.get('lemma'))
                p = word.get('ana')
                if p is None:
                    p = word.get('function')
                    p = pos_map[p[0]]
                else:
                    if p[0] != '#':
                        raise Exception('Expected #')
                    p = pos_map[p[1]]
                s.append((w, p))
            elif word.tag == prefix+'c':
                w = word.text
                p = '.'
                s.append((w, p))
            else:
                print 'warning: %s tag in sentence not handled' % word.tag
        result.append(s)
    return result

def read_hindi_corpus(dir):
    if nltk is None:
        return

    pos_map = read_pos_map('indian.uni.map')
    for f in nltk.corpus.indian.fileids():
        print '\n', f
        sentences = nltk.corpus.indian.tagged_sents(f, simplify_tags=False)
        #used_pos = {p: True for s in sentences for _, p in s}
        converted = []
        for sent in sentences:
            for w, p in sent:
                converted.append([(to_unicode(w), do_pos_map(p, pos_map, f)) for w, p in sent])
        write_sentences_to_file(converted, dir + '/' + f)
        #for p in used_pos:
        #    print p
        #    print_pos_examples(sentences, p)

def print_pos_examples(sent, pos):
    words = {}
    for s in sent:
        for w, p in s:
            if p == pos:
                words[w] = True
    for w in words.keys()[:20]:
        print w

def load_WALS_map(filename):
    WALS_map = {}
    map_file = open(filename)
    for line in map_file:
        s = line.split()
        WALS_map[s[0]] = s[1]# + ' ' + s[2]

    return WALS_map

def load_WALS_map2(filename):
    WALS_map = {}
    map_file = open(filename)
    for i, line in enumerate(map_file):
        if i == 0:
            continue
        s = line.split()
        WALS_map[s[0]] = s[2] #s[1] + ' ' + s[2]
    return WALS_map

def load_pos_sequence_file(filename):
    result = []
    file = open(filename)
    for line in file:
        result.append(line.split())
    return result


def integer_sentences(sentences, pos=None, max_words=None):
    sentences_flat = [i for s in sentences for i in s]
    word_counts = Counter(w for w, _ in sentences_flat)
    pos_counts = Counter(p for _, p in sentences_flat)
    words = word_counts.keys()
    words.sort(key=lambda x: word_counts[x], reverse=True)
    unk_id = -1
    if max_words is not None and len(words) > max_words:
        words = words[:max_words-1]
        unk_id = len(words)
        words.append('<UNK>')
    if pos is None:
        pos = pos_counts.keys()
    word_ids = {w:i for i, w in enumerate(words)}
    pos_ids = {p:i for i, p in enumerate(pos)}
    sentence_words = []
    sentence_pos = []
    for sentence in sentences:
        s = [word_ids[w] if w in word_ids else unk_id for w, _ in sentence]
        sentence_words.append(s)
        s = [pos_ids[p] for _, p in sentence]
        sentence_pos.append(s)
    return sentence_words, words, sentence_pos, pos

def integer_sentences_yuan(sentences, pos=None, max_words=None, unk_word='<UNK>'):
    sentences_flat = [i for s in sentences for i in s]
    word_counts = Counter(w for w, _ in sentences_flat)
    pos_counts = Counter(p for _, p in sentences_flat)
    full_words = word_counts.keys()
    full_words.sort(key=lambda x: word_counts[x], reverse=True)
    unk_id = -1

    count = [word_counts[word] for word in full_words]
    #print sum(count)
    thresh = 0
    for i in xrange(1, len(count)):
        if count[i] > 2:
            thresh = i
        count[i] += count[i - 1]
    #print count[-1], count[1000], count[2000], thresh, count[thresh]
    #print len(count)
    #tmp = sys.stdin.readline()

    if max_words is not None and len(full_words) > max_words:
        freq_words = full_words[:max_words-1]
        freq_words.append(unk_word)
    else:
        freq_words = full_words[:]

    full_words = full_words[:thresh + 1]
    unk_id = len(full_words)
    full_words.append(unk_word)

    if pos is None:
        pos = pos_counts.keys()
    word_ids = {w:i for i, w in enumerate(full_words)}
    pos_ids = {p:i for i, p in enumerate(pos)}
    sentence_words = []
    sentence_pos = []
    for sentence in sentences:
        s = [word_ids[w] if w in word_ids else unk_id for w, _ in sentence]
        sentence_words.append(s)
        s = [pos_ids[p] for _, p in sentence]
        sentence_pos.append(s)
    return sentence_words, full_words, freq_words, sentence_pos, pos

def read_test_sentences_from_file(filename, words, unk_id, pos):
    f = open(filename, 'r')
    text_sentences = []
    sentence_words = []
    sentence_pos = []
    word_ids = {w:i for i, w in enumerate(words)}
    pos_ids = {p:i for i, p in enumerate(pos)}
    for line in f:
        s = line.strip().split()
        sent = [w for i, w in enumerate(s) if i % 2 == 0]
        word_int = [word_ids[w] if w in word_ids else unk_id for w in sent]
        pos_int = [pos_ids[p] for i, p in enumerate(s) if i % 2 == 1]
        text_sentences.append(sent)
        sentence_words.append(word_int)
        sentence_pos.append(pos_int)
    return text_sentences, sentence_words, sentence_pos

if __name__ == "__main__":
    for multext_lang in ['bg', 'cs', 'en', 'et', 'fa', 'hu', 'pl', 'ro', 'sl', 'sr']:  # TODO: add sk, mk
        s = read_xml_sentences('../MultextEast1984-ana/oana-%s.xml' % multext_lang)
        write_sentences_to_file(s, 'pos_data/multext-%s.pos' % multext_lang)
    for conll_lang in ['basque07', 'catalan07', 'greek07', 'hungarian07', 'italian07', 'arabic', 'dutch', 'spanish', 'german', 'czech', 'swedish', 'chinese', 'danish', 'bulgarian', 'english07', 'japanese', 'portuguese', 'turkish', 'slovene']:
        s = read_conll_sentences('paths/'+conll_lang+'.train', 'paths/'+conll_lang+'.uni.map')
        write_sentences_to_file(s, 'pos_data/conll-%s.pos' % conll_lang)
        s = read_conll_sentences('paths/'+conll_lang+'.train', 'paths/'+conll_lang+'.uni.map', True)
        write_sentences_to_file(s, 'pos_data/conll-%s.spec.pos' % conll_lang)
    read_hindi_corpus('pos_data')
