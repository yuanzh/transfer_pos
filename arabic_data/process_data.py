import sys
import re

def process_annotation():
    with open('EGY-coda-tok-pos-MSA.buk.new', 'r') as f:
        lines = f.read().splitlines()

    f = open('egy.buk.gold.post', 'w')
    sent = ''
    for line in lines[1:]:
        if line.find('--------------------') != -1 or line.find('\t\t\t') != -1:
            f.write(sent.strip() + '\n')
            sent = ''
            continue
        print line
        data = line.split('\t')[4]
        data = data.split(' ')
        for d in data:
            tokens = d.split('+')
            stem = False
            for i, token in enumerate(tokens):
                s = token.split('/')
                if s[0] in prefix_set and not stem and i < len(tokens) - 1:
                    sent += s[0] + '+\t' + map_egy_tag(s[1]) + '\t'
                    continue
                #print s
                stem = True
                if i > 0 and (s[0] in suffix_set or s[0] in suffix_set2 or s[0] in suffix_set3 or s[1] == 'NSUFF'):
                    sent += '+' + s[0] + '\t' + map_egy_tag(s[1]) + '\t'
                else:
                    sent += s[0] + '\t' + map_egy_tag(s[1]) + '\t'
    if sent != '':
        f.write(sent.strip() + '\n')
    f.close()

def is_ascii(s):
    try:
        s.decode('ascii')
        return True
    except ValueError:
        return False

def process_egy_tweet():
    with open('egy.raw', 'r') as f:
        lines = f.read().splitlines()

    f = open('egy.buk.raw', 'w')
    for line in lines:
        #line = line[line.find(': ') + 2:].split(' ')
        line = line.split(' ')
        line = [s for s in line if is_ascii(s)]
        if len(line) > 0:
            f.write(' '.join(line) + '\n')
    f.close()

def clean_egy_tweet():
    with open('egyptian-tweets.txt', 'r') as f:
        lines = f.read().splitlines()

    f = open('egy.ar.raw', 'w')
    for line in lines:
        line = line[line.find(': ') + 2:]
        f.write(line + '\n')
    f.close()

def process_ud():
    with open('../../google/data/universal-dependencies-1.2/UD_Arabic/ar-ud-train.conllu', 'r') as f:
        lines = f.read().splitlines()

    f = open('msa.gold', 'w')
    sent = ''
    for line in lines:
        if line == '':
            f.write(sent.strip() + '\n')
            sent = ''
        elif line[0] == '#':
            continue
        else:
            d = line.split('\t')
            sent += d[1] + '\t' + d[3] + '\t'

    if sent != '':
        f.write(sent.strip() + '\n')
    f.close()

def remove_vowel():
    with open('msa.buk.vowel.gold', 'r') as f:
        lines = f.read().splitlines()

    f = open('msa.buk.gold', 'w')
    for line in lines:
        data = line.split('\t')
        for i in xrange(0, len(data), 2):
            data[i] = re.sub('[aeiou~]+', '', data[i])
        f.write('\t'.join(data) + '\n')
    f.close()

def vocab_overlap():
    #with open('egy.gold', 'r') as f:
    with open('egy.buk.gold.post', 'r') as f:
        lines = f.read().splitlines()
    voc_gold = set([])
    for line in lines:
        data = line.split('\t')
        for i in xrange(0, len(data), 2):
            voc_gold.add(data[i])
    print len(voc_gold)

    with open('egy.buk.raw.post', 'r') as f:
    #with open('egy.raw', 'r') as f:
        lines = f.read().splitlines()
    voc_raw = set([])
    for line in lines:
        data = line.split(' ')
        for i in xrange(len(data)):
            voc_raw.add(data[i])
    print len(voc_raw)

    c = 0
    for w in voc_gold:
        if w in voc_raw:
            c += 1
        #else:
        #    print w
        #    tmp = sys.stdin.readline()
    print c, float(c) / len(voc_gold)

def process_msa_gold():
    with open('msa.buk.gold', 'r') as f:
        lines = f.read().splitlines()

    f = open('msa.buk.gold.v2', 'w')
    for line in lines:
        data = line.split(' ')
        for i in xrange(len(data)):
            data[i] = re.sub("['`]", '', data[i])
        f.write(' '.join(data) + '\n')
    f.close()

def process_msa_raw():
    with open('msa.buk.raw', 'r') as f:
        lines = f.read().splitlines()

    f = open('msa.buk.raw.v2', 'w')
    for line in lines:
        data = line.split(' ')
        for i in xrange(len(data)):
            data[i] = re.sub("['`]", '', data[i])
            if len(data[i]) > 1 and data[i][0] == 'w': data[i] = 'w ' + data[i][1:]
            elif len(data[i]) > 1 and data[i][0] == 'b': data[i] = 'b ' + data[i][1:]
        f.write(' '.join(data) + '\n')
    f.close()

def add_word_to_dict(word, segment, seg_dict):
    if not word in seg_dict: seg_dict[word] = {}
    d = seg_dict[word]
    if not segment in d: d[segment] = 0
    cnt = d[segment]
    d[segment] = cnt + 1

#spmrl_tagmap = {'P':'PREP', 'C':'CONJ', 'PNX':'PUNCT', 'N':'EMP', 'PN':'EMP', 'V':'EMP', 'AJ':'EMP', 'PRO':'PRON', 'PRT':'PART', 'ABBREV':'EMP', 'REL':'EMP', 'AV':'EMP', 'CONJ':'CONJ'}
spmrl_tagmap = {'P':'PREP', 'C':'CONJ', 'PNX':'PUNCT', 'N':'NOUN', 'PN':'NOUN', 'V':'VERB', 'AJ':'ADJ', 'PRO':'PRON', 'PRT':'PART', 'ABBREV':'NOUN', 'REL':'PRON', 'AV':'ADV', 'CONJ':'CONJ'}
def map_spmrl_tag(tag):
    if tag in spmrl_tagmap:
        return spmrl_tagmap[tag]
    else:
        print tag
        tmp = sys.stdin.readline()
        return 'EMP'

def add_word_spmrl(word, segment, data):
    word += data[1]
    if segment != '': segment += '+'
    if data[5].find('det=y') != -1 and data[1][:2] == 'Al':
        segment += 'Al@#DET+' + data[1][2:] + '@#' + map_spmrl_tag(data[3]) 
    else:
        segment += data[1] + '@#' + map_spmrl_tag(data[3])
    return word, segment

def get_msa_raw_vocab():
    with open('msa.buk.raw.post', 'r') as f:
        lines = f.read().splitlines()
    vocab = {}
    for line in lines:
        data = line.split(' ')
        for w in data:
            if w not in vocab: vocab[w] = 1
            else:
                cnt = vocab[w]
                vocab[w] = cnt + 1
    return vocab

def extract_suffix(word, tag, vocab):
    cnt = vocab[word] if word in vocab else 0
    suffix = ''
    for s in suffix_set2:
        if word.endswith(s) and len(word) > len(s):
            w = word[:-len(s)]
            c = vocab[w] if w in vocab else 0
            if c > cnt:
                cnt = c
                suffix = s
    if suffix != '':
        return word[:-len(suffix)] + '\t' + tag + '\t+' + suffix + '\tNSUFF'
    else:
        return word + '\t' + tag

def generate_spmrl_data():
    with open('../../SegParser/data/full/spmrl.seg.train', 'r') as f:
        lines = f.read().splitlines()

    vocab = get_msa_raw_vocab()
    is_gold = True
    new_line = ''
    f = open('spmrl.buk.gold.post', 'w')
    stem = False
    for i, line in enumerate(lines):
        if line == '':
            is_gold = not is_gold
            if new_line != '':
                f.write(new_line.strip() + '\n')
                new_line = ''
            continue

        if not is_gold: continue

        data = line.split('\t')

        seg_id = int(data[0].split('/')[1])
        if seg_id == 0: stem = False

        if not stem and data[1] in prefix_set and i < len(lines) - 1:
            next_seg_id = int(lines[i + 1].split('\t')[0].split('/')[1])
            if next_seg_id != 0:
                new_line += '\t' + data[1] + '+\t' + map_spmrl_tag(data[3])
                continue

        if data[5].find('det=y') != -1 and data[1][:2] == 'Al':
            new_line += '\tAl+\tDET\t' + extract_suffix(data[1][2:], map_spmrl_tag(data[3]), vocab)
        elif data[1] in suffix_set or data[1] in suffix_set3:
            new_line += '\t+' + data[1] + '\t' + map_spmrl_tag(data[3])
        else:
            new_line += '\t' + extract_suffix(data[1], map_spmrl_tag(data[3]), vocab)
        stem = True
    if new_line != '':
        f.write(new_line.strip() + '\n')
    f.close()

def get_spmrl_dict():
    with open('../../SegParser/data/full/spmrl.seg.train', 'r') as f:
        lines = f.read().splitlines()

    is_gold = True
    word = ''
    segment = ''
    seg_dict = {}
    for line in lines:
        if line == '':
            is_gold = not is_gold
            continue

        if not is_gold: continue

        data = line.split('\t')
        seg_id = int(data[0].split('/')[1])

        if seg_id == 0:
            if word != '':
                add_word_to_dict(word, segment, seg_dict)
            word = ''
            segment = ''
            word, segment = add_word_spmrl(word, segment, data)
        else:
            word, segment = add_word_spmrl(word, segment, data)
    if word != '':
        add_word_to_dict(word, segment, seg_dict)

    for word in seg_dict:
        d = seg_dict[word]
        best_cnt = 0
        best_seg = ''
        for s in d:
            if d[s] > best_cnt: best_cnt, best_seg = d[s], s
        seg_dict[word] = best_seg

    return seg_dict

def map_egy_tag(tag):
    if tag == 'V':
        tag = 'VERB'
    elif tag.find('NOUN') != -1:
        tag = 'NOUN'
    elif tag.find('ADJ') != -1:
        tag = 'ADJ'
    elif tag.find('PRON') != -1:
        tag = 'PRON'
    elif tag.find('PART') != -1:
        tag = 'PART'
    elif tag == 'PUNC':
        tag = 'PUNCT'
    elif tag == 'CASE':
        #tag = 'NSUFF'
        pass
    elif tag == 'PREP' or tag == 'CONJ' or tag == 'NSUFF' or tag == 'DET' or tag == 'NUM' or tag == 'ADV' or tag == 'FOREIGN':
        pass
    elif tag == 'MENTION' or tag == 'HASH' or tag == 'URL' or tag == 'RETWEET' or tag == 'EMOT':
        tag = 'X'
    else:
        print 'aaa', tag
        tmp = sys.stdin.readline()
    return tag

def get_egy_dict():
    with open('EGY-coda-tok-pos-MSA.buk', 'r') as f:
        lines = f.read().splitlines()
    seg_dict = {}
    for line in lines[1:]:
        data = line.split('\t')
        if line.startswith('--------------'): continue
        data = data[4].split(' ')
        for segment_str in data:
            segments = segment_str.split('+')
            word = ''
            segment = []
            tag = []
            for s in segments:
                w, t = s.split('/')
                word += w
                t = map_egy_tag(t)
                if t.find('SUFF') != -1:
                    segment[-1] += w
                else:
                    segment.append(w)
                    tag.append(t)
            s = []
            for seg, t in zip(segment, tag):
                s.append(seg + '/' + t)
            s = '+'.join(s)
            add_word_to_dict(word, s, seg_dict)

    for word in seg_dict:
        d = seg_dict[word]
        best_cnt = 0
        best_seg = ''
        for s in d:
            if d[s] > best_cnt: best_cnt, best_seg = d[s], s
        seg_dict[word] = best_seg

    return seg_dict

def get_seg_from_dict(token, seg_dict, tag):
    segments = seg_dict[token].split('+')
    ret = []
    for seg in segments:
        d = seg.split('@#')
        ret.append((d[0], d[1] if d[1] != 'EMP' else tag))
    return ret

suffix_set = set(['h', 'hA', 'km', 'kn', 'y', 'ny', 'nA'])
def segment_token(token, seg_dict=None, tag=None, remove_diac=False, prefix=False, Al=False, suffix=False):
    orig_token = token
    if remove_diac:
        token = re.sub('[aiou~NFK]+', '', token)
    segment = []
    if seg_dict is not None and token in seg_dict:
        segment = get_seg_from_dict(token, seg_dict, tag)
    else:
        if prefix and len(token) > 1:
            if token[0] == 'w':
                segment.append(('w', 'CONJ'))
                token = token[1:]
            elif token[0] == 'f':
                segment.append(('f', 'CONJ' if tag is not None and tag != 'VERB' else 'PART'))
                token = token[1:]
            elif token[0] == 'b' or token[0] == 'l' or token[0] == 's':
                segment.append(('f', 'PREP' if tag is not None and tag != 'VERB' else 'PART'))
                token = token[1:]

        if Al and len(token) > 2 and token[:2] == 'Al':
            segment.append(('Al', 'DET'))
            token = token[2:]

        if suffix:
            if len(token) > 1 and token[-1] in suffix_set:
                segment.append((token[:-1], tag))
                segment.append((token[-1], 'PRON'))
            elif len(token) > 2 and token[-2:] in suffix_set:
                segment.append((token[:-2], tag))
                segment.append((token[-2:], 'PRON'))
            else:
                segment.append((token, tag))
        else:
            segment.append((token, tag))

    #print orig_token, segment
    #tmp = sys.stdin.readline()
    return segment

def process_msa_gold(seg_dict):
    with open('msa.buk.gold', 'r') as f:
        lines = f.read().splitlines()
    f = open('msa.buk.gold.post', 'w')
    for line in lines:
        data = line.split('\t')
        new_line = []
        for i in xrange(0, len(data), 2):
            segments = segment_token(data[i], seg_dict, data[i + 1], remove_diac=True, prefix=False, Al=True, suffix=False)
            for s in segments:
                new_line.append(s[0])
                new_line.append(s[1])
        f.write('\t'.join(new_line) + '\n')
    f.close()

def process_msa_raw(seg_dict):
    with open('msa.buk.raw', 'r') as f:
        lines = f.read().splitlines()
    f = open('msa.buk.raw.post', 'w')
    for line in lines:
        data = line.split(' ')
        new_line = []
        for i, token in enumerate(data):
            segments = segment_token(token, seg_dict, tag=None, remove_diac=True, prefix=True, Al=True, suffix=True)
            for s in segments:
                new_line.append(s[0])
        f.write(' '.join(new_line) + '\n')
    f.close()

def process_egy_gold(seg_dict):
    pass

def process_segmentation():
    spmrl_seg_dict = get_spmrl_dict()
    print len(spmrl_seg_dict)
    #process_msa_gold(spmrl_seg_dict)
    #process_msa_raw(spmrl_seg_dict)
    egy_seg_dict = get_egy_dict()
    print len(egy_seg_dict)
    #for w in egy_seg_dict:
    #    print w, egy_seg_dict[w]
    #    tmp = sys.stdin.readline()

prefix_set = set(['w', 'f', 'l', 's', 'b', 'Al', 'k'])
egy_prefix_set = set(['mA', 'm'])
suffix_set2 = set(['t', 'p', 'At', 'A', 'yn', 'yA', 'hAt', 'y']) #NSUFF
suffix_set3 = set(['hm', 'k', 'hmA', 'n', 'hn', 'kmA', 'wn', 'wA', 'An']) #not NSUFF
egy_suffix_set = set(['$', 'Ah'])
def process_msa_buk_raw_qcri():
    with open('msa.buk.raw.qcri', 'r') as f:
        lines = f.read().splitlines()
    f = open('msa.buk.raw.post', 'w')
    for line in lines:
        new_line = ''
        data = line.split(' ')
        for w in data:
            if len(w) > 1 and w.find('+') != -1 and w[0] != '+' and w[-1] != '+':
                sw = w.split('+')
                cnt = 0
                for s in sw:
                    if s in prefix_set and cnt == 0:
                        new_line += ' ' + s + '+'
                    elif s in suffix_set or s in suffix_set2 or s in suffix_set3:
                        new_line += ' +' + s 
                    else:
                        new_line += ' ' + s
                        cnt += 1
                if cnt > 1:
                    print w
                    tmp = sys.stdin.readline()
            else:
                new_line += ' ' + w
        f.write(new_line.strip() + '\n')
    f.close()

def get_nsuff_set():
    with open('egy.gold', 'r') as f:
        lines = f.read().splitlines()
    s = set([])
    for line in lines:
        data = line.split('\t')
        for i in xrange(0, len(data), 2):
            if data[i + 1].find('SUFF') != -1: s.add(data[i])
    for w in s: print w

def process_egy_buk_raw_qcri():
    with open('egy.buk.raw.qcri', 'r') as f:
        lines = f.read().splitlines()

    f = open('egy.buk.raw.post', 'w')
    for line in lines:
        #line = line[line.find(': ') + 2:].split(' ')
        line = line.split(' ')
        line = [s for s in line if is_ascii(s) and len(s) > 0]
        if len(line) > 0:
            #f.write(' '.join(line) + '\n')
            line = ' '.join(line)
        else:
            continue
        new_line = ''
        data = line.split(' ')
        for w in data:
            if len(w) > 1 and w.find('+') != -1 and w[0] != '+' and w[-1] != '+':
                sw = w.split('+')
                cnt = 0
                for s in sw:
                    if (s in prefix_set or s in egy_prefix_set) and cnt == 0:
                        new_line += ' ' + s + '+'
                    elif s in suffix_set or s in suffix_set2 or s in suffix_set3 or s in egy_suffix_set:
                        new_line += ' +' + s
                    else:
                        new_line += ' ' + s
                        cnt += 1
                if cnt > 1:
                    print w
                    #tmp = sys.stdin.readline()
            else:
                new_line += ' ' + w
        f.write(new_line.strip() + '\n')
    f.close()

def add_count(counter, line):
    line = line.split(' ')
    for w in line:
        if not w in counter: counter[w] = 1
        else:
            cnt = counter[w]
            counter[w] = cnt + 1

def add_tag(tag_dict, line):
    #print line
    line = line.split('\t')
    for i in xrange(0, len(line), 2):
        if not line[i] in tag_dict:
            tag_dict[line[i]] = {}
        add_count(tag_dict[line[i]], line[i + 1])

def choose_frequent_tag(tag_dict):
    new_tag = {}
    for w in tag_dict:
        max_cnt = 0
        max_tag = ''
        s = 0
        d = tag_dict[w]
        for t in d:
            s += d[t]
            if d[t] > max_cnt:
                max_cnt = d[t]
                max_tag = t
        if float(max_cnt) / s > 0.8:
            new_tag[w] = t
    return new_tag

def generate_word_pairs():
    with open('msa.buk.raw.post', 'r') as f:
        lines = f.read().splitlines()
    msa_counter = {}
    for line in lines:
        add_count(msa_counter, line)
    print 'aaa'

    with open('egy.buk.raw.post', 'r') as f:
        lines = f.read().splitlines()
    egy_counter = {}
    for line in lines:
        add_count(egy_counter, line)
    print 'bbb'

    with open('msa.buk.gold.post', 'r') as f:
        lines = f.read().splitlines()
    msa_tag_dict = {}
    for line in lines:
        add_tag(msa_tag_dict, line)
    msa_tag_dict = choose_frequent_tag(msa_tag_dict)
    print 'ccc'

    with open('egy.buk.gold.post', 'r') as f:
        lines = f.read().splitlines()
    egy_tag_dict = {}
    for line in lines:
        add_tag(egy_tag_dict, line)
    egy_tag_dict = choose_frequent_tag(egy_tag_dict)
    print 'ddd'

    f = open('word_pair', 'w')
    for w in msa_counter:
        if w in egy_counter and msa_counter[w] >= 1000 and egy_counter[w] >= 1000:
            if w == 'b+':
                print w in msa_tag_dict, w in egy_tag_dict
            if w in msa_tag_dict and w in egy_tag_dict and msa_tag_dict[w] != egy_tag_dict[w]:
                continue
            elif w in msa_tag_dict and w not in egy_tag_dict:
                continue
            elif not w in msa_tag_dict and w in egy_tag_dict:
                continue
            f.write(w + ' ||| ' + w + '\n')
    f.close()

def map_new_msa_tag(tag):
    if tag == 'V':
        tag = 'VERB'
    elif tag == 'PUNC':
        tag = 'PUNCT'
    elif tag == 'ABBREV':
        tag = 'NOUN'
    elif tag.find('PART') != -1 or tag == 'JUS':
        tag = 'PART'
    elif tag == 'VSUFF':
        tag = 'NSUFF'
    elif tag == 'NOUN' or tag == 'ADJ' or tag == 'PRON' or tag == 'PREP' or tag == 'CONJ' or tag == 'NSUFF' or tag == 'DET' or tag == 'NUM' or tag == 'ADV' or tag == 'FOREIGN' or tag == 'CASE':
        pass
    else:
        print 'aaa', tag
        tmp = sys.stdin.readline()
    return tag

def process_new_msa():
    with open('train.buk', 'r') as f:
        lines = f.read().splitlines()

    new_line = ''
    stem = False
    f = open('msa.buk.gold.post', 'w')
    for i, line in enumerate(lines):
        #print line
        if line == '':
            f.write(new_line.strip() + '\n')
            new_line = ''
            stem = False
        elif line.endswith('\tO'):
            stem = False
        else:
            word, tag = line.split('\t')
            if not is_ascii(word): continue
            ending = True
            if i < len(lines) - 1 and not lines[i + 1].endswith('\tO'):
                 ending = False
            #if word == 'Al':
            #    print word in prefix_set, stem, ending
            #    tmp = sys.stdin.readline()
            if word in prefix_set and not stem and not ending:
                new_line += word + '+\t' + map_new_msa_tag(tag) + '\t'
                continue
            #print s
            if stem and (word in suffix_set or word in suffix_set2 or word in suffix_set3 or tag.endswith('SUFF')):
                new_line += '+' + word + '\t' + map_new_msa_tag(tag) + '\t'
            else:
                new_line += word + '\t' + map_new_msa_tag(tag) + '\t'
            stem = True
    f.close()

def process_dictionary():
    with open('EGY-coda-tok-POS-Top1000-LM.buk', 'r') as f:
        lines = f.read().splitlines()

    f = open('egy.dict', 'w')
    for line in lines[1:]:
        print line
        data = line.split('\t')[4]
        #if data.find(' ') != -1: raise Exception()
        data = data.split(' ')
        for d in data:
            tokens = d.split('+')
            stem = False
            word = ''
            segment = ''
            for i, token in enumerate(tokens):
                s = token.split('/')
                if word != '': word += '+'
                word += s[0]
                if (s[0] in prefix_set or s[0] in egy_prefix_set) and not stem and i < len(tokens) - 1:
                    segment += s[0] + '+\t' + map_egy_tag(s[1]) + '\t'
                    continue
                stem = True
                if i > 0 and (s[0] in suffix_set or s[0] in suffix_set2 or s[0] in suffix_set3 or s[0] in egy_suffix_set or s[1] == 'NSUFF'):
                    segment += '+' + s[0] + '\t' + map_egy_tag(s[1]) + '\t'
                else:
                    segment += s[0] + '\t' + map_egy_tag(s[1]) + '\t'
            f.write(word + '\t' + segment.strip() + '\n')
    f.close()

def oov_rate():
    with open('train.buk', 'r') as f:
        lines = f.read().splitlines()
    vocab = set([])
    for line in lines:
        if line.endswith('\tO') or line == '': continue
        vocab.add(line.split('\t')[0])
    token = 0.0
    hit = 0.0
    with open('test.buk', 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        if line.endswith('\tO') or line == '': continue
        token += 1
        if line.split('\t')[0] in vocab: hit += 1
    print token, hit, hit / token
    with open('egy.buk.gold.post', 'r') as f:
        lines = f.read().splitlines()
    token = 0.0
    hit = 0.0
    for line in lines:
        data = line.split('\t')
        for i in xrange(0, len(data), 2):
            token += 1
            if data[i] in vocab: hit += 1
    print token, hit, hit / token
    
#process_annotation()
#process_egy_tweet()
#process_ud()
#remove_vowel()
#vocab_overlap()
#process_msa_gold()
#process_msa_raw()
#process_segmentation()
#generate_spmrl_data()
#process_msa_buk_raw_qcri()
#get_nsuff_set()
#clean_egy_tweet()
#process_egy_buk_raw_qcri()
#generate_word_pairs()
#process_new_msa()
#process_dictionary()
oov_rate()
