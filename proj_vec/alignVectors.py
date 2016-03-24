import sys
import argparse
import numpy
import gzip
import math

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename, k):
  wordVectors = {}
  #if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  #else: fileObject = open(filename, 'r')
  f = open(filename)
  fileObject = f.readlines()
  f.close()

  cnt = 0
  for lineNum, line in enumerate(fileObject):
    cnt += 1
    if lineNum < 2:
        continue
    #line = line.strip().lower()
    line = line.strip()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)        
    ''' normalize weight vector '''
    #wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
    if cnt == k:
      break
            
  sys.stderr.write("Vectors read from: "+filename+" \n")
  print cnt, len(wordVectors)
  return wordVectors

def get_k(wordAlignFile, file1, file2, k):
  pairs = {}
  for line in open(wordAlignFile, 'r'):
    w1, w2 = line.strip().split(' ||| ')
    pairs[w1] = w2

  voc2 = set([])
  cnt = 0
  for line in open(file2, 'r'):
    cnt += 1
    if cnt <= 2:
      continue
    voc2.add(line.strip().split()[0])

  cnt = 0
  total = 0
  for line in open(file1, 'r'):
    cnt += 1
    if cnt <= 2:
      continue
    w = line.strip().split()[0]
    if w in pairs and pairs[w] in voc2:
      total += 1
    if total == k:
      break
  return cnt

def save_orig_subset_and_aligned(outFileName, lang2WordVectors, lang1AlignedVectors):
  outFile = open(outFileName+'_orig_subset.txt','w')
  for word in lang1AlignedVectors:
    outFile.write(word+' '+' '.join([str(val) for val in lang2WordVectors[word]])+'\n')
  outFile.close()  
  
  outFile = open(outFileName+'_new_aligned.txt','w')
  for word in lang1AlignedVectors:
    outFile.write(word+' '+' '.join([str(val) for val in lang1AlignedVectors[word]])+'\n')
  outFile.close()
  
def save_orig_subset_and_aligned2(outFileName, lang2WordVectors, lang1AlignedVectors):
  outFile = open(outFileName+'_orig_subset.txt','w')
  for data in lang1AlignedVectors:
    outFile.write(data[0]+' '+' '.join([str(val) for val in lang2WordVectors[data[0]]])+'\n')
  outFile.close()

  outFile = open(outFileName+'_new_aligned.txt','w')
  for data in lang1AlignedVectors:
    outFile.write(data[0]+' '+' '.join([str(val) for val in data[1]])+'\n')
  outFile.close()

def get_aligned_vectors(wordAlignFile, lang1WordVectors, lang2WordVectors):
  alignedVectors = {}
  lenLang1Vector = len(lang1WordVectors[lang1WordVectors.keys()[0]])
  pairs = []
  for line in open(wordAlignFile, 'r'):
    lang1Word, lang2Word = line.strip().split(" ||| ")
    if lang2Word not in lang2WordVectors: 
        #print 'lang2', lang2Word
        continue
    if lang1Word not in lang1WordVectors: 
        #print 'lang1', lang1Word
        continue
    alignedVectors[lang2Word] = numpy.zeros(lenLang1Vector, dtype=float)
    alignedVectors[lang2Word] += lang1WordVectors[lang1Word]
    pairs.append(lang1Word + ' ||| ' + lang2Word)

  sys.stderr.write("No. of aligned vectors found: "+str(len(alignedVectors))+'\n')
  print alignedVectors.keys()
  for pair in pairs:
    print pair
  return alignedVectors

def get_aligned_vectors2(wordAlignFile, lang1WordVectors, lang2WordVectors):
  alignedVectors = []
  lenLang1Vector = len(lang1WordVectors[lang1WordVectors.keys()[0]])
  pairs = []
  for line in open(wordAlignFile, 'r'):
    lang1Word, lang2Word = line.strip().split(" ||| ")
    if lang2Word not in lang2WordVectors:
        #print 'lang2', lang2Word
        continue
    if lang1Word not in lang1WordVectors:
        #print 'lang1', lang1Word
        continue
    alignedVectors.append([lang2Word, lang1WordVectors[lang1Word]])
    pairs.append(lang1Word + ' ||| ' + lang2Word)

  sys.stderr.write("No. of aligned vectors found: "+str(len(alignedVectors))+'\n')
  #print alignedVectors.keys()
  #fout = open(wordAlignFile + '.pair', 'w')
  #for pair in pairs:
  #  fout.write(pair + '\n')
  #fout.close()
  #print 'lalala'
  #tmp = sys.stdin.readline()
  return alignedVectors

if __name__=='__main__':
    
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--wordaligncountfile", type=str, help="Word alignment count file")
  parser.add_argument("-w1", "--wordproj1", type=str, help="Word proj of lang1")
  parser.add_argument("-w2", "--wordproj2", type=str, help="Word proj of lang2")
  parser.add_argument("-o", "--outputfile", type=str, help="Output file for storing aligned vectors")
  parser.add_argument("-k", "--topkpairs", type=int, help="Top k pairs for proj")
    
  args = parser.parse_args()
  k = get_k(args.wordaligncountfile, args.wordproj1, args.wordproj2, args.topkpairs)
  lang1WordVectors = read_word_vectors(args.wordproj1, k)
  lang2WordVectors = read_word_vectors(args.wordproj2, -1)
    
  lang1AlignedVectors = get_aligned_vectors2(args.wordaligncountfile, lang1WordVectors, lang2WordVectors)
  save_orig_subset_and_aligned2(args.outputfile, lang2WordVectors, lang1AlignedVectors)
