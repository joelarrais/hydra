#!/usr/bin/env 1

from __future__ import division, print_function
from multiprocessing import cpu_count, Process
from scipy.fftpack import dct, idct
from sklearn.externals import joblib
import numpy
import sys
import time


def log(*objects, sep=' ', end='\n', localtime=True, cli=True, file='info.log', mode='a+'):
    if cli:
        print(*objects, end=end, flush=True)
    with open(file, mode) as logFile:
        for line in logFile:
            continue
        if localtime:
            logFile.write('[{0}] '.format(time.asctime(time.localtime(time.time())).upper()))
        print(*objects, sep=sep, end='\n' if localtime else end, file=logFile)


def calculate_score(ids, gos, features, classifier, ignore=False, default=False):
    for protein0 in ids[0]:
        for protein1 in ids[1]:
            ppi = (protein0, protein1)
            lGoPairs, lInteractionScores = [], []

            try:
                # verificar se há alguma entrada que está vazia antes!
                lGOs = [gos[ppi[0]], gos[ppi[1]]]
                bNoGO = False
            except:
                bNoGO = True

            if bNoGO:
                lGoPairs.append('NO_GO')
            else:
                for go0 in lGOs[0]:
                    for go1 in lGOs[1]:
                        if go1 < go0:
                            goPair = go1 + '_' + go0
                        else:
                            goPair = go0 + '_' + go1
                        if goPair in classifier:
                            lGoPairs.append(goPair)

            for goPair in lGoPairs:
                if not default:
                    if goPair == 'NO_GO':
                        lInteractionScores.append(0)
                        continue

                if goPair in ignore:
                    lInteractionScores.append(0)
                    continue

                if goPair in classifier:
                    score = classifier[goPair].predict_proba(numpy.concatenate((features[ppi[0]], features[ppi[1]])))[0][1]
                    lInteractionScores.append(score)
                    #print (str(score) +"--"+str(classifier[goPair].predict(numpy.concatenate((features[ppi[0]], features[ppi[1]])))[0]))
            
            if len(lInteractionScores)==0:
                score = classifier['NO_GO'].predict_proba(numpy.concatenate((features[ppi[0]], features[ppi[1]])))[0][1]
                lInteractionScores.append(score)
            log(ppi[0], ppi[1], str(numpy.amax(lInteractionScores)), sep='\t', localtime=False, file='ppis.txt', cli=False)


def read_fasta(fastaFile):
    name, seq = None, []

    for line in fastaFile:
        line = line.rstrip()
        if line.startswith('>'):
            if name:
                yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)

    if name:
        yield (name, ''.join(seq))


def read_settings(settingsFile, settings):
    for line in settingsFile:
        line = line.strip()
        if line and line[0] == '[' and line[-1] == ']':
            line = line.strip('[]').split(':')
            if line[0] in settings:
                if len(line) < 3:
                    settings[line[0]] = line[-1]
                else:
                    for item in line:
                        if item == line[0]:
                            continue
                        settings[line[0]].append(item)
            else:
                raise KeyError('invalid parameter "{0}" encountered'.format(line[0]))

    return settings


def substitution(aminoAcids):
    correspondence = {'Z':'0','B':'0','U':'0','X':'0','A': '0', 'G': '0', 'V': '0', 'I': '1', 'L': '1', 'F': '1', 'P': '1', 'Y': '2', 'M': '2', 'T': '2', 'S': '2', 'H': '3', 'N': '3', 'Q': '3', 'W': '3', 'R': '4', 'K': '4', 'D': '5', 'E': '5', 'C': '6'}
    newChain = ''

    for aminoAcid in aminoAcids:
        newChain += correspondence[aminoAcid]

    return newChain


if __name__ == '__main__':
    args = sys.argv[1:]
    nMaxFeatures = 600
    bUseUniRef, bNoGoDefault = False, True
    fData, fProtein, fSettings = '../data.dat', [], 'settings.ini'
    lProteinIDs, lInteractions, lJob = [], [], []
    dData, dProteinUniRef, dUniRefProtein, dGeneticCode, dProteinFeatures, dUniProtGO, dPPiGo = {}, {}, {}, {}, {}, {}, {}
    dSettings = {'FASTA_FILE': []}

    if args:
        if len(args) > 6:
            raise SyntaxError('too many arguments')

        bFirst = True
        lFiles, lValues = [], []

        for arg in args:
            try:
                lValues.append(bool(int(arg)))
            except ValueError:
                lFiles.append(arg)

        if (lValues and len(lValues) != 2) or (not lFiles or len(lFiles) > 3):
            raise SyntaxError('invalid combination of arguments')

        for value in lValues:
            if bFirst:
                bUseUniRef = value
                bFirst = False
            else:
                bNoGoDefault = value

        for f in lFiles:
            if f.lower().endswith('.dat'):
                fData = f
            else:
                fProtein.append(f)  # rever

        if not fProtein or len(fProtein) > 2:  # rever
            raise SyntaxError('invalid combination of arguments')
    else:
        log('Reading settings file...', end=' ')
        with open(fSettings, 'r') as settingsFile:
            dSettings = read_settings(settingsFile, dSettings)
        log('Done')

    log('Loading data from file "{0}"...'.format(fData), end=' ')
    dData = joblib.load(fData)
    log('Done')

    log('Loading UniProt IDs and sequences...', end=' ')
    for fasta in dSettings['FASTA_FILE']:
        lProteinIDs.append([])
        with open(fasta, 'r') as fastaFile:
            for name, seq in read_fasta(fastaFile):
                name = name.split('|')[1]
                if name not in dGeneticCode:
                    dGeneticCode[name] = seq.upper()
                if name not in lProteinIDs[-1]:
                    lProteinIDs[-1].append(name)
    log('Done')

    log('Calculating the most frequent features for each protein...', end=' ')
    for proteins in lProteinIDs:
        for protein in proteins:
            if protein not in dProteinFeatures:
                # The protein's sequence is obtained as a sequence of numbers ranging
                # from 0 to 6, depending on the probable aminoacid substitutions
                sequence = dGeneticCode[protein]
                pcChain = substitution(sequence)

                # The protein's sequence is converted to a 1D list/array
                pcChain = list(pcChain)
                values = [float(value) for value in pcChain]
                values = numpy.array(values, dtype='float')

                # A Discrete Cosine Transform is applied to the substitution sequence,
                # in order to determine the most frequent features present in the sequence
                transformed = dct(values)
                transformed = values[:nMaxFeatures]
                # A matrix of 600 (maxFeatures default value) is created
                zeros = numpy.zeros(nMaxFeatures)

                if len(zeros) < len(transformed):
                    c = transformed.copy()
                    c[:len(zeros)] += zeros
                else:
                    c = zeros.copy()
                    c[:len(transformed)] += transformed

                # An Inverse Discrete Cosine Transform is applied to the values
                reconstructed = idct(c)

                average = numpy.average(reconstructed)
                stdDev = numpy.std(reconstructed)

                # Values normalization; the average becomes 0 and the standard deviation 1
                for i in range(0, len(reconstructed)):
                    reconstructed[i] = (reconstructed[i] - average) / stdDev

                dProteinFeatures[protein] = reconstructed
    log('Done')

    log('Preparing UniProt information...', end=' ')
    if len(lProteinIDs) == 1:
        lProteinIDs.append(lProteinIDs[0])
    log('Done')

    log('Calculating interaction probabilities for each PPI...', end=' ')
    nProteinsPerCore = len(lProteinIDs[0]) // (cpu_count() - 1)
    for i in range(0, cpu_count()):
        lProteinIDsDivided = [lProteinIDs[0][nProteinsPerCore * i: nProteinsPerCore * (i + 1) if i < (cpu_count() - 1) else None], lProteinIDs[1]]

        j = Process(target=calculate_score, args=(lProteinIDsDivided, dData['UNIPROTGO'], dProteinFeatures, dData['CLASSIFIER'], dData['IGNORE'], bNoGoDefault))
        lJob.append(j)

    for j in lJob:
        j.start()

    for j in lJob:
        j.join()
    log('Done')
