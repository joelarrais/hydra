#!/usr/bin/env python

from __future__ import print_function
from scipy.fftpack import dct, idct
from sklearn import cross_validation, svm
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import cpu_count, Process, Queue
import numpy
import os
import random
import sys
import time
import numpy as np
import threading

try:
    import urllib.request as urllib2
except:
    import urllib2

'''

'''


def train_classifier(dGoPPiData, goPair, maxFeatures, dProteinFeatures, acd, classifier_method='RBF'):
    dataset = numpy.zeros((len(dGoPPiData[goPair]), maxFeatures * 2))
    output = numpy.zeros(len(dGoPPiData[goPair]))
    for i, ppi in enumerate(dGoPPiData[goPair]):
        proteinFeatures = []
        for item in ppi:
            if isinstance(item, int):
                output[i] = item
                break
            proteinFeatures.append(dProteinFeatures[item])
        dataset[i] = numpy.concatenate((proteinFeatures[0], proteinFeatures[1]))
    
    c, gamma = 10.0, 0.001
    C_range = np.logspace(-2, 7,num=10)
    gamma_range = np.logspace(-10, 2,num=13)

    #for c in C_range:
    #for gamma in gamma_range:
    classifier = ""
    if classifier_method == 'RBF':
        classifier = svm.SVC(C=c, gamma=gamma, kernel='rbf', probability=True)
    elif classifier_method == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=9)
    #scores = cross_validation.cross_val_score(classifier, dataset, output, cv=5)
    #f1_scores = cross_validation.cross_val_score(classifier, dataset, output, cv=5, scoring='f1')
    #precision_scores = cross_validation.cross_val_score(classifier, dataset, output, cv=5, scoring='precision')
    auc_scores = cross_validation.cross_val_score(classifier, dataset, output, cv=5, scoring='roc_auc')
    #log('       Accuracy: {0} (+/- {1})'.format(scores.mean(), scores.std()))
    #log('       Precision: {0} (+/- {1})'.format(precision_scores.mean(), precision_scores.std()))
    #log('       F1: {0} (+/- {1})'.format(f1_scores.mean(), f1_scores.std()))
    log(' AUC: {3} {1} {2} {0}'.format(auc_scores.mean(), c,gamma,goPair))
    acd.put((goPair, classifier.fit(dataset, output)))


def log(*objects, separator=' ', terminator='\n', localtime=True):
    print(*objects, sep=separator, end=terminator, flush=True)
    with open('info.log', 'a+') as logFile:
        for line in logFile:
            continue
        localTime = time.asctime(time.localtime(time.time()))
        if localtime:
            logFile.write('[{0}] '.format(localTime.upper()))
        print(*objects, sep=separator, end='\n' if localtime else terminator, file=logFile)


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


def get_index(iterable, char):
    for i in range(len(iterable)):
        if iterable[i] == char:
            return i
    return -1


def read_settings(settingsFile, settings):
    for line in settingsFile:
        start, end = get_index(line, '['), get_index(line, ']')
        if start < 0 or end < 0:
            continue
        line = line[start + 1:end].split(':')
        if line[0] in settings:
            items = line[1:] if len(line[1:]) > 1 else line[1]
            settings[line[0]] = items
        else:
            raise KeyError('invalid parameter "{0}" encountered'.format(line[0]))

    return settings


def validate_settings(settings):
    settings['TOP_GO_NUMBER'] = int(settings['TOP_GO_NUMBER'])
    for i in range(2):
        settings['PPI_NUMBER'][i] = int(settings['PPI_NUMBER'][i])
    settings['CLASSIFIER'] = settings['CLASSIFIER'].upper()

    return settings


def substitution(aminoAcids):
    correspondence = {'X':'0','A': '0', 'G': '0', 'V': '0', 'I': '1', 'L': '1', 'F': '1', 'P': '1', 'Y': '2', 'M': '2', 'T': '2', 'S': '2', 'H': '3', 'N': '3', 'Q': '3', 'W': '3', 'R': '4', 'K': '4', 'D': '5', 'E': '5', 'C': '6'}
    newChain = ''

    for aminoAcid in aminoAcids:
        newChain += correspondence[aminoAcid]

    return newChain


if __name__ == '__main__':

    for arg in sys.argv:
        if arg is sys.argv[0]:
            log('>', arg.split('\\')[-1], terminator=' ', localtime=False)
            continue
        log(arg, terminator=' ', localtime=False)
    log(localtime=False)

    args = sys.argv[1:]
    maxFeatures = 400
    settingsFilename, unirefHomology = 'settings.ini', ''
    settings = {'PPI_FILE': '', 'FASTA_FILE': '', 'UNIREF_FILE': [], 'TOP_GO_NUMBER': 0, 'PPI_NUMBER': [], 'GO': [], 'CLASSIFIER': ''}

    lInteractions, lUniRefIDs, lGoIgnore = [], {}, []
    dGeneticCode, dUniProtGO, dProteinFeatures, dGoPPiData, dGoProteinData, dClassifier = {}, {}, {}, {}, {}, {}

    log('Reading settings file...', terminator=' ')
    with open(settingsFilename, 'r') as settingsFile:
        settings = read_settings(settingsFile, settings)
    unirefHomology = 'UniRef' + settings['UNIREF_FILE'][0]
    settings['UNIREF_FILE'] = settings['UNIREF_FILE'][-1]
    settings['PPI_NUMBER'] = [int(i) for i in settings['PPI_NUMBER']]
    settings['TOP_GO_NUMBER'] = int(settings['TOP_GO_NUMBER'])
    settings['CLASSIFIER'] = settings['CLASSIFIER'].upper()
    #print(settings)
    log('Done')

    if args:
        if args[0] != '!':
            log('  -> PPIs File:  ', args[0])
            settings['PPI_FILE'] = args[0]
            log('  -> Fasta File: ', args[1])
            settings['FASTA_FILE'] = args[1]
            log('  -> Uniref File:', args[2])
            settings['UNIREF_FILE'] = args[2]
            log('  -> Number of top GOs:', args[3])
            settings['TOP_GO_NUMBER'] = int(args[3])
            log('  -> Min PPIs per GO:  ', args[4])
            settings['PPI_NUMBER'][0] = int(args[4])
            log('  -> Max PPIs per GO:  ', args[5])
            settings['PPI_NUMBER'][1] = int(args[5])
        else:
            for i in range(0, len(args)):
                if args[i].startswith('-'):
                    if args[i] in args[i + 1:]:
                        raise SyntaxError('repeated parameter "{0}" detected'.format(args[i].lstrip('-')))
                    args[i] = args[i].upper().lstrip('-')
                    if args[i] in settings:
                        if isinstance(settings[args[i]], list):
                            settings[args[i]] = []
                            for j in range(i + 1, len(args)):
                                if '|' in args[j]:
                                    args[j] = args[j].split('|')
                                for k in args[j]:
                                    if k:
                                        settings[args[i]].append(k)
                                try:
                                    if args[j + 1].startswith('-'):
                                        break
                                    if not args[j + 1].endswith('|') and not args[j + 2].startswith('|'):
                                        break
                                except IndexError:
                                    break
                        else:
                            settings[args[i]] = settings[args[i + 1]]
                    else:
                        raise KeyError('invalid parameter "{0}" detected'.format(args[i]))
            settings = validate_settings(settings)

    log('Loading PPIs...', terminator=' ')
    with open(settings['PPI_FILE'], 'r') as ppiFile:
        lInteractions = ppiFile.readlines()
    log('Done')

    log('Loading UniProt IDs and sequences...', terminator=' ')
    with open(settings['FASTA_FILE'], 'r') as fastaFile:
        for name, seq in read_fasta(fastaFile):
            name = name.split('|')[1]
            if name not in dGeneticCode:
                dGeneticCode[name] = seq
    log('Done')

    log('Fetching GO UniProt ID lists:')
    # for go in settings['GO']:
    #      log('  -> GO:{0}...'.format(go), terminator=' ')
    #      url = 'http://www.uniprot.org/uniprot/?query=go%3a{0}&force=yes&format=list'.format(go)
    #      page = urllib2.urlopen(url)
    #      # try:
    #      #     proteinList = page.read().decode('utf-8').split('\n')
    #      # except:
    #      #     proteinList = str(page.read(), encoding='utf8').split('\n')
    #      proteinList = page.read().decode('utf-8').split('\n')
    #      log('Complete')
    #      log('     Organizing the GO:{0} protein list...'.format(go), terminator=' ')
    #      for protein in proteinList:
    #          if protein not in dUniProtGO:
    #              dUniProtGO[protein] = []
    #          dUniProtGO[protein].append(go)
    #      log('Complete')
    

    for go in settings['GO']:
        if os.path.isfile("./up/" + go):
            text_file = open("./up/"+go, "r")
            proteinList = [a.strip() for a in text_file.readlines()]
            #print(proteinList[1] + len(proteinList))
            text_file.close()
        else:
            log('  -> GO:{0}...'.format(go), terminator=' ')
            url = 'http://www.uniprot.org/uniprot/?query=go%3a{0}&force=yes&format=list'.format(go)
            time.sleep(2)
            page = urllib2.urlopen(url)
            # try:
            #     proteinList = page.read().decode('utf-8').split('\n')
            # except:
            #     proteinList = str(page.read(), encoding='utf8').split('\n')
            proteinList = page.read().decode('utf-8').split('\n')
            text_file = open("./up/"+go, "w")
            for item in proteinList:
                text_file.write("%s\n" % item)
            text_file.close()

        log('     Organizing the GO:{0} protein list...'.format(go), terminator=' ')
        for protein in proteinList:
            if protein not in dUniProtGO:
                dUniProtGO[protein] = []
            dUniProtGO[protein].append(go)
        log('Complete')
    log('Done')

    log('Calculating the most frequent features for each protein...', terminator=' ')
    for ppi in lInteractions:
        ppi = ppi.split()
        for protein in ppi:
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
                transformed = values[:maxFeatures]
                # A matrix of 600 (maxFeatures default value) is created
                zeros = numpy.zeros(maxFeatures)

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


    #joblib.dump(lInteractions, 'lInteractions.dat', compress=3, cache_size=1000)

    log('Atributing PPIs to their respective GOs...', terminator=' ')
    for ppi in lInteractions:
        goPairs = []
        ppi = ppi.split()
        try:
            for go0 in dUniProtGO[ppi[0]]:
                for go1 in dUniProtGO[ppi[1]]:
                    if go1 < go0:
                        goPairs.append(go1 + '_' + go0)
                    else:
                        goPairs.append(go0 + '_' + go1)
        except KeyError:
            goPairs.append('NO_GO')

        for goPair in goPairs:
            if goPair not in dGoPPiData:
                dGoPPiData[goPair] = []
            dGoPPiData[goPair].append([ppi[0], ppi[1], 1])

            if goPair not in dGoProteinData:
                dGoProteinData[goPair] = []
            for protein in ppi:
                if protein not in dGoProteinData[goPair]:
                    dGoProteinData[goPair].append(protein)
    log('Done')

    log('Determining which GOs to discard...', terminator=' ')
    log('NGOSpairs{0}',len(dGoPPiData), terminator=' ')
    for goPair in list(dGoPPiData.keys()):
        if goPair != 'NO_GO':
            if len(dGoPPiData[goPair]) < settings['PPI_NUMBER'][0]:
                lGoIgnore.append(goPair)
                # for ppi in dGoPPiData[goPair]:
                #     dGoPPiData['NO_GO'].append(ppi)
                del dGoPPiData[goPair]
                # for protein in dGoProteinData[goPair]:
                #     if protein not in dGoProteinData['NO_GO']:
                #         dGoProteinData['NO_GO'].append(protein)
                del dGoProteinData[goPair]
    log('NGOSpairs{0}',len(dGoPPiData), terminator=' ')
    log('Done')

    log('Limiting the number of GOs...', terminator=' ')
    i, foundNoGo = 0, 0
    aux0, aux1 = {}, {}
    for goPair in sorted(dGoPPiData, key=lambda goPair: len(dGoPPiData[goPair]), reverse=True)[:settings['TOP_GO_NUMBER'] + 1]:
        if i < settings['TOP_GO_NUMBER']:
            if goPair == 'NO_GO':
                foundNoGo = 1
                continue
            aux0[goPair] = dGoPPiData[goPair]
            aux1[goPair] = dGoProteinData[goPair]
        i += 1
    aux0['NO_GO'] = dGoPPiData['NO_GO']
    aux1['NO_GO'] = dGoProteinData['NO_GO']
    for goPair in sorted(dGoPPiData, key=lambda goPair: len(dGoPPiData[goPair]), reverse=True)[settings['TOP_GO_NUMBER'] + foundNoGo:]:
        aux0['NO_GO'].extend(dGoPPiData[goPair])
        aux1['NO_GO'].extend(dGoProteinData[goPair])
    aux1['NO_GO'] = sorted(list(set(aux1['NO_GO'])))
    dGoPPiData = aux0
    dGoProteinData = aux1
    log('Done')

    log('Reducing the number of positive PPIs per GO...', terminator=' ')
    for goPair in dGoPPiData:
        if len(dGoPPiData[goPair]) > int(settings['PPI_NUMBER'][1] / 2):
            random.shuffle(dGoPPiData[goPair])
            dGoPPiData[goPair] = dGoPPiData[goPair][:int(settings['PPI_NUMBER'][1] / 2)]
    log('Done')

    log('Creating negative PPIs for each GO...', terminator=' ')
    sys.stdout.flush()
    for goPair in dGoPPiData:
        for i in range(0, len(dGoPPiData[goPair])):
            ppi = []
            for i in range(0, 2):
                ppi.append(dGoProteinData[goPair][random.randint(0, (len(dGoProteinData[goPair]) - 1))])
            ppi.append(0)
            dGoPPiData[goPair].append(ppi)
    log('Done')

    lJob = []
    log('Training a classifier for each pair of GOs:')
    log(' - Chosen classifier:', settings['CLASSIFIER'])
    acd = Queue()
    for goPair in dGoPPiData:
        log('  ->', goPair, 'with', len(dGoPPiData[goPair]), 'PPIs...')
        j = Process(target=train_classifier, args=(dGoPPiData, goPair, maxFeatures, dProteinFeatures, acd, settings['CLASSIFIER']))
        lJob.append(j)
    log('Done')

    k = 0
    for j in lJob:
        while (threading.activeCount()>32):
            time.sleep(60)
        j.start()

    
    for j in lJob:
        result = acd.get()
        dClassifier[result[0]] = result[1]

    for j in lJob:
        j.join()
    
    log('     Complete')
    log('Loading Uniprot IDs and UniRef Cluster IDs...', terminator=' ')
    #List and filter of GO terms used in the classifiers
    unique_gos = []
    for goPair in dGoPPiData:
        for i in [0, 1]:
            if goPair.split('_')[i] not in unique_gos:
                unique_gos.append(goPair.split('_')[i])
    dUniProtGO2 = {}
    for c in dUniProtGO:
        if set(dUniProtGO[c]).intersection(unique_gos):
            dUniProtGO2[c] = dUniProtGO[c]  # [int(go) for go in dUniProtGO[c]]

    #Insert protein with uniref annotation
    lunirefprot = {}
    lprotuniref = []
    with open(settings['UNIREF_FILE'], 'r') as unirefFile:
        for line in unirefFile:
            line = line.split()
            if line[1] == unirefHomology:
                uniref_id = line[2].split('_')[1]
                if uniref_id in dUniProtGO2:
                    dUniProtGO2[line[0]] = dUniProtGO2[uniref_id]
    log('Done')

    log('Dumping all data to file "data.dat"...', terminator=' ')
    #dData = {'CLASSIFIER': dClassifier, 'IGNORE': lGoIgnore, 'UNIPROTGO': dUniProtGO2}
    #joblib.dump(dData, 'data.dat', compress=3, cache_size=1000)
    log('Done')



