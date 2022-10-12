
# -*-coding:Utf-8 -*

# Copyright: Marielle MALFANTE - GIPSA-Lab -
# Univ. Grenoble Alpes, CNRS, Grenoble INP, GIPSA-lab, 38000 Grenoble, France
# (04/2018)
#
# marielle.malfante@gipsa-lab.fr (@gmail.com)
#
# This software is a computer program whose purpose is to automatically
# processing time series (automatic classification, detection). The architecture
# is based on machine learning tools.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL

import json
from os.path import isfile, isdir
import datetime
from features import FeatureVector
import pickle
import numpy as np
from DataReadingFunctions import requestObservation, Read_Soufriere
from sklearn import preprocessing
from tools import butter_bandpass_filter
from featuresFunctions import energy, energy_u
from math import sqrt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from tools import print_cm,  butter_bandpass_filter, display_observation, getClasses
import time
from tools import extract_features
from copy import deepcopy
import pandas
import collections


# Program to find most frequent
# element in a list
def most_frequent(List):
    return max(set(List), key = List.count)

class Analyzer:
    """ Object containing the needed tools to analyze a Dataset.
    It contains the features scaler, the model, and the labels encoder,
    and can be used to train a model from supervised data.
    - scaler: None if learn has not been called, the learnt scaler otherwise
    - model: None if learn has not been called, the learnt model otherwise
    - labelEncoder: None if learn has not been called, the label encoder (label proper translation
    to int) otherwise
    - pathToCatalogue: path to the labeling catalogue. /!\ Catalogue should have a definite format.
    Check out README for more information on the catalogue shape.
    - catalogue: loaded catalogue of labels
    - _verbatim: how chatty do you want your Analyze to be?
    """

    def __init__(self, config, verbatim=0):
        """
        Initialization method
        """
        self.scaler = None
        self.model = deepcopy(config.learning['algo'])
        self.labelEncoder = None
        self.pathToCatalogue = config.general['project_root']+config.application['name'].upper()+'/'+config.learning['path_to_catalogue']
        self.catalogue = pickle.load(open(self.pathToCatalogue,'rb'))
        self.output_Features=config.learning["output_Features"]
        self._verbatim = verbatim
        if self._verbatim>0:
            print('\n\n *** ANALYZER ***')
        return

    def __repr__(self):
        """
        Representation method (transform the object to str for display)
        """
        s = 'Analyzer object with model and scaler being: '+str(self.model)+' and ' +str(self.scaler)
        s += '\nCatalogue is at %s'%self.pathToCatalogue
        return s

    def learn(self, config, verbatim=None, forModelSelection=False, sss=None, model=None, featuresIndexes=None, returnData=False,delay=3):
        """
        Method to train the analyzer.
        Labeled data are read from the catalogue, the data are preprocessed,
        features are extracted and scaled, and model is trained (with the
        stardard labels).
        All the arguments with default values are for a "non classic" use of
        the analyzer object (model selection for example)
        Return None, but can return the data and labels if specified in returnData.
        """
        if verbatim is None:
            verbatim=self._verbatim

        # Get or define usefull stuff
        features = FeatureVector(config, verbatim=verbatim)
        nData = len(self.catalogue.index)
        if returnData:
            allData = np.zeros((nData,),dtype=object)
        allLabels = np.zeros((nData,),dtype=object)
        allFeatures = np.zeros((nData,features.n_domains*features.n_features),dtype=float)

        # Read all labeled signatures (labels+data) from the catalogue, and extract features
        tStart = time.time()
        for i in range(len(self.catalogue.index)):
            if self._verbatim > 2:
                print('Data index: ', i)
            secondFloat = self.catalogue.iloc[i]['second']
            tStartSignature = datetime.datetime(int(self.catalogue.iloc[i]['year']),     \
                                                int(self.catalogue.iloc[i]['month']),    \
                                                int(self.catalogue.iloc[i]['day']),      \
                                                int(self.catalogue.iloc[i]['hour']),     \
                                                int(self.catalogue.iloc[i]['minute']),   \
                                                int(self.catalogue.iloc[i]['second'])   )   #, \
            # if int(self.catalogue.iloc[i]['length']) > delay:
            #     duration=delay
            # else:
            #     delayBis=(delay-int(self.catalogue.iloc[i]['length']))/2
            #     duration=delay
            #     tStartSignature=tStartSignature+datetime.timedelta(seconds=-delayBis)


            # duration=int(self.catalogue.iloc[i]['length'])+6
            # tStartSignature=tStartSignature+datetime.timedelta(seconds=0)

            if delay == 5:
                duration=5
                tStartSignature=tStartSignature+datetime.timedelta(seconds=-1)
            elif delay == 15:
                duration=15
                tStartSignature=tStartSignature+datetime.timedelta(seconds=-3)
            else:
                duration=int(self.catalogue.iloc[i]['length'])+6
                tStartSignature=tStartSignature+datetime.timedelta(seconds=0)




            #station="MML"
            station='TAG'#self.catalogue.iloc[i]['station']
            #path = self.catalogue.iloc[i]['path']
            #print("On est à l'itération"+str(i))
            #print(self.catalogue.iloc[i])

            (fs, signature) = Read_Soufriere(tStartSignature, duration, station, verbatim=0)
            #print(len(signature))
            # If problem
            if len(signature) < 40:
                if verbatim > 2:
                    print('Data is not considered', tStartSignature)
                allFeatures[i] = None
                allLabels[i] = None
                continue

            if returnData:
                allData[i] = signature

            # Get label and check that it is single label (multi label not supported yet)
            lab = self.catalogue.iloc[i]['Type']
            if type(lab) is list:
                print('Multi label not implemented for learning yet')
                return None
            allLabels[i] = lab

            # Filtering if needed
            f_min = 0.8#self.catalogue.iloc[i]['f1']
            f_max = 25#self.catalogue.iloc[i]['f2']
            if f_min and f_max:
                butter_order = config.analysis['butter_order']
                signature = butter_bandpass_filter(signature, f_min, f_max, fs, order=butter_order)

            # Preprocessing & features extraction

            allFeatures[i] = extract_features(config, signature.reshape(1, -1), features, fs)

        tEnd = time.time()
        if verbatim>0:
            print('Training data have been read and features have been extracted ', np.shape(allFeatures))
            print('Computation time: ', tEnd-tStart)


        # Compress labels and features in case of None values (if reading is empty for example)
        i = np.where(allLabels != np.array(None))[0]
        allFeatures = allFeatures[i]
        allLabels = allLabels[i]
        if returnData:
            allData = allData[i]

        # Transform labels
        self.labelEncoder = preprocessing.LabelEncoder().fit(allLabels)
        allLabelsStd = self.labelEncoder.transform(allLabels)
        if verbatim>0:
            print('Model will be trained on %d classes'%len(self.labelEncoder.classes_), np.unique(allLabelsStd), self.labelEncoder.classes_)

        # Scale features and store scaler
        # self.scaler = preprocessing.StandardScaler().fit(allFeatures)
        # allFeatures = self.scaler.transform(allFeatures)
        if verbatim>0:
            print('Features have been scaled')

        # Get model from learning configuration file and learn
        self.model = deepcopy(config.learning['algo'])

        if forModelSelection:
            if model is None:
                pass
            else:
                self.model = model

        tStartLearning = time.time()
        if featuresIndexes is None:
            self.model = self.model.fit(allFeatures, allLabelsStd)
        else:
            self.model = self.model.fit(allFeatures[:,featuresIndexes], allLabelsStd)
        tEndLearning = time.time()

        #  Model Evaluation (a) with score, (b) with X-validation
        if verbatim>0:
            # NB: When model is trained (and evaluated by X-validation or score),
            # threshold is NOT used. Threshold is only used when the 'unknown'
            # class can occur (and this is obvisouly not the case with supervised
            # training)
            print('Model has been trained: ', self.model)
            print('Computation time: ', tEndLearning-tStartLearning)

            if featuresIndexes is None:
                allPredictions = self.model.predict(allFeatures)
            else:
                allPredictions = self.model.predict(allFeatures[:,featuresIndexes])

            # (a) Score evaluation
            print('Model score is: ', accuracy_score(allLabelsStd,allPredictions))
            lab = list(range(len(self.labelEncoder.classes_))) # 'unknown' class not needed.
            CM = confusion_matrix(allLabelsStd,allPredictions,labels=lab)
            print('and associated confusion matrix is:')
            print_cm(CM, list(self.labelEncoder.classes_),hide_zeroes=True,max_str_label_size=2,float_display=False)

            # (b) X-validation
            sss = config.learning['cv']
            print(sss)
            CM=list()
            acc=list()
            model_Xval = deepcopy(self.model)
            for (i, (train_index, test_index)) in enumerate(sss.split(allFeatures, allLabelsStd)):
                predictionsStd = model_Xval.fit(allFeatures[train_index], allLabelsStd[train_index]).predict(allFeatures[test_index])
                predictions = self.labelEncoder.inverse_transform(predictionsStd)



                CM.append(confusion_matrix(allLabels[test_index],predictions, labels=self.labelEncoder.classes_))
                acc.append(accuracy_score(allLabels[test_index],predictions))
            print('Cross-validation results: ', np.mean(acc)*100, ' +/- ', np.std(acc)*100, ' %')
            accmean=np.mean(acc)*100
            accstd=np.std(acc)*100
            print(self.labelEncoder.classes_)
            print_cm(np.mean(CM, axis=0),self.labelEncoder.classes_,hide_zeroes=True,max_str_label_size=8,float_display=False)

        #     #np.save('/home/falcin/Desktop/test_index.npy',test_index)
        #     #np.save('/home/falcin/Desktop/true_label.npy',allLabels[test_index])
        #     #np.save('/home/falcin/Desktop/prediction.npy',predictions)
        #
        #     np.save("/Users/alwardas/Documents/Python/PhD_Alexis/GL_DataLinux/npy/featureDetec.npy",allFeatures)



        if returnData:
            return allFeatures, allLabels
        else:
            return accmean,accstd

    def apply(self,config,verbatim=0):
        pathCatalog='/Users/alwardas/Documents/Python/PhD_Alexis/GL_DataLinux/SOUFRIERE/Catalogue/NewCatPostDataset.pd'
        cat=pickle.load(open(pathCatalog,'rb'))
        features = FeatureVector(config, verbatim=verbatim)
        nData = len(cat.index)
        allLabels = np.zeros((nData,),dtype=object)
        allFeatures = np.zeros((nData,features.n_domains*features.n_features),dtype=float)

        for i in range(len(cat.index)):
            tStartSignature = datetime.datetime(int(cat.iloc[i]['year']),     \
                                                int(cat.iloc[i]['month']),    \
                                                int(cat.iloc[i]['day']),      \
                                                int(cat.iloc[i]['hour']),     \
                                                int(cat.iloc[i]['minute']),   \
                                                int(cat.iloc[i]['second'])   )
            duration = int(cat.iloc[i]['length'])
            station='TAG'

            (fs, signature) = Read_Soufriere(tStartSignature, duration, station, verbatim=0)
            if len(signature) < 40:
                if verbatim > 2:
                    print('Data is not considered', tStartSignature)
                allFeatures[i] = None
                allLabels[i] = None
                continue
            # Filtering if needed
            f_min = 0.8#self.catalogue.iloc[i]['f1']
            f_max = 25#self.catalogue.iloc[i]['f2']
            if f_min and f_max:
                butter_order = config.analysis['butter_order']
                signature = butter_bandpass_filter(signature, f_min, f_max, fs, order=butter_order)
            allFeatures[i] = extract_features(config, signature.reshape(1, -1), features, fs)
        #allFeatures = self.scaler.transform(allFeatures)
        allPredictions = self.model.predict(allFeatures)
        predictions = self.labelEncoder.inverse_transform(allPredictions)
        print(predictions)
        """
        representations
        """
        window_length_t = config.analysis['window_length']
        delta = config.analysis['delta']
        n_window = config.analysis['n_window']
        n_bands = config.analysis['nBands']
        window_length_n = int(window_length_t * 100)

        for i in range(len(cat.index)):
            tStartSignature = datetime.datetime(int(cat.iloc[i]['year']),     \
                                                int(cat.iloc[i]['month']),    \
                                                int(cat.iloc[i]['day']),      \
                                                int(cat.iloc[i]['hour']),     \
                                                int(cat.iloc[i]['minute']),   \
                                                int(cat.iloc[i]['second'])   )
            tStartSignature=tStartSignature+datetime.timedelta(seconds=-7.5)
            duration = int(cat.iloc[i]['length'])+15
            station='TAG'
            (fs, signature) = Read_Soufriere(tStartSignature, duration, station, verbatim=0)
            butter_order = config.analysis['butter_order']
            signature = butter_bandpass_filter(signature, f_min, f_max, fs, order=butter_order)
            if config.preprocessing['energy_norm']:
                E = energy(signature, arg_dict={'E_u':energy_u(signature)})
                signature = signature / sqrt(E)

            class_name = predictions[i]+' '+str(tStartSignature)
            window_size_t = window_length_n / 100
            figure_path='/Users/alwardas/Documents/Python/PhD_Alexis/GL_DataLinux/SOUFRIERE/res/Results/'+str(tStartSignature)
            display_observation(signature, 1, 25, 100, window_size_t, \
                                config, class_name, figure_path)

        return

    def save(self, config):
        """
        Method used to save the object for later use (depending on the
        application, training can take a while and you might want to save the analyzer)
        """
        path = config.general['project_root'] + config.application['name'].upper() + '/res/' + config.configuration_number + '/' + config.general['path_to_res']
        savingPath = path+'analyzer'
        pickle.dump(self.__dict__,open(savingPath,'wb'),2)
        if self._verbatim > 0:
            print('Analyzer has been saved at: ', savingPath)
        return

    def load(self, config):
        """
        Method used to load the object.
        """
        verbatim = self._verbatim
        path = config.general['project_root'] + config.application['name'].upper() + '/res/' + config.configuration_number + '/' + config.general['path_to_res']
        savingPath = path+'analyzer'
        tmp_dict = pickle.load(open(savingPath,'rb'))
        self.__dict__.update(tmp_dict)
        self._verbatim = verbatim
        if self._verbatim > 0:
            print('Analyzer has been loaded from: ', savingPath)
        return

    def saveFeatures(self,config,verbatim=None, forModelSelection=False, sss=None, model=None,
                    featuresIndexes=None, returnData=False, pathFeatures=None,station='TAG'):

        if verbatim is None:
            verbatim=self._verbatim

        # Get or define usefull stuff
        features = FeatureVector(config, verbatim=verbatim)
        nData = len(self.catalogue.index)
        if returnData:
            allData = np.zeros((nData,),dtype=object)
        allLabels = np.zeros((nData,),dtype=object)
        allFeatures = np.zeros((nData,features.n_domains*features.n_features),dtype=float)

        # Read all labeled signatures (labels+data) from the catalogue, and extract features
        tStart = time.time()
        for i in range(len(self.catalogue.index)):
            if self._verbatim > 2:
                print('Data index: ', i)
            if i % 100 == 0:
                print(i)

            #print(tStartSignature)
            #print(self.catalogue.iloc[i])
            #tStartSignature = self.catalogue['date'].iloc[i].to_pydatetime()

            tStartSignature = datetime.datetime.strptime(self.catalogue['date'].iloc[i],'%Y-%m-%d %H:%M:%S')
            duration = int(self.catalogue.iloc[i]['length'])+6
            tStartSignature=tStartSignature+datetime.timedelta(seconds=-3)

            #station='TAG'

            (fs, signature) = Read_Soufriere(tStartSignature, duration, station, verbatim=0)
            #print(len(signature))
            # If problem
            if len(signature) < 40:
                if verbatim > 2:
                    print('Data is not considered', tStartSignature)
                allFeatures[i] = None
                allLabels[i] = None
                continue

            if returnData:
                allData[i] = signature


            # Filtering if needed
            f_min = 0.8#self.catalogue.iloc[i]['f1']
            f_max = 25#self.catalogue.iloc[i]['f2']
            if f_min and f_max:
                butter_order = config.analysis['butter_order']
                signature = butter_bandpass_filter(signature, f_min, f_max, fs, order=butter_order)

            # Preprocessing & features extraction

            allFeatures[i] = extract_features(config, signature.reshape(1, -1), features, fs)

        tEnd = time.time()
        if verbatim>0:
            print('Training data have been read and features have been extracted ', np.shape(allFeatures))
            print('Computation time: ', tEnd-tStart)


        # Compress labels and features in case of None values (if reading is empty for example)
        i = np.where(allLabels != np.array(None))[0]
        allFeatures = allFeatures[i]
        if returnData:
            allData = allData[i]

        # Transform labels

        # Scale features and store scaler
        # self.scaler = preprocessing.StandardScaler().fit(allFeatures)
        # allFeatures = self.scaler.transform(allFeatures)
        if verbatim>0:
            print('Features have been scaled')



        return allFeatures, allLabels

    def learnFilter(self, config, verbatim=None, forModelSelection=False, sss=None, model=None, featuresIndexes=None, returnData=False,delay=3,f1=0.8,f2=20):
        """
        Method to train the analyzer.
        Labeled data are read from the catalogue, the data are preprocessed,
        features are extracted and scaled, and model is trained (with the
        stardard labels).
        All the arguments with default values are for a "non classic" use of
        the analyzer object (model selection for example)
        Return None, but can return the data and labels if specified in returnData.
        """
        if verbatim is None:
            verbatim=self._verbatim

        # Get or define usefull stuff
        features = FeatureVector(config, verbatim=verbatim)
        nData = len(self.catalogue.index)
        if returnData:
            allData = np.zeros((nData,),dtype=object)
        allLabels = np.zeros((nData,),dtype=object)
        allFeatures = np.zeros((nData,features.n_domains*features.n_features),dtype=float)

        # Read all labeled signatures (labels+data) from the catalogue, and extract features
        tStart = time.time()
        for i in range(len(self.catalogue.index)):
            if self._verbatim > 2:
                print('Data index: ', i)
            secondFloat = self.catalogue.iloc[i]['second']
            tStartSignature = datetime.datetime(int(self.catalogue.iloc[i]['year']),     \
                                                int(self.catalogue.iloc[i]['month']),    \
                                                int(self.catalogue.iloc[i]['day']),      \
                                                int(self.catalogue.iloc[i]['hour']),     \
                                                int(self.catalogue.iloc[i]['minute']),   \
                                                int(self.catalogue.iloc[i]['second'])   )   #, \
            #print(tStartSignature)
            duration = int(self.catalogue.iloc[i]['length'])
            tStartSignature=tStartSignature+datetime.timedelta(seconds=-delay)
            duration=duration+(2*delay)
            station=self.catalogue.iloc[i]['station']
            #path = self.catalogue.iloc[i]['path']
            #print("On est à l'itération"+str(i))
            (fs, signature) = Read_Soufriere(tStartSignature, duration, station, verbatim=0)
            #print(len(signature))
            # If problem
            if len(signature) < 40:
                if verbatim > 2:
                    print('Data is not considered', tStartSignature)
                allFeatures[i] = None
                allLabels[i] = None
                continue

            if returnData:
                allData[i] = signature

            # Get label and check that it is single label (multi label not supported yet)
            lab = self.catalogue.iloc[i]['Type']
            if type(lab) is list:
                print('Multi label not implemented for learning yet')
                return None
            allLabels[i] = lab

            # Filtering if needed
            f_min = f1
            f_max = f2
            if f_min and f_max:
                butter_order = config.analysis['butter_order']
                signature = butter_bandpass_filter(signature, f_min, f_max, fs, order=butter_order)

            # Preprocessing & features extraction

            allFeatures[i] = extract_features(config, signature.reshape(1, -1), features, fs)

        tEnd = time.time()
        if verbatim>0:
            print('Training data have been read and features have been extracted ', np.shape(allFeatures))
            print('Computation time: ', tEnd-tStart)


        # Compress labels and features in case of None values (if reading is empty for example)
        i = np.where(allLabels != np.array(None))[0]
        allFeatures = allFeatures[i]
        allLabels = allLabels[i]
        if returnData:
            allData = allData[i]

        # Transform labels
        self.labelEncoder = preprocessing.LabelEncoder().fit(allLabels)
        allLabelsStd = self.labelEncoder.transform(allLabels)
        if verbatim>0:
            print('Model will be trained on %d classes'%len(self.labelEncoder.classes_), np.unique(allLabelsStd), self.labelEncoder.classes_)

        # Scale features and store scaler
        self.scaler = preprocessing.StandardScaler().fit(allFeatures)
        allFeatures = self.scaler.transform(allFeatures)
        if verbatim>0:
            print('Features have been scaled')

        # Get model from learning configuration file and learn
        self.model = deepcopy(config.learning['algo'])

        if forModelSelection:
            if model is None:
                pass
            else:
                self.model = model

        tStartLearning = time.time()
        if featuresIndexes is None:
            self.model = self.model.fit(allFeatures, allLabelsStd)
        else:
            self.model = self.model.fit(allFeatures[:,featuresIndexes], allLabelsStd)
        tEndLearning = time.time()

        #  Model Evaluation (a) with score, (b) with X-validation
        if verbatim>0:
            # NB: When model is trained (and evaluated by X-validation or score),
            # threshold is NOT used. Threshold is only used when the 'unknown'
            # class can occur (and this is obvisouly not the case with supervised
            # training)
            print('Model has been trained: ', self.model)
            print('Computation time: ', tEndLearning-tStartLearning)

            if featuresIndexes is None:
                allPredictions = self.model.predict(allFeatures)
            else:
                allPredictions = self.model.predict(allFeatures[:,featuresIndexes])

            # (a) Score evaluation
            print('Model score is: ', accuracy_score(allLabelsStd,allPredictions))
            lab = list(range(len(self.labelEncoder.classes_))) # 'unknown' class not needed.
            CM = confusion_matrix(allLabelsStd,allPredictions,labels=lab)
            print('and associated confusion matrix is:')
            print_cm(CM, list(self.labelEncoder.classes_),hide_zeroes=True,max_str_label_size=2,float_display=False)

            # (b) X-validation
            sss = config.learning['cv']
            print(sss)
            CM=list()
            acc=list()
            model_Xval = deepcopy(self.model)
            testPred=[]
            testTrue=[]
            for (i, (train_index, test_index)) in enumerate(sss.split(allFeatures, allLabelsStd)):
                predictionsStd = model_Xval.fit(allFeatures[train_index], allLabelsStd[train_index]).predict(allFeatures[test_index])
                predictions = self.labelEncoder.inverse_transform(predictionsStd)


                #print(test_index)
                #print(type(test_index))
                #print(allLabels[test_index])
                #print(type(allLabels[test_index]))
                #print(predictions)
                print(len(predictions))
                #print()
                testPred.append(predictions)
                testTrue.append(allLabels[test_index])
                CM.append(confusion_matrix(allLabels[test_index],predictions, labels=self.labelEncoder.classes_))
                acc.append(accuracy_score(allLabels[test_index],predictions))
            print('Cross-validation results: ', np.mean(acc)*100, ' +/- ', np.std(acc)*100, ' %')
            accmean=np.mean(acc)*100
            accstd=np.std(acc)*100
            print_cm(np.mean(CM, axis=0),self.labelEncoder.classes_,hide_zeroes=True,max_str_label_size=8,float_display=False)

            #np.save('/home/falcin/Desktop/test_index.npy',test_index)
            #np.save('/home/falcin/Desktop/true_label.npy',allLabels[test_index])
            #np.save('/home/falcin/Desktop/prediction.npy',predictions)

        if returnData:
            return allData, allLabels
        else:
            return testPred, testTrue
            #return accmean, accstd

###############################################################################################################################


    def Multilearn(self, config, verbatim=None, forModelSelection=False, sss=None, model=None, featuresIndexes=None, returnData=False,delay=3):
        """
        Method to train the analyzer.
        Labeled data are read from the catalogue, the data are preprocessed,
        features are extracted and scaled, and model is trained (with the
        stardard labels).
        All the arguments with default values are for a "non classic" use of
        the analyzer object (model selection for example)
        Return None, but can return the data and labels if specified in returnData.
        """
        if verbatim is None:
            verbatim=self._verbatim

        # Get or define usefull stuff
        features = FeatureVector(config, verbatim=verbatim)
        nData = len(self.catalogue.index)
        if returnData:
            allData = np.zeros((nData,),dtype=object)
        allLabels = np.zeros((nData,),dtype=object)
        allFeatures = np.zeros((nData,features.n_domains*features.n_features*3),dtype=float)
        print('\n')
        print(allFeatures.shape)

        # Read all labeled signatures (labels+data) from the catalogue, and extract features
        tStart = time.time()
        for i in range(len(self.catalogue.index)):
            if self._verbatim > 2:
                print('Data index: ', i)
            secondFloat = self.catalogue.iloc[i]['second']
            tStartSignature = datetime.datetime(int(self.catalogue.iloc[i]['year']),     \
                                                int(self.catalogue.iloc[i]['month']),    \
                                                int(self.catalogue.iloc[i]['day']),      \
                                                int(self.catalogue.iloc[i]['hour']),     \
                                                int(self.catalogue.iloc[i]['minute']),   \
                                                int(self.catalogue.iloc[i]['second'])   )   #, \
            #print(tStartSignature)
            duration = int(self.catalogue.iloc[i]['length'])
            tStartSignature=tStartSignature+datetime.timedelta(seconds=-delay)
            duration=duration+(2*delay)

            # station=self.catalogue.iloc[i]['station']
            # station1=self.catalogue.iloc[i]['station1']
            # station2=self.catalogue.iloc[i]['station2']
            # station3='TAG'
            station='TAG'
            station1='MML'
            station2='CAG'


            #path = self.catalogue.iloc[i]['path']
            #print("On est à l'itération"+str(i))
            (fs, signature) = Read_Soufriere(tStartSignature, duration, station, verbatim=0)
            (fs, signature1) = Read_Soufriere(tStartSignature, duration, station1, verbatim=0)
            (fs, signature2) = Read_Soufriere(tStartSignature, duration, station2, verbatim=0)
        #    (fs, signature3) = Read_Soufriere(tStartSignature, duration, station3, verbatim=0)

            #print(len(signature))
            # If problem
            if len(signature) < 40 or len(signature1) < 40 or len(signature2) < 40:
                if verbatim > 2:
                    print('Data is not considered', tStartSignature)
                allFeatures[i] = None
                allLabels[i] = None
                continue

            if returnData:
                allData[i] = signature

            # Get label and check that it is single label (multi label not supported yet)
            lab = self.catalogue.iloc[i]['Type']
            if type(lab) is list:
                print('Multi label not implemented for learning yet')
                return None
            allLabels[i] = lab

            # Filtering if needed
            f_min = self.catalogue.iloc[i]['f1']
            f_max = self.catalogue.iloc[i]['f2']
            if f_min and f_max:
                butter_order = config.analysis['butter_order']
                signature = butter_bandpass_filter(signature, f_min, f_max, fs, order=butter_order)
                signature1 = butter_bandpass_filter(signature1, f_min, f_max, fs, order=butter_order)
                signature2 = butter_bandpass_filter(signature2, f_min, f_max, fs, order=butter_order)
        #        signature3 = butter_bandpass_filter(signature3, f_min, f_max, fs, order=butter_order)
            # Preprocessing & features extraction

            feat = extract_features(config, signature.reshape(1, -1), features, fs)
            feat1 = extract_features(config, signature1.reshape(1, -1), features, fs)
            feat2 = extract_features(config, signature2.reshape(1, -1), features, fs)
        #    feat3 = extract_features(config, signature3.reshape(1, -1), features, fs)
            print(feat.shape)
            print(np.concatenate((feat,feat1,feat2),axis=1).shape)
            allFeatures[i]=np.concatenate((feat,feat1,feat2),axis=1)



        tEnd = time.time()
        if verbatim>0:
            print('Training data have been read and features have been extracted ', np.shape(allFeatures))
            print('Computation time: ', tEnd-tStart)


        # Compress labels and features in case of None values (if reading is empty for example)
        i = np.where(allLabels != np.array(None))[0]
        allFeatures = allFeatures[i]
        allLabels = allLabels[i]
        if returnData:
            allData = allData[i]

        # Transform labels
        self.labelEncoder = preprocessing.LabelEncoder().fit(allLabels)
        allLabelsStd = self.labelEncoder.transform(allLabels)
        if verbatim>0:
            print('Model will be trained on %d classes'%len(self.labelEncoder.classes_), np.unique(allLabelsStd), self.labelEncoder.classes_)

        # # Scale features and store scaler
        # self.scaler = preprocessing.StandardScaler().fit(allFeatures)
        # allFeatures = self.scaler.transform(allFeatures)
        # if verbatim>0:
        #     print('Features have been scaled')

        # Get model from learning configuration file and learn
        self.model = deepcopy(config.learning['algo'])

        if forModelSelection:
            if model is None:
                pass
            else:
                self.model = model

        tStartLearning = time.time()
        if featuresIndexes is None:
            self.model = self.model.fit(allFeatures, allLabelsStd)
        else:
            self.model = self.model.fit(allFeatures[:,featuresIndexes], allLabelsStd)
        tEndLearning = time.time()

        #  Model Evaluation (a) with score, (b) with X-validation
        if verbatim>0:
            # NB: When model is trained (and evaluated by X-validation or score),
            # threshold is NOT used. Threshold is only used when the 'unknown'
            # class can occur (and this is obvisouly not the case with supervised
            # training)
            print('Model has been trained: ', self.model)
            print('Computation time: ', tEndLearning-tStartLearning)

            if featuresIndexes is None:
                allPredictions = self.model.predict(allFeatures)
            else:
                allPredictions = self.model.predict(allFeatures[:,featuresIndexes])

            # (a) Score evaluation
            print('Model score is: ', accuracy_score(allLabelsStd,allPredictions))
            lab = list(range(len(self.labelEncoder.classes_))) # 'unknown' class not needed.
            CM = confusion_matrix(allLabelsStd,allPredictions,labels=lab)
            print('and associated confusion matrix is:')
            print_cm(CM, list(self.labelEncoder.classes_),hide_zeroes=True,max_str_label_size=2,float_display=False)

            # (b) X-validation
            sss = config.learning['cv']
            print(sss)
            CM=list()
            acc=list()
            model_Xval = deepcopy(self.model)
            for (i, (train_index, test_index)) in enumerate(sss.split(allFeatures, allLabelsStd)):
                predictionsStd = model_Xval.fit(allFeatures[train_index], allLabelsStd[train_index]).predict(allFeatures[test_index])
                predictions = self.labelEncoder.inverse_transform(predictionsStd)


                #print(test_index)
                #print(type(test_index))
                #print(allLabels[test_index])
                #print(type(allLabels[test_index]))
                #print(predictions)
                #print(type(predictions))
                #print()
                CM.append(confusion_matrix(allLabels[test_index],predictions, labels=self.labelEncoder.classes_))
                acc.append(accuracy_score(allLabels[test_index],predictions))
            print('Cross-validation results: ', np.mean(acc)*100, ' +/- ', np.std(acc)*100, ' %')
            accmean=np.mean(acc)*100
            accstd=np.std(acc)*100
            print_cm(np.mean(CM, axis=0),self.labelEncoder.classes_,hide_zeroes=True,max_str_label_size=8,float_display=False)

            #np.save('/home/falcin/Desktop/test_index.npy',test_index)
            #np.save('/home/falcin/Desktop/true_label.npy',allLabels[test_index])
            #np.save('/home/falcin/Desktop/prediction.npy',predictions)

        if returnData:
            return allData, allLabels
        else:
            return allFeatures, allLabels

###########################################################

    def saveBis(self, config, verbatim=None, forModelSelection=False, sss=None, model=None, featuresIndexes=None, returnData=False,delay=3):
        """
        Method to train the analyzer.
        Labeled data are read from the catalogue, the data are preprocessed,
        features are extracted and scaled, and model is trained (with the
        stardard labels).
        All the arguments with default values are for a "non classic" use of
        the analyzer object (model selection for example)
        Return None, but can return the data and labels if specified in returnData.
        """
        if verbatim is None:
            verbatim=self._verbatim

        # Get or define usefull stuff
        features = FeatureVector(config, verbatim=verbatim)
        nData = len(self.catalogue.index)
        if returnData:
            allData = np.zeros((nData,),dtype=object)
        allLabels = np.zeros((nData,),dtype=object)
        allFeatures = np.zeros((nData,features.n_domains*features.n_features),dtype=float)
        #print(self.catalogue['Type'].value_counts)
        # Read all labeled signatures (labels+data) from the catalogue, and extract features
        tStart = time.time()
        for i in range(len(self.catalogue.index)):
            print(i)
            if self._verbatim > 2:
                print('Data index: ', i)
            secondFloat = self.catalogue.iloc[i]['second']
            tStartSignature = datetime.datetime(int(self.catalogue.iloc[i]['year']),     \
                                                int(self.catalogue.iloc[i]['month']),    \
                                                int(self.catalogue.iloc[i]['day']),      \
                                                int(self.catalogue.iloc[i]['hour']),     \
                                                int(self.catalogue.iloc[i]['minute']),   \
                                                int(self.catalogue.iloc[i]['second'])   )   #, \
            #print(tStartSignature)
            duration = int(self.catalogue.iloc[i]['length'])
            tStartSignature=tStartSignature+datetime.timedelta(seconds=-0)
            duration=duration+(delay)
            #station="MML"
            station='MML'#self.catalogue.iloc[i]['station']
            #path = self.catalogue.iloc[i]['path']
            #print("On est à l'itération"+str(i))
            #print(self.catalogue.iloc[i])

            (fs, signature) = Read_Soufriere(tStartSignature, duration, station, verbatim=0)
            #print(len(signature))
            # If problem
            if len(signature) < 40:
                if verbatim > 2:
                    print('Data is not considered', tStartSignature)
                allFeatures[i] = None
                allLabels[i] = None
                continue

            if returnData:
                allData[i] = signature

            # Get label and check that it is single label (multi label not supported yet)
            # lab = self.catalogue.iloc[i]['Type']
            # if type(lab) is list:
            #     print('Multi label not implemented for learning yet')
            #     return None
            # allLabels[i] = lab

            # Filtering if needed
            f_min = 0.8#self.catalogue.iloc[i]['f1']
            f_max = 25#self.catalogue.iloc[i]['f2']
            if f_min and f_max:
                butter_order = config.analysis['butter_order']
                signature = butter_bandpass_filter(signature, f_min, f_max, fs, order=butter_order)

            # Preprocessing & features extraction

            allFeatures[i] = extract_features(config, signature.reshape(1, -1), features, fs)

        tEnd = time.time()
        if verbatim>0:
            print('Training data have been read and features have been extracted ', np.shape(allFeatures))
            print('Computation time: ', tEnd-tStart)


        # Compress labels and features in case of None values (if reading is empty for example)
        i = np.where(allLabels != np.array(None))[0]


        if returnData:
            return allFeatures, allLabels
        else:
            return allFeatures
