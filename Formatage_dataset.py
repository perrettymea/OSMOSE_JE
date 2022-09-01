# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 10:28:19 2022

@author: tperret
"""
# Import librairies

import pandas as pd
from pydub import AudioSegment
from pathlib import Path
import os
import math
import shutil


# Tâche n°4 : Strong2weak
def remove_file(path): 
    os.remove(path)
    
def strong2weak(strong_annotation, directory_strong_son, output_weak,ecraser, temps_ech=10):
    """""""""""
    Le but de cette fonction est de dégrader des annotations fortes en annotations "faibles" 
    pour tester la pertinence d'une approche weak labeling.
    Le format d'entrée est le suivant :
    dataset filename start_time end_time start_freq end_freq annotation annotator start_datetime end_datetime
    Le format voulu en sortie est le suivant:
    [filename (string)][tab][event_labels (strings)]
    Par exemple :
    Y-BJNMHMZDcU_50.000_60.000.wav Alarm_bell_ringing,Dog
    
    Le temps d'échantillonnage choisi est le même que sur la plateforme DCASE ie 10s.
    """""""""""
    if ecraser:
        #suppression du dataset précédent
         #on supprime le dataset précédent si besoin
            if os.path.exists(output_weak):
                for root_folder, folders, files in os.walk(output_weak):
                    for file in files:
                        # file path
                        file_path = os.path.join(root_folder, file)
                        remove_file(file_path)
            else:
                # file/folder is not found
                print(f'"{output_weak}" is not found')

            print("Les fichiers précédents ont été supprimées ! ")
        
    myFile = open(output_weak+"weak_annotation"+'.TXT', "w+")
    #chargement des fichiers sons et découpage en plus petits fichiers de 10s
    for i in range(0,len(strong_annotation)):
        fichier_son,start_time,end_time,annotation=strong_annotation.iloc[i]
        #print(fichier_son)
        audio = AudioSegment.from_wav(directory_strong_son+fichier_son)
        #conversion en ms
        interval = temps_ech * 1000
        start=math.ceil(start_time/10)*10*1000-10000
        end=start
        while end<=((math.ceil(end_time/10)*10+10)*1000):#10 multiples secondes suivantes ex 383 => 390
            end=start+interval
            chunk = audio[start:end] 
            #on prend le même debut de nom de fichier qu'originalement plus start/end en s
            start_file=start/1000
            end_file=end/1000
            name=Path(fichier_son).stem
            filename = name+"_"+str(int(start_file))+'_'+str(int(end_file))+'.wav'
            #print(filename)
            myFile.writelines(filename+"\t"+annotation)
            chunk.export(output_weak+filename, format ="wav") 
            #Exports to a wav file in the current path.
            #formatage des annotations faibles
            start=end
    myFile.close()
        
        
def dataset2task5(strong_annotation,INPUT_SON_dir,directory_task_5_output,nb_classes,classe_validation, nb_shot_max,nb_detection_train):
    """""""""""
    Le format d'entrée est le suivant : 
    filename    start_time  end_time    annotation
    Le format de sortie est le suivant :
    Training : Audiofilename,Starttime,Endtime,CLASS_1,CLASS_2,...,CLASS_N
    Validation : Audiofilename,Starttime,Endtime,Q
    
    Les fichiers seront annotés et en Training et en validation (pour les derniers n_shotmax fichiers .wav)
    Les fichiers .csv sont enregistrés dans le dossier directory_task_5_output
    """""""""""
    
    #TRAINING
    new_dataframe_training=strong_annotation
    for i in range(1,nb_classes+1):
        new_dataframe_training[f"CLASS_{i}"]='NEG'
    
    #ajout des valeurs positives là ou il y a un label dans strong annotation
    #on recupère les annotations différentes
    annotation = strong_annotation.groupby('annotation')['annotation'].nunique()
    for i in range(0,len(strong_annotation)):
        NUM_CLASS=1
        for n_annotation in annotation.index:
            if strong_annotation["annotation"][i]==n_annotation:
                new_dataframe_training[f"CLASS_{NUM_CLASS}"][i]='POS'
            NUM_CLASS+=1
    new_dataframe_training=new_dataframe_training.drop(columns=['annotation'])
    #on choisit une classe, par exemple la Bm.D (classe_validation : 3 la quatrième en python)
    #et on supprime cette colonne afin de ne pas avoir de recouvrement entre les classes de la validation et du training
    for i in range(0,len(annotation.index)):
        if i==classe_validation:
            new_dataframe_training2=new_dataframe_training.drop(columns=[f'CLASS_{i+1}'])
    #enregistrement des fichiers trainings
    nb_fichier=len(strong_annotation.groupby('filename')['filename'].nunique())
    lim=max(nb_fichier-nb_shot_max,nb_detection_train)
    filenames=new_dataframe_training2.groupby('filename')['filename'].nunique()
    for file_sound in filenames.index[:nb_detection_train]:
        if os.path.exists(INPUT_SON_dir+file_sound):
            dataframe=new_dataframe_training2.loc[new_dataframe_training["filename"]==file_sound]
            if len(dataframe)>=5: #on veut au moins 5 échantillons par fichier
                name=Path(file_sound).stem
                dataframe.to_csv(directory_task_5_output+"/Training_Set/"+name+'.csv', sep=',', header=True, index=False)
                shutil.copyfile(INPUT_SON_dir+file_sound, directory_task_5_output+"/Training_Set/"+file_sound)
    print("Training conversion DONE !")
    
    #VALIDATION
    #on choisit une classe, par exemple la Bm.D (classe_validation : 3 la quatrième en python)
    #et on supprime les colonnes qui ne nous intéressent pas
    for i in range(0,len(annotation.index)):
        if i!=classe_validation:
            new_dataframe_training=new_dataframe_training.drop(columns=[f'CLASS_{i+1}'])
        if i==classe_validation:
            #le nom de la colonne devient Q
            new_dataframe_training.rename(columns={f'CLASS_{i+1}':'Q'})
    for file_sound in filenames.index[lim:]:
        if os.path.exists(INPUT_SON_dir+file_sound):
            #on ne prend que les fichiers qui ont la classe considérée plus de 5 (nb shot) fois positives
            dataframe=new_dataframe_training.loc[new_dataframe_training["filename"]==file_sound]
            dataframe_POS=dataframe.loc[dataframe[f'CLASS_{classe_validation+1}']=='POS']
            dataframe=dataframe.rename(columns={f'CLASS_{classe_validation+1}':'Q'})
            if len(dataframe_POS)>=5:
                name=Path(file_sound).stem
                dataframe.to_csv(directory_task_5_output+"/Validation_Set/"+name+'.csv', sep=',', header=True, index=False)
                shutil.copyfile(INPUT_SON_dir+file_sound, directory_task_5_output+"/Validation_Set/"+file_sound)
    print("Validation conversion DONE ! ")


def dataset2task5_with_recouvrement(strong_annotation,INPUT_SON_dir,directory_task_5_output,nb_classes, nb_shot_max):
    """""""""""
    Le format d'entrée est le suivant : 
    filename    start_time  end_time    annotation
    Le format de sortie est le suivant :
    Training : Audiofilename,Starttime,Endtime,CLASS_1,CLASS_2,...,CLASS_N
    Validation : Audiofilename,Starttime,Endtime,Q
    
    Les fichiers seront annotés et en Training et en validation (pour les derniers n_shotmax fichiers .wav)
    Les fichiers .csv sont enregistrés dans le dossier directory_task_5_output
    """""""""""
    
    #TRAINING
    new_dataframe_training=strong_annotation
    for i in range(1,nb_classes+1):
        new_dataframe_training[f"CLASS_{i}"]='NEG'
    
    #ajout des valeurs positives là ou il y a un label dans strong annotation
    #on recupère les annotations différentes
    annotation = strong_annotation.groupby('annotation')['annotation'].nunique()
    for i in range(0,len(strong_annotation)):
        NUM_CLASS=1
        for n_annotation in annotation.index:
            if strong_annotation["annotation"][i]==n_annotation:
                new_dataframe_training[f"CLASS_{NUM_CLASS}"][i]='POS'
            NUM_CLASS+=1
    new_dataframe_training=new_dataframe_training.drop(columns=['annotation'])
    #on choisit une classe, par exemple la Bm.D (classe_validation : 3 la quatrième en python)
    #et on supprime cette colonne afin de ne pas avoir de recouvrement entre les classes de la validation et du training
    #for i in range(0,len(annotation.index)):
    #    if i==classe_validation:
     #       new_dataframe_training2=new_dataframe_training.drop(columns=[f'CLASS_{i+1}'])
    #enregistrement des fichiers trainings
    nb_fichier=len(strong_annotation.groupby('filename')['filename'].nunique())
    lim=nb_fichier-nb_shot_max
    filenames=new_dataframe_training.groupby('filename')['filename'].nunique()
    for file_sound in filenames.index[:]:
        if os.path.exists(INPUT_SON_dir+file_sound):
            dataframe=new_dataframe_training.loc[new_dataframe_training["filename"]==file_sound]
            if len(dataframe)>=5: #on veut au moins 5 échantillons par fichier
                name=Path(file_sound).stem
                dataframe.to_csv(directory_task_5_output+"/Training_Set/"+name+'.csv', sep=',', header=True, index=False)
                shutil.copyfile(INPUT_SON_dir+file_sound, directory_task_5_output+"/Training_Set/"+file_sound)
    print("Training conversion DONE !")
    
    #VALIDATION
    #on choisit une classe, par exemple la Bm.D (classe_validation : 3 la quatrième en python)
    #et on supprime les colonnes qui ne nous intéressent pas
    #for i in range(0,len(annotation.index)):
     #   if i!=classe_validation:
     #       new_dataframe_training=new_dataframe_training.drop(columns=[f'CLASS_{i+1}'])
     #   if i==classe_validation:
            #le nom de la colonne devient Q
      #      new_dataframe_training.rename(columns={f'CLASS_{i+1}':'Q'})
    for file_sound in filenames.index[lim:]:
        if os.path.exists(INPUT_SON_dir+file_sound):
            #on ne prend que les fichiers qui ont la classe considérée plus de 5 (nb shot) fois positives
            dataframe=new_dataframe_training.loc[new_dataframe_training["filename"]==file_sound]
            #print(dataframe)
            dataframe_POS=dataframe.loc[dataframe['CLASS_1']=='POS']
            for i in range(1,nb_classes):
                dataframe_POS=dataframe_POS.drop(columns=[f'CLASS_{i+1}'])
            data_POS = dataframe_POS.rename(columns={'CLASS_1':'Q'})
            dataframe_global=data_POS 
            for classe_validation in range(1,nb_classes):
                dataframe_POS=dataframe.loc[dataframe[f'CLASS_{classe_validation+1}']=='POS']
                for i in range(0,nb_classes):
                    if i!=classe_validation:
                        dataframe_POS=dataframe_POS.drop(columns=[f'CLASS_{i+1}'])
                data_POS = dataframe_POS.rename(columns={f'CLASS_{classe_validation+1}':'Q'})
                #print(data_POS)
                dataframe_global.append(data_POS)
                print(dataframe_global)
            
            
            #dataframe=dataframe.rename(columns={f'CLASS_{classe_validation+1}':'Q'})
            if len(dataframe_global)>=5:
                print(dataframe_global)
                name=Path(file_sound).stem
                dataframe_global.to_csv(directory_task_5_output+"/Validation_Set/"+name+'.csv', sep=',', header=True, index=False)
                shutil.copyfile(INPUT_SON_dir+file_sound, directory_task_5_output+"/Validation_Set/"+file_sound)
    print("Validation conversion DONE ! ")


if __name__ == "__main__":
    # Chargement des annotations
    #lien vers le dataset du client, annotations et fichiers .wav
    INPUT_ANNOTATION='/lab/acoustock/Bioacoustique/DATASETS/GLIDER/St_Paul_Amsterdam/APLOSE_Glider_SPAmsLF_ManualAnnotations.csv'
    INPUT_SON_dir='/lab/acoustock/Bioacoustique/DATASETS/GLIDER/St_Paul_Amsterdam/600_48000/'
    strong_annotation=pd.read_csv(INPUT_ANNOTATION,
                        header=0, delimiter=",", quotechar='"')
    strong_annotation.head(20)
    #pour avoir le nombre de classes dans le dataset
    resultat = strong_annotation.groupby('annotation')['annotation'].nunique()
    print("Classes dans ce dataset: ",resultat)
    #     Conversion de ce format d'annotation au format DCASE
    # [filename (string)][tab][onset (in seconds) (float)][tab][offset (in seconds) (float)][tab][event_label (string)]
    # Par exemple : YOTsn73eqbfc_10.000_20.000.wav 0.163 0.665 Alarm_bell_ringing
    directory_dataset="/home/perretty/Documents/DATASET"
    #one garde que les informations pertinentes pour construire le dataset
    strong_annotation=strong_annotation.drop(columns=['dataset','start_frequency','end_frequency','annotator', 'start_datetime','end_datetime'])
    strong_annotation.head(10)
    Tache4 = False
    Tache5 = True
    ecraser=False #si on a déjà un jeu de données à l'endroit de sortie
    directory_strong_son=INPUT_SON_dir
    recouvrement = False
    
    
    if Tache4 :
        output_weak=directory_dataset+"/weak/"
        strong2weak(strong_annotation, directory_strong_son, output_weak, ecraser,temps_ech=10)
    
    
    if Tache5:
    #         Le format d'annotation est le suivant :
    
    # Training annotation format : (sep=',')
    
    #     Audiofilename,Starttime,Endtime,CLASS_1,CLASS_2,...,CLASS_N 
    #     audio.wav,1.1,2.2,POS,NEG,...,NEG
        
    # Les annotations seront de type strong.
    
    # Pour le format d'annotation pour la partie validation : indique seulement la présence d'un évenement à classifier; 
    # Pas de classes
    
    #     Audiofilename,Starttime,Endtime,Q
    #     audio_val.wav,1.1,2.2,POS
        if recouvrement :
            # Il n'y a pas d'opération effectuée sur les fichiers .wav dans la fonction suivante : 
            directory_task_5_output="/home/perretty/Task5/dcase-few-shot-bioacoustic/Development_Set_Glider_with_recouvrement/"
            nb_classes=5 #le dataset contient 5 classes (différent de DCASE task5)
            nb_shot_max = 500#nombre de fichiers que l'on veut pour faire la validation
            dataset2task5_with_recouvrement(strong_annotation,INPUT_SON_dir,directory_task_5_output,nb_classes, nb_shot_max)
            #Enregistrement du jeu de données glider terminé !
            #on autorise le recourvement entre les classes de la validation et du training

        else:

        # Il n'y a pas d'opération effectuée sur les fichiers .wav dans la fonction suivante : 
            directory_task_5_output="/home/perretty/Task5/dcase-few-shot-bioacoustic/Development_Set_Glider_v2/"
            nb_classes=5 #le dataset contient 5 classes (différent de DCASE task5)
            classe_validation=0 #ici on fait la validation sur la classe 1 ie BB.Aus
            nb_detection_train = 500
            nb_shot_max = 500 #nombre de fichiers que l'on veut pour faire la validation
            dataset2task5(strong_annotation,INPUT_SON_dir,directory_task_5_output,nb_classes,classe_validation, nb_shot_max,nb_detection_train)
            #Enregistrement du jeu de données glider terminé !
        
        
    print('L\'ensemble des conversions demandées a été réalisé')