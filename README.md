# FINAL OSMOSE REPORT

These files are provided by Junior Impact as part of the OSMOSE 2022 study.
Please note that due to factors explained in the final report of the study, the codes for task 4 2021 are not usable. They are provided as an indication.

## Organisation of files : 

## Task 4 : 2021

Baseline files python : /home/perretty/Task4/DESED_task/recipes/dcase2021_task4_baseline/  
Baseline config : /home/perretty/Task4/DESED_task/recipes/dcase2021_task4_baseline/confs/  
Nam files python : /home/perretty/Task4/FilterAugSED/  
Data DCASE 2021 task 4 (incomplete because permissions on youtube videos in private/delete)  

## Task5 : 2022 (same as 2021)
Baseline files python : /home/perretty/Task5/dcase-few-shot-bioacoustic/baselines/deep_learning/  
Baseline config : /home/perretty/Task5/dcase-few-shot-bioacoustic/baselines/deep_learning/config.yaml  
Data base GLIDER task5 : /home/perretty/Task5/dcase-few-shot-bioacoustic/Development_Set_Glider (and _v2)

## Data Formatting

Formatting of datasets from client data is available via the [Formatage.py](https://github.com/perrettymea/OSMOSE_JE/blob/main/Formatage_dataset.py) file. It is possible to create weak datasets (for task 4) or strong datasets with the appropriate annotations for task 5. It is possible, in particular for task 5 (few shot) to choose the number of labels in the training set and in the validation set, to choose the class which will not be present in the training set and which will constitute the validation set (non-overlapping of classes in a few-shot framework) (cf dataset2task5 function). It is also possible to create a dataset with an overlap of classes (dataset2task5_with_recouvrement). However, in this case, we leave the few-shot approach.

## Task5

Only the baseline of task 5 is available with interesting results, see [README.md](https://github.com/perrettymea/OSMOSE_JE/blob/main/task5/baselines/deep_learning/README.md) for further details.

