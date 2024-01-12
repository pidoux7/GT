######################################################################################################################
###################################################### Imports #######################################################
######################################################################################################################
import numpy as np
import pandas as pd
import csv
import os
import pickle as pkl
from torch_geometric.data import Data
import torch
#import torch.nn as nn
#import torch.nn.functional as F
from tqdm import tqdm
import itertools

######################################################################################################################
###################################################### Classe ########################################################
######################################################################################################################

def creation_dic_label(path_labels):

    # creation du dictionnaire stay_id -> label
    dic_stay_id_to_label = {}
    # ouvrir le csv 
    with open(path_labels, 'r') as f:
        reader = csv.reader(f)
        labels_csv = list(reader)

    for i in tqdm(range(1,len(labels_csv))):
        dic_stay_id_to_label[int(labels_csv[i][0])] = int(labels_csv[i][1])
        
    return dic_stay_id_to_label


def recuperer_label(subject_id, dic_subject_id_icu, dic_stay_id_to_label):

    # recuperer le stay_id
    stay_id = dic_subject_id_icu[subject_id][1]
    
    # recuperer le label
    label = dic_stay_id_to_label[int(stay_id)]

    return label


def recuperer_demo(subject_id, hadm_id):

    with open('./data/csv_data_target/' + str(subject_id) + '/' + str(hadm_id) + '/demo.csv', 'r') as f:
        reader = csv.reader(f)
        demo_csv = list(reader)
    
    age = demo_csv[1][0]
    time = demo_csv[1][3]
    rang = demo_csv[1][4]
    adm_type = demo_csv[1][5]

    return int(age), int(time), int(rang), int(adm_type)
    
def recuperer_static(subject_id, hadm_id, dic_global, liste_node, liste_type):

    with open('./data/csv_data_target/' + str(subject_id) + '/' + str(hadm_id) + '/static.csv', 'r') as f:
        reader = csv.reader(f)
        static_csv = list(reader)
    df = pd.read_csv('./data/csv_data_target/' + str(subject_id) + '/' + str(hadm_id) + '/static.csv', header=None)

    #recuperer les indices des colonnes avec les 1 de la 3eme ligne avec pandas pour récuperer les valeurs de la deuxième ligne correspondantes
    third_row_int = pd.to_numeric(df.iloc[2], errors='coerce')
    indices = df.columns[third_row_int == 1].tolist()
    static = [static_csv[1][int(index)] for index in indices]

    for diag in static:
        node = int(dic_global[diag]['identifiant']) + 1 
        type_node = int(dic_global[diag]['type_code'])
        liste_node.append(node) # le dic commence à 0 mais on en a besoin pour faire le VST
        liste_type.append(type_node)

    return liste_node, liste_type


def recuperer_dynamic(subject_id, hadm_id, dic_global, liste_node, liste_type):

    with open('./data/csv_data_target/' + str(subject_id) + '/' + str(hadm_id) + '/dynamic.csv', 'r') as f:
        reader = csv.reader(f)
        dynamic_csv = list(reader)
    df = pd.read_csv('./data/csv_data_target/' + str(subject_id) + '/' + str(hadm_id) + '/dynamic.csv', header=None)

    #recuperer les indices des colonnes avec les 1 de la 3eme ligne et les indices des colonnes avec MED dans la première ligne avec pandas pour récuperer les valeurs de la deuxième ligne correspondantes
    indices_m = df.columns[df.iloc[0] == 'MEDS'].tolist()
    indices_p = df.columns[df.iloc[0] == 'PROC'].tolist()
    third_row_int = pd.to_numeric(df.iloc[2], errors='coerce')
    indices = df.columns[third_row_int == 1].tolist()

    # combiner les deux indices pour recuperer uniquement les colonnes avec MED dans la 1ere ligne et 1 dans la 3eme ligne pour recuperer la valeur de la 2eme ligne
    indices_med = list(set(indices) & set(indices_m))
    indices_proc = list(set(indices) & set(indices_p))
    dynamic_med = [dynamic_csv[1][index] for index in indices_med]
    dynamic_proc = [dynamic_csv[1][index] for index in indices_proc]

    for med in dynamic_med:
        node = int(dic_global[med]['identifiant']) + 1
        type_node = int(dic_global[med]['type_code'])
        liste_node.append(node)
        liste_type.append(type_node)

    for proc in dynamic_proc:
        node = int(dic_global[proc]['identifiant']) + 1
        type_node = int(dic_global[proc]['type_code'])
        liste_node.append(node) 
        liste_type.append(type_node)

    return liste_node, liste_type

def creation_edge_index(x, liste_type, dic_type):
    # Edges (graphe complet)
    edge_attr = []
    edge_index = []
    all_edges = []

    for i in range(len(x)):
        for j in range(i+1,len(x)):
            all_edges.append((i, j))
            edge_attr.append(dic_type[(liste_type[i], liste_type[j])])
    source, target = zip(*all_edges)

    edge_index = torch.tensor([source, target], dtype=torch.int64)
    edge_attr = torch.tensor(edge_attr, dtype=torch.int64)

    return edge_index , edge_attr


######################################################################################################################
######################################################### Main #######################################################
######################################################################################################################
    
if __name__ == '__main__':
    
    # creation du dictionnaire stay_id -> label
    path_labels = './data/labels.csv'
    dic_stay_id_to_label = creation_dic_label(path_labels)

    # import dictionnaire pickle dic_subject_id_icu
    with open('./data/dic_subject_id_icu.pkl', 'rb') as fp:
        dic_subject_id_icu = pkl.load(fp)

    # import dictionnaire pickle dic_sub_id_to_hadm_id
    with open('./data/dic_sub_id_to_hadm_id.pkl', 'rb') as fp:
        dic_sub_id_to_hadm_id = pkl.load(fp)

    # import dictionnaire pickle dic_global
    with open('./data/dic_global.pkl', 'rb') as fp:
        dic_global = pkl.load(fp)

    # nombre de noeuds dont vst
    num_nodes = len(dic_global) + 1

    # dictionnaire type correspondance
    dic_type = {(0,0): 0,
                (0,1): 0,
                (1,0): 0, 
                (0,2): 0,
                (2,0): 0,
                (0,3): 0,
                (3,0): 0,
                (1,1): 1,
                (1,2): 2,
                (2,1): 2,
                (1,3): 3,
                (3,1): 3,
                (2,2): 4,
                (2,3): 5,
                (3,2): 5,
                (3,3): 6
                }

    # creation des objets Data
    liste_dataset = []

    for patient, visistes in tqdm(dic_sub_id_to_hadm_id.items()):
        liste_patient = []

        for visite in visistes:
            visite_objet = Data()
            # recuperer le subject_id et le hadm_id
            visite_objet.subject_id = torch.tensor([patient], dtype=torch.int64)
            visite_objet.hadm_id = torch.tensor([visite], dtype=torch.int64)

            # recuperer le label
            label = recuperer_label(patient, dic_subject_id_icu, dic_stay_id_to_label)
            visite_objet.label = torch.tensor([label], dtype=torch.int64)
            
            
            # recuperer les infos de demo.csv : age, date, rang, adm_type
            age, date, rang, adm_type = recuperer_demo(patient, visite)

            visite_objet.age = torch.tensor([age], dtype=torch.int64)
            visite_objet.time = torch.tensor([date], dtype=torch.int64)
            visite_objet.rang = torch.tensor([rang], dtype=torch.int64)
            visite_objet.type = torch.tensor([adm_type], dtype=torch.int64)

            liste_node ,liste_type = [], []
            # recuperer les infos de static.csv : ajout des noeuds
            liste_node, liste_type = recuperer_static(patient, visite, dic_global, liste_node, liste_type)

            # recuperer les infos de dynamic.csv : ajout des noeuds
            liste_node, liste_type = recuperer_dynamic(patient, visite, dic_global,liste_node,liste_type)

            # ajout du VST et construction de la liste des noeuds
            visite_objet.x = torch.cat([torch.tensor([0]), torch.tensor(liste_node, dtype=torch.int64)]).reshape(1,-1)
            liste_node = [0] + liste_node
            liste_type = [0] + liste_type

            # ajout des edges
            edge_index, edge_attr = creation_edge_index(liste_node, liste_type, dic_type) 
            visite_objet.edge_index = edge_index
            visite_objet.edge_attr = edge_attr

            liste_patient.append(visite_objet)

        liste_dataset.append(liste_patient) 


with open('./data/dic_type_correspondance.pkl', 'wb') as fp:
    pkl.dump(dic_type, fp)

with open('./data/dic_stay_id_to_label.pkl', 'wb') as fp:
    pkl.dump(dic_stay_id_to_label, fp)

with open('./data/data.pkl', 'wb') as fp:
    pkl.dump(liste_dataset, fp)