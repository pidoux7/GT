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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import itertools

######################################################################################################################
##################################################### Fonctions ######################################################
######################################################################################################################
def create_graph_pad(patient):
    visite = Data(
        x = torch.tensor([0,10001]).to(torch.int64),
        edge_index = torch.tensor([[0],[1]], dtype=torch.int64),
        edge_attr = (torch.ones(1) * 10001).to(torch.int64),
        label = patient.label,
        age = torch.tensor([10001], dtype=torch.int64),
        time = torch.tensor([10001], dtype=torch.int64),
        rang = torch.tensor([10001], dtype=torch.int64),
        type = torch.tensor([10001], dtype=torch.int64),
        subject_id = patient.subject_id,
        hadm_id = torch.tensor([10001], dtype=torch.int64),
        mask_v = torch.tensor([0], dtype=torch.int64),

        )
    return visite
######################################################################################################################
##################################################### Main ###########################################################
######################################################################################################################

if __name__ == '__main__':
    
    with open('./data/data.pkl', 'rb') as f:
        dataset = pkl.load(f)
    print('dataset loaded')

    # nombre minimal de visite que l'on souhaite prendre en compte
    num_min_visites = 2
    #nombre maximal de visite que l'on souhaite prendre en compte
    num_padding = 50


    dataset_pad = []
    
    for i, patient in tqdm(enumerate(dataset)):
        patient_pad = []
        nb_visites = len(patient)
        patient.sort(key=lambda x: x.rang)
        if nb_visites >= num_min_visites:
            if nb_visites < num_padding:
                nb_padding = num_padding - nb_visites
                patient_liste = []
                patient_pad = [create_graph_pad(patient[0]) for _ in range(nb_padding)]
                rang = 0
                for visite in patient:
                    visite.x = visite.x.squeeze()
                    visite.mask_v = torch.tensor([1], dtype=torch.int64)
                    visite.time = visite.date
                    del visite.date
                    visite.rang = torch.tensor([rang], dtype=torch.int64)
                    rang += 1
                    patient_liste.append(visite)

                patient_pad_liste = patient_pad + patient_liste

            elif nb_visites > num_padding:
                patient_pad = patient[-num_padding:]
                patient_pad_liste = []

                rang = 0
                for visite in patient_pad:
                    visite.x = visite.x.squeeze()               
                    visite.mask_v = torch.tensor([1], dtype=torch.int64)
                    visite.time = visite.date
                    del visite.date
                    visite.rang = torch.tensor([rang], dtype=torch.int64)
                    rang += 1
                    patient_pad_liste.append(visite)
                
            elif nb_visites == num_padding:
                patient_pad = patient
                patient_pad_liste = []
                rang = 0
                for visite in patient_pad:
                    visite.x = visite.x.squeeze()
                    visite.mask_v = torch.tensor([1], dtype=torch.int64)
                    visite.time = visite.date
                    del visite.date
                    visite.rang = torch.tensor([rang], dtype=torch.int64)
                    rang += 1
                    patient_pad_liste.append(visite)

            dataset_pad.append(patient_pad_liste)

print('dataset_pad created')

with open('./data/data_pad.pkl', 'wb') as f:
    pkl.dump(dataset_pad, f)