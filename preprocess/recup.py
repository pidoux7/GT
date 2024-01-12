#############################################################################################################
################################### Importation des librairies ##############################################
#############################################################################################################
import pandas as pd
import pickle
import shutil
import os
import csv
from tqdm import tqdm

###############################################################################################################
####################### jointure entre données ICU et CSV icu pour recup hadm_id ###############################
#############################################################################################################

def recup_hadm_id_from_stay_id(df_icu,stay_id):
    # Filtrer le DataFrame ICU par stay_id
    df_filtre_icu= df_icu[df_icu['stay_id'].isin(stay_id)]

    # Convertir la colonne 'admittime' en un objet datetime
    df_filtre_icu['intime'] = pd.to_datetime(df_filtre_icu['intime'])

    # Tri du DataFrame filtré par 'admittime' dans l'ordre croissant
    df_filtre_icu.sort_values(by='intime', ascending=True, inplace=True)

    # Utilisation de groupby pour obtenir le premier 'hadm_id' et 'subject_id' pour chaque stay_id
    resultats_icu = df_filtre_icu.groupby('stay_id').first()[['hadm_id', 'subject_id', 'intime']].reset_index()

    #renommer colonne intime en admittime
    resultats_icu.rename(columns={'intime':'admittime'}, inplace=True)

    # Utilisez groupby et idxmax pour obtenir les indices des lignes avec les dates les plus grandes
    indices_max_admittime = resultats_icu.groupby('subject_id')['admittime'].idxmax()

    # Sélectionnez les lignes correspondantes
    resultats_icu = resultats_icu.loc[indices_max_admittime]

    return resultats_icu



##############################################################################################################
################################## création des dictionnaires ################################################
#############################################################################################################


def creation_dictionnaire_hadm_id(resultats_icu):
    dic_stay_id = dict(zip(resultats_icu['stay_id'], resultats_icu['hadm_id']))
    dic_hadm_id = dict(zip(resultats_icu['hadm_id'], zip(resultats_icu['subject_id'], resultats_icu['admittime'])))
    dic_subject_id = dict(zip(resultats_icu['subject_id'], resultats_icu['admittime']))
    dic_stay_id_icu = dict(zip(resultats_icu['stay_id'], zip(resultats_icu['subject_id'], resultats_icu['hadm_id'])))
    dic_hadm_id_icu = dict(zip(resultats_icu['hadm_id'], zip(resultats_icu['subject_id'], resultats_icu['stay_id'])))
    dic_subject_id_icu = dict(zip(resultats_icu['subject_id'], zip(resultats_icu['hadm_id'], resultats_icu['stay_id'])))
    return dic_hadm_id, dic_subject_id, dic_stay_id, dic_stay_id_icu, dic_hadm_id_icu, dic_subject_id_icu


##############################################################################################################
################################## création des tuples en vu de demo #########################################
##############################################################################################################

def creation_tuples(df_hosp, dic_subject_id,dic_hadm_id):
    tuples = []
    tuples_icu = []
    hadm_id_liste=[]
    hadm_id_liste_icu = []

    # trier le dataframe par subject_id puis admittime
    df_hosp = df_hosp.groupby('subject_id').filter(lambda x: len(x) > 1)
    df_hosp.sort_values(by=['subject_id', 'admittime'], ascending=True, inplace=True)
    df_hosp['rang'] = df_hosp.groupby('subject_id').cumcount()

    #encoder le type de visite
    admission_types = df_hosp['admission_type'].unique()
    dic_type_to_int = {admission_type: idx for idx, admission_type in enumerate(admission_types)}
    dic_type_to_int['ICU'] = 9
    dict_int_to_type = {idx: admission_type for idx, admission_type in enumerate(admission_types)}
    dict_int_to_type[9] = 'ICU'

    # Parcourir le DataFrame original et vérifier chaque ligne par rapport à la limite dans le dictionnaire
    for index, row in df_hosp.iterrows():
        
        if row['subject_id'] in dic_subject_id:
            limite_admittime = pd.to_datetime(dic_subject_id[row['subject_id']])  # Convertir en Timestamp
            row_admittime = pd.to_datetime(row['admittime'])  # Convertir la colonne 'admittime' en Timestamp
            if row_admittime < limite_admittime and row['hadm_id'] not in dic_hadm_id:
                tuples.append((row['subject_id'], row['hadm_id'], row_admittime.timetuple().tm_yday , row['rang'], dic_type_to_int[row['admission_type']]))
                hadm_id_liste.append((row['hadm_id']))
                

            elif row_admittime <= limite_admittime and row['hadm_id'] in dic_hadm_id:
                tuples_icu.append((row['subject_id'], row['hadm_id'], limite_admittime.timetuple().tm_yday , row['rang'], dic_type_to_int['ICU']))                
                hadm_id_liste_icu.append((row['hadm_id']))

    return tuples, hadm_id_liste, hadm_id_liste_icu, tuples_icu, dic_type_to_int, dict_int_to_type




##############################################################################################################
################################## encodage #########################################
##############################################################################################################

def encodage_condition(fichier_source, set_encodage_condition):
    # lecture du fichier 
    with open(fichier_source, 'r') as input_file:
        csv_reader = csv.reader(input_file)
        data = list(csv_reader)

    # on ajoute les maladies dans le set
    for element in data[1]:
        set_encodage_condition.add(element)
        

    return set_encodage_condition


##############################################################################################################
################################## création des dossiers ####################################################
##############################################################################################################

def applatir_table(fichier_source,fichier_destination, set_encodage_med, set_encodage_proc):
    if os.path.exists(fichier_source):
        # Read the input CSV file
        with open(fichier_source, 'r') as input_file:
            csv_reader = csv.reader(input_file)
            data = list(csv_reader)

        # Transpose the data matrix to group values by column
        transposed_data = list(map(list, zip(*data)))

        # Calculate the sum for each column and create the 3rd line for the new CSV
        third_line = [1 if sum([float(val) if val.strip() else 0 for val in column[2:]]) > 0 else 0 for column in transposed_data]

        # Replace the previous 12 lines with the new 3rd line
        data = data[:2] + [third_line]

        # on veut recuperer les indices des colonnes MEDS et PROC dqns lq pre;iere ligne du csv
        for i in range(len(data[0])):
            if data[0][i] == 'MEDS':
                index_med = i
            
        # boucler sur les indices de ;eds pour qjouter la vqleur de lq 2eme ligne dqns le set
        for element in data[1][:index_med+1]:
            set_encodage_med.add(element)
        for element in data[1][index_med+1:]:
            set_encodage_proc.add(element)

        # Write the modified data to the output CSV file
        with open(fichier_destination, 'w', newline='') as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerows(data)

    return set_encodage_med, set_encodage_proc




def creation_dossiers(dossier_source, dossier_target,hosp,icu, tuples, tuples_icu, dic_hadm_id_final):

    ##############################################################################################################
    # Tuples

    set_encodage_condition = set() 
    set_encodage_med = set()
    set_encodage_proc = set()

    dic_sub_id_to_hadm_id = {}
    dic_hadm_id_to_sub_id = {}
    
    for subject_id, hadm_id, admittime, rang, admission_type in tqdm(tuples):

        # on verifie que dans le dossier hosp_source il y a bien le hadm_id
        dossier_data = os.path.join(dossier_source+hosp, str(hadm_id))
        if os.path.exists(dossier_data):
            # compléter le dictionnaire dic_hadm_id_to_sub_id
            dic_hadm_id_to_sub_id[hadm_id] = subject_id
            if subject_id not in dic_sub_id_to_hadm_id:
                dic_sub_id_to_hadm_id[subject_id] = [hadm_id]
            else:
                dic_sub_id_to_hadm_id[subject_id].append(hadm_id)
            
            # Créer le dossier du subject_id s'il n'existe pas déjà
            dossier_subject_id = os.path.join(dossier_target, str(subject_id))
            if not os.path.exists(dossier_subject_id):
                os.mkdir(dossier_subject_id)

            # Créer le dossier du hadm_id
            dossier_hadm_id = os.path.join(dossier_subject_id, str(hadm_id))
            if not os.path.exists(dossier_hadm_id):
                os.mkdir(dossier_hadm_id)

            # Copier les fichiers static.csv et dynamic.csv depuis le dossier source
            fichier_source = os.path.join(dossier_data, 'static.csv')
            fichier_destination = os.path.join(dossier_hadm_id, 'static.csv')
            if os.path.exists(fichier_source):
                set_encodage_condition = encodage_condition(fichier_source, set_encodage_condition)
                shutil.copy(fichier_source, fichier_destination)

            fichier_source = os.path.join(dossier_data, 'dynamic.csv')
            fichier_destination = os.path.join(dossier_hadm_id, 'dynamic.csv')
            if os.path.exists(fichier_source):
                set_encodage_med, set_encodage_proc = applatir_table(fichier_source,fichier_destination, set_encodage_med, set_encodage_proc)


            # Lire le fichier demo.csv depuis le dossier data
            fichier_demo_source = os.path.join(dossier_data, 'demo.csv')
            if os.path.exists(fichier_demo_source):
                df_demo = pd.read_csv(fichier_demo_source)

                # Créer un nouveau DataFrame avec toutes les colonnes nécessaires
                df_final = pd.DataFrame({
                    'age': df_demo['Age'],
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'date': admittime,
                    'rang': rang,
                    'type': admission_type
                })

                # Écrire le fichier demo.csv dans le dossier hadm_id avec les titres
                fichier_demo_destination = os.path.join(dossier_hadm_id, 'demo.csv')
                df_final.to_csv(fichier_demo_destination, index=False, header=True)



    ##############################################################################################################
    # Tuples_icu
                                
    for subject_id, hadm_id, admittime, rang, admission_type in tqdm(tuples_icu):
        
        # on verifie que subject_id est bien dans le dictionnaire dic_sub_id_to_hadm_id pour avoir au moins 2 dossiers hadm_id
        if subject_id in dic_sub_id_to_hadm_id:
            # on verifie que dans le dossier hosp_source il y a bien le hadm_id
            dossier_data = os.path.join(dossier_source+icu, str(dic_hadm_id_final[hadm_id][1]))
            if os.path.exists(dossier_data):
                # compléter le dictionnaire dic_hadm_id_to_sub_id
                dic_hadm_id_to_sub_id[hadm_id] = subject_id
                if subject_id not in dic_sub_id_to_hadm_id:
                    dic_sub_id_to_hadm_id[subject_id] = [hadm_id]
                else:
                    dic_sub_id_to_hadm_id[subject_id].append(hadm_id)

                # Créer le dossier du subject_id s'il n'existe pas déjà
                dossier_subject_id = os.path.join(dossier_target, str(subject_id))
                if not os.path.exists(dossier_subject_id):
                    os.mkdir(dossier_subject_id)

                # Créer le dossier du hadm_id
                dossier_hadm_id = os.path.join(dossier_subject_id, str(hadm_id))
                if not os.path.exists(dossier_hadm_id):
                    os.mkdir(dossier_hadm_id)

                # Copier les fichiers static.csv et dynamic.csv depuis le dossier data
                fichier_source = os.path.join(dossier_data, 'static.csv')
                fichier_destination = os.path.join(dossier_hadm_id, 'static.csv')
                if os.path.exists(fichier_source):
                    set_encodage_condition = encodage_condition(fichier_source, set_encodage_condition)
                    shutil.copy(fichier_source, fichier_destination)

                fichier_source = os.path.join(dossier_data, 'dynamic.csv')
                fichier_destination = os.path.join(dossier_hadm_id, 'dynamic.csv')
                if os.path.exists(fichier_source):
                    set_encodage_med, set_encodage_proc = applatir_table(fichier_source,fichier_destination, set_encodage_med, set_encodage_proc)

                # Lire le fichier demo.csv depuis le dossier data
                fichier_demo_source = os.path.join(dossier_data, 'demo.csv')
                if os.path.exists(fichier_demo_source):
                    df_demo = pd.read_csv(fichier_demo_source)

                    # Créer un nouveau DataFrame avec toutes les colonnes nécessaires
                    df_final = pd.DataFrame({
                        'age': df_demo['Age'],
                        'subject_id': subject_id,
                        'hadm_id': hadm_id,
                        'date': admittime,
                        'rang': rang,
                        'type': admission_type
                    })

                    # Écrire le fichier demo.csv dans le dossier hadm_id avec les titres
                    fichier_demo_destination = os.path.join(dossier_hadm_id, 'demo.csv')
                    df_final.to_csv(fichier_demo_destination, index=False, header=True)


    # Dictionnaire global
    dic_global = {}

    # Identifiant unique
    unique_id = 0

    # Ajout des éléments de set_encodage_condition
    for element in set_encodage_condition:
        dic_global[element] = {"identifiant": unique_id, "type": "diag", "type_code": 1}
        unique_id += 1

    # Ajout des éléments de set_encodage_med
    for element in set_encodage_med:
        dic_global[element] = {"identifiant": unique_id, "type": "med", "type_code": 2}
        unique_id += 1

    # Ajout des éléments de set_encodage_proc
    for element in set_encodage_proc:
        dic_global[element] = {"identifiant": unique_id, "type": "proc", "type_code": 3}
        unique_id += 1


    # Dictionnaire global reverse
    dic_global_reverse = {}

    for element, details in dic_global.items():
        identifiant = details["identifiant"]
        dic_global_reverse[identifiant] = {"element": element, "type": details["type"], "type_code": details["type_code"]}


    return dic_global, dic_global_reverse, dic_hadm_id_to_sub_id, dic_sub_id_to_hadm_id

    
##############################################################################################################
################################## Main ######################################################################
##############################################################################################################

if __name__ == '__main__':

    ##########################################################################################################
    # Charger CSV originaux dans un DataFrame
    df_hosp = pd.read_csv('./admissions.csv.gz', compression='gzip')
    df_icu = pd.read_csv('./icustays.csv.gz', compression='gzip')

    # Charger la liste des numéros de stay_id à partir du fichier pickle
    with open("./stay_id.pkl", "rb") as fichier_pickle:
        stay_id = pickle.load(fichier_pickle)

    print('chargement data realise')
    #######################################################################################################

    # Créer le DataFrame ICU filtré
    resultats_icu = recup_hadm_id_from_stay_id(df_icu,stay_id)
    print('creation dataframe realisee')
    
    
    # Créer les dictionnaires
    dic_hadm_id, dic_subject_id, dic_stay_id, dic_stay_id_icu, dic_hadm_id_icu, dic_subject_id_icu = creation_dictionnaire_hadm_id(resultats_icu)
    print('creation dictionnaire realisee')

    # Créer les tuples
    tuples, hadm_id_liste, hadm_id_liste_icu, tuples_icu, dic_type_to_int, dict_int_to_type = creation_tuples(df_hosp, dic_subject_id,dic_hadm_id)
    print('creation tuples realisee')


    # Créer les dossiers
    dossier_source = './csv_data_source/'
    dossier_target = './csv_data_target/'
    hosp = 'hosp_source/'
    icu = 'icu_source/'

    dic_global, dic_global_reverse, dic_hadm_id_to_sub_id, dic_sub_id_to_hadm_id = creation_dossiers(dossier_source, dossier_target, hosp, icu, tuples, tuples_icu, dic_hadm_id_icu)
    print('creation dossier realisee')

    # sauvegarde des dictionnaires

    with open("./dic_hadm_id_icu.pkl", "wb") as fichier_pickle:
        pickle.dump(dic_hadm_id_icu, fichier_pickle)

    with open("./dic_subject_id_icu.pkl", "wb") as fichier_pickle:
        pickle.dump(dic_subject_id_icu, fichier_pickle)

    with open("./dic_stay_id_icu.pkl", "wb") as fichier_pickle:
        pickle.dump(dic_stay_id_icu, fichier_pickle)


    with open("./dic_type_to_int.pkl", "wb") as fichier_pickle:
        pickle.dump(dic_type_to_int, fichier_pickle)

    with open("./dict_int_to_type.pkl", "wb") as fichier_pickle:
        pickle.dump(dict_int_to_type, fichier_pickle)


    with open("./dic_global.pkl", "wb") as fichier_pickle:
        pickle.dump(dic_global, fichier_pickle)

    with open("./dic_global_reverse.pkl", "wb") as fichier_pickle:
        pickle.dump(dic_global_reverse, fichier_pickle)


    with open("./dic_hadm_id_to_sub_id.pkl", "wb") as fichier_pickle:
        pickle.dump(dic_hadm_id_to_sub_id, fichier_pickle)
    
    with open("./dic_sub_id_to_hadm_id.pkl", "wb") as fichier_pickle:
        pickle.dump(dic_sub_id_to_hadm_id, fichier_pickle)

