To be able to reproduce the preprocessing, please note the following.


Folder structure:
- preprocess
  - csv_data_source
     - hosp_source
     - icu_source
  - csv_data_target


The folders contain:
  - preprocess/csv_data_source/hosp_source contains all the hadm_id of all the patients.
  - preprocess/csv_data_source/icu_source contains all the stay_id of all icu.
  - preprocess/csv_data_target contains all the subject_id folders.  Within each folder, there are all the hadm_id files related to the patient (hospitalizations and ICUs).


In each hadm_id and stay_id there are 3 files:
   - demo.csv which contains the age, gender, ethnicity and insurance (for those in the source folders) and age, subject_id, hadm_id, date, rang, and type (for those in the target folders)
   - static.csv which contains the diagnostic
   - dynamic.csv which contains the medications and procedures 


The recup.py file aims to join ICU data with hospitalization data to create a time series of a patient. 
The file Create_object.py aims to construct the real dataset from the files created by recup.py (the dictionaries and the patient's medical records).
The file padding.py aims to choose the minimum number of visits (hadm_id) per patient, and if the patient has less than the defined number of visits, new 'padd visits' are generated.
The file construct_artificial_dataset.ipynb aims to create an artificial dataset in order to test the model without the real patient data. 
