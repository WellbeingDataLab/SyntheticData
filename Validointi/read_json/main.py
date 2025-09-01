
import pandas as pd
import numpy as np
import glob
import json
from datetime import datetime
import functions

if __name__ == "__main__":
    files = glob.glob("../subjects/*/*.json")
    # files = glob.glob("../subjects2/*/*.json")

    subjects = []
    for file in files:
        splitted = file.split("/")[2]
        print(splitted)
        if splitted not in subjects:
            subjects.append(splitted)
    
    main_files = ['247ohr', 'activity', 'nightly', 'sleep', 'training']

    """
    MAIN CODE PART II 5.6.2024
    """
    
    # Käydään läpi jokaisen tutkittavan data
    training = {}
    #sleep = {}
    #nightly = {}
    activity = {}
    #ohr247 = {}
    
    for s in subjects:
        for file in files:
            if s in file:
                filename = file.split('/')[3].split('-')[0]
                if "_" in filename:
                    filename = filename.split('_')[0]
                if filename in main_files:
                    if filename == "training":
                        year = int(file.split('/')[3].split('-')[2])
                        month = int(file.split('/')[3].split('-')[3])
                        if (year == 2023 and month >= 9) or year > 2023:
                            datafile = json.load(open(file))
                            training_series, dateTime = functions.read_training(datafile)
                            # training_values = training_series.values.tolist()
                            new_data = [((s, dateTime),training_series)]
                            training.update(new_data)
                            # if (s, dateTime) in training:
                            #     old_and_new = training[(s, dateTime)] + [training_values]
                            #     new_data = [((s, dateTime),old_and_new)]
                            #     training.update(new_data)
                            # else:
                            #     new_data = [((s, dateTime),[training_values])]
                            #     training.update(new_data)

                    if filename == "activity":
                        year = int(file.split('/')[3].split('-')[1])
                        month = int(file.split('/')[3].split('-')[2])
                        if (year == 2023 and month >= 9) or year > 2023:
                            datafile = json.load(open(file))
                            activity_series, dateTime = functions.read_activity(datafile)
    
                            new_data = [((s, dateTime),activity_series)]
                            activity.update(new_data)
                
                                
    # for k in training:
    #     training[k] = [training[k]]
                                
    training_df = pd.DataFrame.from_dict(training)
    training_df = training_df.transpose()
    # training_df.columns = ["training"]
    #data_df["training"] = data_df["training"].astype(object)

    activity_df = pd.DataFrame.from_dict(activity)
    activity_df = activity_df.transpose()

    # Luodaan main_df
    begin_date = datetime(2023, 9, 1)
    end_date = datetime(2024, 10, 31)
    difference = functions.numOfDays(begin_date, end_date)

    columns = pd.DataFrame(pd.date_range(begin_date, periods=difference))
    col_arr = np.array(columns)
    col_arr = [str(col[0]).split("T")[0] for col in col_arr]*len(subjects)
    # Yläpuolella on luotu sarakeotsikkorivi, joka sisältää päivämäärät.
    # Tämä on numpy array -muodossa, joten tätä voi tällaisenaan käyttää multi-index-taulukon toisen tason indeksiksi.
    # Hyödynnetään difference-muuttujaa luomaan id-array, joka sisältää tutkittavan id:n eli testitapauksessa sanan testi
    # yhtä monta kertaa kuin päiviä on.

    id_arr = [subject for subject in subjects for i in range(difference)]

    arrays = [
        id_arr,
        col_arr
    ]

    main_df = pd.DataFrame(index=arrays)
    merged_df = activity_df.join(training_df, how="outer")
    main_df = main_df.join(merged_df, how="outer")
    main_df.to_excel("main_df_19062024.xlsx")
    print("done")