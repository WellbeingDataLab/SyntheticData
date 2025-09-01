"""
FUNCTIONS
"""
import pandas as pd
import numpy as np

def create_samples_cell(exercises):
    
    if 'samples.rr' in exercises.columns:
        rr = pd.json_normalize(exercises['samples.rr'])
        rr = rr.to_numpy().flatten()
        rr = [float(i['duration'].split('T')[1].split('S')[0]) for i in rr]
        avg_rr = np.mean(rr)
        sd_rr = np.std(rr)
        min_rr = np.min(rr)
        max_rr = np.max(rr)
        
    return avg_rr, sd_rr, min_rr, max_rr

def read_training(training_data):
    # Erotellaan training-datasta exercises, periodData ja loadInformation, mikäli mahdollista.
    
    # exercises-muuttujat
    duration = float('nan')
    distance = float('nan')
    sport = str('nan')
    ascent = float('nan')
    descent = float('nan')
    kiloCalories = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    heartRate_min = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    heartRate_avg = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    heartRate_max = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    speed_avg = float('nan')
    speed_max = float('nan')
    cadence_avg = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    cadence_max = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    
    avg_rr = float('nan')
    sd_rr = float('nan')
    min_rr = float('nan')
    max_rr = float('nan')
    
    # if-lause tässä nyt turha ja väärin loogisesti versus palautettavat muuttujat
    if 'exercises' in training_data:
        exercises = pd.json_normalize(training_data, record_path=['exercises'])
        try:
            dateTime = str(exercises['startTime'][0]).split('T')[0]
        except Exception:
            print("dateTime not found")
            dateTime = str("nan")
        try:
            duration = float(exercises['duration'][0].split('T')[1].split('S')[0])
        except Exception:
            print("duration not found")
        try:
            sport = str(exercises['sport'][0])
        except Exception:
            print("sport not found")

        if 'distance' in exercises.columns:
            distance = float(exercises['distance'][0])
        if 'ascent' in exercises.columns:
            ascent = float(exercises['ascent'][0])
        if 'descent' in exercises.columns:
            descent = float(exercises['descent'][0])

        try:
            kiloCalories = int(exercises['kiloCalories'][0])
        except Exception:
            print("kiloCalories not found")
        
        if 'heartRate.min' in exercises:
            heartRate_min = int(exercises['heartRate.min'][0])
        if 'heartRate.avg' in exercises:
            heartRate_avg = int(exercises['heartRate.avg'][0])
        if 'heartRate.max' in exercises:
            heartRate_max = int(exercises['heartRate.max'][0])
        if 'speed.avg' in exercises:
            speed_avg = float(exercises['speed.avg'][0])
        if 'speed.max' in exercises:
            speed_avg = float(exercises['speed.max'][0])
        if 'cadence.avg' in exercises:
            speed_avg = float(exercises['cadence.avg'][0])
        if 'cadence.max' in exercises:
            speed_avg = float(exercises['cadence.max'][0])
        
        if 'samples*' in exercises.columns:
            avg_rr, sd_rr, min_rr, max_rr = create_samples_cell(exercises)            
        
        
        d = pd.Series({sport + "_duration":duration, sport + "_distance":distance,
                     sport + "_ascent":ascent, sport + "_descent":descent, sport + "_kiloCalories":kiloCalories,
                     sport + "_heartRate_min":heartRate_min, sport + "_heartRate_avg":heartRate_avg,
                     sport + "_heartRate_max":heartRate_max, sport + "_speed_avg":speed_avg, sport + "_speed_max":speed_max,
                     sport + "_cadence_avg":cadence_avg, sport + "_cadence_max":cadence_max,
                       sport + "_avg_rr":avg_rr, sport + "_sd_rr":sd_rr, sport + "_min_rr":min_rr, sport + "_max_rr":max_rr})
        
        return d, dateTime
    
def numOfDays(date1, date2):
  #check which date is greater to avoid days output in -ve number
    if date2 > date1:   
        return (date2-date1).days
    else:
        return (date1-date2).days


def read_activity(activity_data):
    # Erotellaan training-datasta exercises, periodData ja loadInformation, mikäli mahdollista.
    
    # exercises-muuttujat
    dateTime = str('nan')
    stepCount = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    stepsDistance = float('nan')
    calories = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    sleepQuality = float('nan')
    sleepDuration = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    inactivityAlertCount = str('nan') # MUISTA LOPUKSI KÄSITELLÄ SARAKE MUUTTAEN SE NUMEERIKSEKSI!
    dailyMetMinutes = float('nan')
    
    activity = pd.json_normalize(activity_data)
    if 'date' in activity_data:
        dateTime = str(activity['date'][0])
    if 'summary.stepCount' in activity.columns:
        stepCount = int(activity['summary.stepCount'][0])
    if 'summary.stepsDistance' in activity.columns:
        stepsDistance = float(activity['summary.stepsDistance'][0])
    if 'summary.calories' in activity.columns:
        calories = int(activity['summary.calories'][0])
    if 'summary.sleepQuality' in activity.columns:
        sleepQuality = float(activity['summary.sleepQuality'][0])
    if 'summary.sleepDuration' in activity.columns:
        sleepDuration = int(activity['summary.sleepDuration'][0].split('T')[1].split('S')[0])
    if 'summary.inactivityAlertCount' in activity.columns:
        inactivityAlertCount = int(activity['summary.inactivityAlertCount'][0])
    if 'summary.dailyMetMinutes' in activity.columns:
        dailyMetMinutes = float(activity['summary.dailyMetMinutes'][0])
        #if 'activityLevels*' in activity.columns:
        #    jotain = create_activityLevels_cell(activity)       
            
    d = pd.Series({"stepCount":stepCount, "stepsDistance":stepsDistance, "calories":calories,
                   "sleepQuality":sleepQuality, "sleepDuration":sleepDuration,
                   "inactivityAlertCount":inactivityAlertCount, "dailyMetMinutes":dailyMetMinutes})
        
    return d, dateTime