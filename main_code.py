import torch
import cv2
import uuid   # Unique identifier
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import io
from PIL import Image
from pathlib import Path
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from matplotlib.colors import ListedColormap
import serial

%matplotlib inline


#INITIALIZE MODEL START

os.environ['KMP_DUPLICATE_LIB_OK']='True'
number_model = None
pest_model = None

if number_model == None:
    number_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'number_model.pt', force_reload=True)
    number_model.conf = 0.7
if pest_model == None:
    pest_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'pest_model.pt', force_reload=True)
    pest_model.conf = 0.5

#INITIALIZE MODEL END



#DEFINE FUNCTIONS START

def remove_nan(pest_list):
    occurence = pest_list.count(nan)
    for i in range(occurence):
        pest_list.remove(nan)
    return pest_list



def get_max(main):
    pestcount = []
    for i in range(len(main)):
        pestcount.append(main[i][1])
    max_value = max(pestcount)
    index = pestcount.index(max_value)
        
    return index


def add_df (plot, count, time, date):
    df2 = pd.DataFrame({'Plot: ' + str(plot) + ' Pest Count': [count],
                    'Plot: ' + str(plot) + ' Time':[time],
                    'Plot: ' + str(plot) + ' Date': [date]})
    df1 = pd.concat([df,df2])
    return df1


def add_dataframe (plot, count, time, date):
    global plots_recorded
    if plot not in plots_recorded:
        df['Plot: ' + str(plot) + ' Pest Count'] = []
        df['Plot: ' + str(plot) + ' Time'] = []
        df['Plot: ' + str(plot) + ' Date'] = []
        plots_recorded.append(plot)
        df2 = pd.DataFrame({'Plot: ' + str(plot) + ' Pest Count': [count],
                'Plot: ' + str(plot) + ' Time':[time],
                'Plot: ' + str(plot) + ' Date': [date]})
        df1 = pd.concat([df,df2])
        return df1
        
    else:
        df2 = pd.DataFrame({'Plot: ' + str(plot) + ' Pest Count': [count],
                        'Plot: ' + str(plot) + ' Time':[time],
                        'Plot: ' + str(plot) + ' Date': [date]})
        df1 = pd.concat([df,df2])
        return df1

    


def detect(image):  #FOR CAMERA INPUT
    global dummy
    values = [] #This includes the image, pestcount, time and date
    pest_count = 0
    parts = []
    total_pest = []
    x_var =[0,0,450,450] 
    y_var =[0,450,0,450]
    for i in range(4):
        frame = cv2.resize(image,(900,900))
        x = x_var[i]
        y = y_var[i]
        frame = frame[x: x + 450, y: y + 450]
        frame = cv2.resize(frame,(640,640))
        results = pest_model(frame)
        frame =  np.squeeze(results.render())
        parts.append(frame)
        labels = results.xyxyn[0][:, -1]
        #print(labels)
        total_pest += labels
        pest_count += len(labels)
    top = np.hstack((parts[0], parts[1]))
    bottom = np.hstack((parts[2], parts[3]))   
    combine = np.vstack((top,bottom))
    time_s = datetime.now() + timedelta(minutes= dummy) #USE THIS FOR TESTING (THIS ADDS 5 MINS FOR EVERY DETECTION FOR BETTER GRAPHS VISUALIZATION)
    time_s = time_s.strftime('%H:%M')
    dummy +=15
    values.append(combine)
    values.append(pest_count)
    values.append(time_s)
    values.append(datetime.now().strftime('%m/%d/%Y'))
    if 0 in total_pest and 1 in total_pest:
        values.append('Caterpillar and Flea Beetle')
    elif 0 in total_pest:
        values.append('Caterpillar')
    elif 1 in total_pest:
        values.append('Flea Beetle')
    else: values.append(None)
    return values

def pest_alert(image):
    name = 'Pest_' + datetime.now().strftime('%I_%M %p') + '.jpeg'
    im = Image.fromarray(image)
    im.save(name)
    bucket = storage.bucket()
    blob = bucket.blob('Pest/' + name)
    blob.upload_from_filename(name)
    os.remove(name)
    
def create_plot(plot):
    row = int(plot/5) + 1
    fig, axs = plt.subplots(nrows= row, ncols=5)
    fig.suptitle('Grid of Plots')
    fig.set_figheight(10)
    fig.set_figwidth(20)
    
    iterate = 0
    for i in range (row): #THIS LOOP CREATES SUBPLOT (5 Columns Every Row)
        for j in range(5):
            if iterate == plot:
                break
            else:
                try:
                    axs[j].title.set_text("Plot " + str(plots_recorded[j]))
                    axs[j].plot(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist(), df['Plot: ' +  str(plots_recorded[j]) + ' Pest Count'].tolist())
                    axs[j].set_xticks(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist()[::2])
                    axs[j].set_xticklabels(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist()[::2], rotation=45)
                    
                    #frequency = 2
                    #plt.xticks(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist()[::frequency], df['Plot: ' +  str(plots_recorded[j]) + ' Pest Count'].tolist()[::frequency])
                    iterate +=1
                except TypeError:
                    pass
    
                
    for i in range (4, int(int(str(plot/5)[2])/2) -1   , -1): #THIS DELETES THE EXCESS SUBPLOTS
        fig.delaxes(axs[row - 1][i])
        
    name = 'Pest_Graphs.jpg'
    fig.savefig(name)
    path_on_cloud = "stats/" + name
    path_local = name
    storage.child(path_on_cloud).put(path_local)
    print(path_on_cloud)
    os.remove(name)
    print('CREATE PLOT DATA SUCCESS')
       
def create_plot_single(plot):
    fig, axs = plt.subplots(nrows= 1, ncols = plot)
    fig.suptitle('Grid of Plots')
    fig.set_figheight(5)
    fig.set_figwidth(20)
    iterate = 0
    for j in range(plot):
        if iterate == plot:
            break
        else:
            if plot == 1:
                fig.set_figheight(5)
                fig.set_figwidth(5)
                axs.plot(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist(), df['Plot: ' +  str(plots_recorded[j]) + ' Pest Count'].tolist())
                axs.set_xticks(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist()[::2])
                axs.set_xticklabels(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist()[::2], rotation=45)
                axs.title.set_text("Plot " + str(plots_recorded[j]))
            else:
                #y = func(x)
                try:
                    axs[j].title.set_text("Plot " + str(plots_recorded[j]))
                    axs[j].plot(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist(), df['Plot: ' +  str(plots_recorded[j]) + ' Pest Count'].tolist())
                    axs[j].set_xticks(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist()[::2])
                    axs[j].set_xticklabels(df['Plot: ' + str(plots_recorded[j]) + ' Time'].tolist()[::2], rotation=45)
                    
                    
                except TypeError:
                    pass
            
    name = 'Pest_Graphs.jpg'
    fig.savefig(name)
    path_on_cloud = "stats/" + name
    path_local = name
    storage.child(path_on_cloud).put(path_local)
    print(path_on_cloud)
    os.remove(name)
    print('CREATE SINGLE PLOT DATA SUCCESS')
            
                            
            
            
            
#Main data Transfer
def SendData(img_array,img_name, Instance, Plot, Type):  
    Test = {
    'Date' : datetime.now().strftime('%m-%d-%Y'),
    'Field' : '2',
    'Image' : img_name,
    'Instance' : Instance,
    'Plot' : str(Plot),
    'Type' : Type
    }
    
    doc_ref = db.collection(u'Detected').add(Test) # send the data
    #doc_ref.set(Test)
    name = 'Pest_'+ datetime.now().strftime('%m-%d-%Y')+ '_'  + datetime.now().strftime('%H_%M_%S') + '.jpg'
    im = Image.fromarray(img_array)
    im.save(name)
    path_on_cloud = "Pest/" + name
    path_local = name
    storage.child(path_on_cloud).put(path_local)
    print(path_on_cloud)
    os.remove(name)
    
    
    
def create_heatmap(plot):
    pest_column =[]
    pest_values = []
    plot_count = []
    for i in range(plot):
        print(plots_recorded[i])
        pest_column.append('Plot: ' +  str(plots_recorded[i]) + ' Pest Count')
    for i in range(plot):
        plot_count.append(plots_recorded[i])
        pest_values.append(df[pest_column].max()[i])
    heat_df = pd.DataFrame([pest_values], columns=plot_count)
    fig =plt.figure(figsize = (10,3))
    p1 = sns.heatmap(heat_df,cmap=ListedColormap(['#00fc3b','#8ffc00','#ffee00','#ffa30f','#ff6b0f', '#ff2f00','#fc0000','#c90f02','#b80704','#330401']))
    name = 'heatmap.jpg'
    fig.savefig(name)
    path_on_cloud = "stats/" + name
    path_local = name
    storage.child(path_on_cloud).put(path_local)
    print(path_on_cloud)
    os.remove(name)
    print('HEATMAP SUCCESS')
    
    
def create_prediction_single_ETS(plot):
    main = []
    for i in range(plot):
        pest_count = df['Plot: ' + str(plots_recorded[i]) + ' Pest Count'].tolist()
        x = df ['Plot: ' + str(plots_recorded[i]) + ' Time'].tolist()
        pest = pd.Series(pest_count, index=x)
        temp = pest
        main.append(temp)
  #print(len(main))
    fig, axs = plt.subplots(plot)
    fig.set_figheight(10)
    fig.set_figwidth(5) 
    if plot == 1:
        fig.set_figheight(5)
        fig.set_figwidth(5) 
        for i in range(plot):
            model = ETSModel(main[i])
            fit = model.fit(maxiter=10000)
            main[i].plot(ax=axs,label="data")
            fit.fittedvalues.plot(ax=axs,label="statsmodels fit")
            plt.ylabel("Pest Count")
            plt.xlabel("Time")
            axs.title.set_text("Plot " + str(plots_recorded[i]))
            # obtained from R
            params_R = [0.99989969, 0.11888177503085334, 0.80000197, 36.46466837, 34.72584983]
            yhat = model.smooth(params_R).fittedvalues
            yhat.plot(ax=axs,label="R fit", linestyle="--")
        axs.legend()
        
        
    else:
        for i in range(plot):
            model = ETSModel(main[i])
            fit = model.fit(maxiter=10000)
            main[i].plot(ax=axs[i],label="data")
            fit.fittedvalues.plot(ax=axs[i],label="statsmodels fit")
            plt.ylabel("Pest Count")
            plt.xlabel("Time")
            axs[i].title.set_text("Plot " + str(plots_recorded[i]))
            # obtained from R
            params_R = [0.99989969, 0.11888177503085334, 0.80000197, 36.46466837, 34.72584983]
            yhat = model.smooth(params_R).fittedvalues
            yhat.plot(ax=axs[i],label="R fit", linestyle="--")
        axs[0].legend()
    
    
    name = 'prediction.jpg'
    fig.savefig(name)
    path_on_cloud = "stats/" + name
    path_local = name
    storage.child(path_on_cloud).put(path_local)
    print(path_on_cloud)
    os.remove(name)
    print('SINGLE ETS SUCCESS')
    
def create_prediction_ETS(plot):
    main = []
    
    for i in range(plot):
        pest_count = df['Plot: ' + str(plots_recorded[i]) + ' Pest Count'].tolist()
        x = df ['Plot: ' + str(plots_recorded[i])  + ' Time'].tolist()
        pest = pd.Series(pest_count, index=x)
        temp = pest
        main.append(temp)
    #print(len(main))
    row = int(plot/5) + 1
    fig, axs = plt.subplots(nrows= row, ncols=5)
    fig.set_figheight(10)
    fig.set_figwidth(20) 
    iterate = 0
    for i in range(row):
        for j in range(5):
            if iterate == plot:
                break
            else:
                model = ETSModel(main[j])
                fit = model.fit(maxiter=10000)
                main[j].plot(ax=axs[i,j],label="data")
                fit.fittedvalues.plot(ax=axs[i,j],label="statsmodels fit")
                plt.ylabel("Pest Count")
                plt.xlabel("Time")
                axs[i,j].title.set_text("Plot " + str(plots_recorded[j]))
                # obtained from R
                params_R = [0.99989969, 0.11888177503085334, 0.80000197, 36.46466837, 34.72584983]
                yhat = model.smooth(params_R).fittedvalues
                yhat.plot(ax=axs[i,j],label="R fit", linestyle="--")
            
                iterate+=1
    axs[0,0].legend()

    for i in range (4, int(int(str(plot/5)[2])/2) -1   , -1): #THIS DELETES THE EXCESS SUBPLOTS
        fig.delaxes(axs[row - 1][i])
        
    name = 'prediction.jpg'
    fig.savefig(name)
    path_on_cloud = "stats/" + name
    path_local = name
    storage.child(path_on_cloud).put(path_local)
    print(path_on_cloud)
    os.remove(name)
    print('ETS SUCCESS')
    
def create_prediction_SES(plot):
    total = 0 
    for i in range (plot):
        data = pd.Series(df['Plot: '  + str(plots_recorded[i]) + ' Pest Count'].tolist(), df['Plot: ' + str(plots_recorded[i]) + ' Time'].tolist())
        alpha = 0.2
        ses = SimpleExpSmoothing(data)
        model = ses.fit(smoothing_level = alpha, optimized = False)
        forecast = model.forecast()
        total += forecast
    print(float((total/plot)))
        #Set the capital field
    doc_ref = db.collection(u'Statsdata').document(u'data')
    new = "Pest count prediction: " + str (float((total/plot)))
    doc_ref.update({u'value': new})
    print('SES SUCCESS')
                    


#DEFINE FUNCTIONS END

#FIREBASE START

import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

config = { 
    "apiKey": "AIzaSyCVHKh14cQYOtpsuhfdYFDnPXIQrW4zdJY",
    "authDomain": "pestdetection-ea9ef.firebaseapp.com",
    "projectId": "pestdetection-ea9ef",
    "storageBucket": "pestdetection-ea9ef.appspot.com",
    "messagingSenderId": "760821888728",
    "appId": "1:760821888728:web:a123bed5a194db1d0989ad",
    "measurementId": "G-M9ZQCG3WGB",
    "serviceAccount": "serviceAccount.json",
    "databaseURL": "https://pestdetection-ea9ef-default-rtdb.asia-southeast1.firebasedatabase.app/"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

#FIREBASE END




#MAIN ALGO START

#ON EDIT


number_model.conf = 0.7
dummy = 0


df = pd.DataFrame()
plots_recorded = []
timer = 0
line = ''
repeat = True
rotate = False
plot = None

while True:
    while repeat == True:
        while True:
            try:
                #if __name__ == '__main__':
                ser = serial.Serial(port='COM4', baudrate =9600, timeout=1)
                ser.reset_input_buffer()
                print('Arduino Connected')
                break
                
            except:
                print('Waiting for Arduino to start')
                time.sleep(1)
         
        #ser.flush()
        #ser.write(b"Start Running!") 
        #time.sleep(0.5)
        #ser.write(b"Start Running!") 
        #time.sleep(0.5)
        #ser.write(b"Start Running!") 
   

        print("Robot Moving")
    
            
        cam = cv2.VideoCapture(0)      
        time.sleep(1)
        ser.flush()
        ser.write(b"Start Running!") 
        time.sleep(0.5)
        ser.write(b"Start Running!") 
        time.sleep(0.5)
        ser.write(b"Start Running!") 
        #line = ser.readline().decode('utf-8').rstrip()
        #print(line)
        #ser.close()
        
        
        
        
        
        while cam.isOpened():
            ret, frame = cam.read()
                
            # Make detections 
            results = number_model(frame)
            
            cv2.imshow('YOLO', np.squeeze(results.render()))
            
            detections = results.xyxyn[0][:, -1]
            if len(detections) == 1:
                xmax_value = results.pandas().xyxy[0]['xmax']
                if xmax_value.item() > 300 and xmax_value.item() < 400:
                    repeat = False
                    plot = int(detections[0])
                    if plot == 0:
                            plot = 10
                            
                    if plot not in plots_recorded:
                        plots_recorded.append(plot)
                    else:
                        pass
                    plots_recorded = sorted(plots_recorded)
                    break
            if timer == 20:
                line = ser.readline().decode('utf-8').rstrip()
                timer = 0
            if line == 'Rotating':
                print('Rotating Robot')
                repeat = True
                rotate = True
                break
            timer +=1
            
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        cam.release()
        cv2.destroyAllWindows()
        
        
        line = ''
        timer = 0
        if rotate == True:
            while True:
                line = ser.readline().decode('utf-8').rstrip()
                if line == 'Done Rotating':
                    rotate = False
                    print(line)
                    break
                else : pass
        ser.close()    
    
    
    
    repeat = True
    while True:
        try:
            #if __name__ == '__main__':
            ser = serial.Serial(port='COM4', baudrate =9600, timeout=1)
            ser.reset_input_buffer()
            print('Arduino Connected')
            break
            
        except:
            print('Waiting for Arduino to start')
            time.sleep(1)
    print('Robot Stop Moving')        
    ser.flush()
    ser.write(b"Stop!") 
    time.sleep(0.5)
    ser.write(b"Stop!") 
    time.sleep(0.5)
    ser.write(b"Stop!") 
    
    
    #line = ser.readline().decode('utf-8').strip()
    #print(line)
    ser.close()
    
    
    
    
    
    
    
    #INSERT HERE THE CODE FOR PEST DETECTION
    print('Robot Running Pest Detection on Plot Number ', plot)
    cam = cv2.VideoCapture(0)
    result, image = cam.read()
    detection = detect(image)
    print('Done Running Pest Detection')
    if detection[1] > 0: #SENDS DATA TO FIREBASE IS PEST IS DETECTED
        print('Pest Detected')
        print('Sending to image to firebase')
        SendData(detection[0][:,:,::-1],'Pest_'+ datetime.now().strftime('%m-%d-%Y')+ '_'  + datetime.now().strftime('%H_%M_%S') + '.jpg', datetime.now().strftime('%H:%M'), plot, detection[4])
    print('Adding data to dataframe')
    df = add_df(plot, detection[1], detection[2], detection[3]) #Plot, Pest Count, Time, Date
    df = df.apply(lambda x: pd.Series(x.dropna().values)) 
    df = df.reindex(sorted(df.columns, reverse=False), axis='columns')
    print('Finished adding data to dataframe')
    
    
    
    #IF CURRENT TIME - START TIME = 3 Hours
    #SEND THE STATISTICAL DATA TO THE FIREBASE
    #END
    
    
    
    #time.sleep(5)
    
    




#MAIN ALGO END

