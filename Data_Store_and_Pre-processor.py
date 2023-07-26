import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from streamlit_option_menu import option_menu
import plotly.express as px
from IPython.display import  HTML
import os
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
import wikipedia
import streamlit.components.v1 as com
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from streamlit_option_menu import option_menu
from PIL import Image
import sqlite3

st.markdown('''
<style>
<style>
div.css-14xtw13.e8zbici0{
  display:none;
  visibility:none;
}
.css-1dp5vir.e8zbici1{
  background-image:linear-gradient(90deg, rgb(255, 255, 255), rgb(255, 255, 255));
}
.css-1q1n0ol.egzxvld0{
  display:none;
  visibility:none;
}
   div.css-6qob1r.e1fqkh3o3{
   background-color:#cef8f7;
   }
   
   .css-6kekos{
   background-color:#cef8f7;
   color:black;
   font-weight:600;
   }
   
   .menu.nav-item.nav-link{
    background-color:#cef8f7;
   color:white;
   font-weight:600;
   }
   .css-qri22k.egzxvld0{
   display:none;
   }
</style>
''',unsafe_allow_html = True)

selected = option_menu(
    menu_title="Main Menu",
    options=['DL Page','ML Page','CRUD', 'IES', 'EDA Page'],
    icons=['house', 'file-bar-graph-fill', 'steam'],
    menu_icon='cast',
    default_index=0,
    orientation="horizontal"
)
if selected == "ML Page":
    import os
    from pathlib import Path
    import pickle
    import numpy as np
    import wikipedia
    import pandas as pd
    import streamlit as st
    from sklearn.svm import SVC
    import matplotlib.pyplot as plt
    import streamlit.components.v1 as com
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, \
        confusion_matrix

    st.markdown('''
       <style>
       .css-6kekos.edgvbvh9{
       background-color:red;
       color:white;
       font-family:calibri;
       font-weight:600;
       }
       </style>
       ''', unsafe_allow_html=True)
    st.sidebar.header("Diabetes Prediction Web App")
    st.sidebar.image('data:image/jpeg;base64,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')
    st.subheader("Medical patient Record")
    col1, col2,col3,col4 = st.columns(4)
    with col1:
        input1 = st.text_input('Age', key=1)
        input2 = st.text_input('Gender', key=2)
        input3 = st.text_input('Polyuria', key=3)
        input4 = st.text_input('Polydipsia', key=4)
    with col2:
        input5 = st.text_input('sudden weight loss', key=5)
        input6 = st.text_input('weakness', key=6)
        input7 = st.text_input('Polyphagia', key=7)
        input8 = st.text_input('Genital thrush', key=8)
    with col3:
        input9 = st.text_input('visual blurring', key=9)
        input10 = st.text_input('Itching', key=10)
        input11 = st.text_input('Irritability', key=11)
        input12 = st.text_input('delayed healing', key=12)
    with col4:
        input13 = st.text_input('partial paresis', key=13)
        input14 = st.text_input('muscle stiffness', key=14)
        input15 = st.text_input('Alopecia', key=15)
        input16 = st.text_input('Obesity', key=16)



    if st.button("Predict Medical Status"):
        data1 = [input1]
        data2 = [input2]
        data3 = [input3]
        data4 = [input4]
        data5 = [input5]
        data6 = [input6]
        data7 = [input7]

        data8 = [input8]
        data9 = [input9]
        data10 = [input10]
        data11 = [input11]
        data12 = [input12]
        data13 = [input13]
        data14 = [input14]
        data15 = [input15]
        data16 = [input16]

        conn = sqlite3.connect('MedicalPatientRecord.db')
        cursor = conn.cursor()
        conn.execute("""
                INSERT INTO patient_record_table (age,gender,polyuria,polydipsia,sudden_weight_loss, weakness,
        polyphagia, genital_thrush, visual_blurring, itching,
        irritability,delayed_healing, partial_paresis, muscle_stiffnes,
        alopecia,obesity) VALUES(?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,?,?)""", (input1, input2, input3, input4, input5, input6, input7, input8, input9,
                                              input10, input11, input12, input13, input14, input15, input16))
        conn.commit()
        cursor.close()
        conn.close()

        da1 = pd.Series(data1)
        da2 = pd.Series(data2)
        da3 = pd.Series(data3)
        da4 = pd.Series(data4)
        da5 = pd.Series(data5)
        da6 = pd.Series(data6)
        da7 = pd.Series(data7)

        da8 = pd.Series(data8)
        da9 = pd.Series(data9)
        da10 = pd.Series(data10)
        da11 = pd.Series(data11)
        da12 = pd.Series(data12)
        da13 = pd.Series(data13)
        da14 = pd.Series(data14)
        da15 = pd.Series(data15)
        da16 = pd.Series(data16)

        frame = {"Age": da1, "Gender": da2, "Polyuria": da3,
                 "Polydipsia": da4, "sudden weight loss": da5, "weakness": da6,
                 "Polyphagia": da7, "Genital thrush": da8, "visual blurring": da9, "Itching": da10,
                 "Irritability": da11, "delayed healing": da12, "partial paresis": da13,
                 "muscle stiffness": da14, "Alopecia": da15, "Obesity": da16
                 }
        dataframe = pd.DataFrame(frame)
        print(dataframe)
        st.write('---')
        st.write(dataframe)
        file_path1 = Path(__file__).parent / "model2.pkl"
        with file_path1.open("rb") as f1:
            testmodel = pickle.load(f1)

        # labelInit = LabelEncoder()
        # objectList = dataframe.select_dtypes(include="object").columns
        # for feature in objectList:
        #     dataframe[feature] = labelInit.fit_transform(dataframe[feature].astype(str))
        # print(dataframe)

        dataframe = dataframe.replace(['Male', 'Female', 'Yes', 'No'], (1, 0, 1, 0))
        print(dataframe)
        result = testmodel.predict(dataframe)
        predicted_data = np.array(result)
        print(predicted_data)
        dataframe2 = pd.DataFrame(data=predicted_data, columns=['PredictedData'])
        dff5 = int(dataframe2['PredictedData'].to_numpy())

        if dff5 == 1:

            st.error(f"The prediction result for your diabetes test is Positive. It means you do have diabetes. The main reason"
                       f" for getting these result is that you have a very bad medical status on the condition that is listed below."
                       f" N.B As a recommendattion, You have to work on these status by tracking the 5 condition listed below, these is because they are the 5 most important features that could determine the result of the prediction.")
            info2 = wikipedia.summary("Symptoms of diabetes", 3)
            sorted_idx = testmodel.best_estimator_.feature_importances_.argsort()[-5:]
            series1 = pd.Series(dataframe.columns[sorted_idx])
            series2 = pd.Series((testmodel.best_estimator_.feature_importances_[sorted_idx]) * 100)
            frame1 = pd.DataFrame({'Top 5 Best sympthoms which determines the result of the prediction': series1,
                                   'Result of the feature importance out of 100%': series2})
            coll5,coll6 = st.columns(2)
            with coll5:
                st.write(frame1.sort_values(by=['Result of the feature importance out of 100%'], axis=0,
                                            ascending=False))
            with coll6:
                fig = plt.figure(figsize=(20, 10))
                sorted_idx = testmodel.best_estimator_.feature_importances_.argsort()[-5:]
                plt.barh(dataframe.columns[sorted_idx], testmodel.best_estimator_.feature_importances_[sorted_idx])
                st.pyplot(fig)
            print(info2)
            
            st.write('---')
            coll3,coll4 = st.columns(2)
            with coll3:
                st.subheader("Disease Description")
                st.write(info2)
            with coll4:
                query1 = "Diabetes"
                wp_page = wikipedia.page(query1)
                list_img_urls = wp_page.images

                tab1, tab2, tab3 = st.tabs(['Related image 1','Related image 2','Related images 3'])
                with tab1:
                    st.image(list_img_urls[7])
                with tab2:
                    st.image(list_img_urls[8])
                with tab3:
                        st.image(list_img_urls[4])


                
        elif dff5 == 0:
            st.success(f"The prediction result for your diabetes test is Negative. It means you do not have diabetes. The main reason"
                       f" for getting these result is that you have a good medical status on the condition that is listed below."
                       f" N.B As a recommendattion, You have to maintain these status by tracking the 5 condition listed below, these is because they are the 5 most important features that could determine the result of the prediction.")
            sorted_idx = testmodel.best_estimator_.feature_importances_.argsort()[-5:]
            series1 = pd.Series(dataframe.columns[sorted_idx])
            series2 = pd.Series((testmodel.best_estimator_.feature_importances_[sorted_idx]) * 100)
            frame1 = pd.DataFrame({'Top 5 Best sympthoms which determines the result of the prediction': series1,
                                   'Result of the feature importance out of 100%': series2})
            coll5,coll6 = st.columns(2)
            with coll5:
                st.write(frame1.sort_values(by=['Result of the feature importance out of 100%'], axis=0,
                                            ascending=False))
            with coll6:
                fig = plt.figure(figsize=(20, 10))
                sorted_idx = testmodel.best_estimator_.feature_importances_.argsort()[-5:]
                plt.barh(dataframe.columns[sorted_idx], testmodel.best_estimator_.feature_importances_[sorted_idx])
                st.pyplot(fig)

if selected == "DL Page":
    st.sidebar.header('Dog Breed Classifier web app')
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=['Browse File', 'Take Camera'],
            icons=['steam', 'yin-yang'],
            menu_icon='cast',
            default_index=0,
            orientation="vertical"
        )

    if selected == "Browse File":

        file_uploader = st.file_uploader("Upload an image file", '.jpg')

        if file_uploader is not None:

            loaded_model = tf.keras.models.load_model('C:/Users/GL/PycharmProjects/VirtualAssistant/DBC.h5')
            img1 = image.load_img(file_uploader, target_size=(224, 224))
            img2 = image.load_img(file_uploader)
            conn = sqlite3.connect('image.db')
            cursor = conn.cursor()
            name = "Uploaded Image From Folder"
            conn.execute("""
            INSERT INTO my_table (name,data) VALUES(?,?)""", (name, file_uploader.name))
            conn.commit()
            cursor.close()
            conn.close()

            test_image = image.img_to_array(img1)
            test_image = np.expand_dims(test_image, axis=0)
            images = np.vstack([test_image])
            val = loaded_model.predict(images)

            st.write('---')
            col1, col2 = st.columns(2)
            with col1:
                st.image(img2)
                st.write('---')
                array = np.array(val)
                if array[0][0] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 30 to 32 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][1] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][2] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][3] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][4] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][5] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][6] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][7] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][8] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif array[0][9] == 1:
                    with st.expander('Size'):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
            with col2:
                array = np.array(val)

                if array[0][0] == 1:
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;    
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Scottish Deerhound</h3>
                        <p class="Description">
                            The Scottish Deerhound, or simply the Deerhound, is a large breed of sighthound,
                             once bred to hunt the red deer by coursing. In outward appearance, the Scottish Deerhound 
                             is similar to the Greyhound, but larger and more heavily boned with a rough-coat.
                        </p>

                        <h4 class="tx1">Life Expectancy: 8-11 years.</h4>
                        <h4 class="tx2">Temperament: Dignified, Docile, Friendly, Gentle</h4>
                        <h4 class="tx3">Hypoallergenic: No</h4>
                        <h4 class="tx4">Origin: Scotland</h4>
                        <h4 class="tx5">Weight: Male: 39–50 kg, Female: 34–43 kg</h4>
                        <h4 class="tx6">Colors: Brindle, Fawn, Red Fawn, Blue, Grey, Yellow</h4>
                        <h4 class="tx7">The Kennel Club: standard</h4>
                    </div>
                    """, width=450, height=400)
                    st.write('---')
                    query1 = "Scottish Deerhound"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[5])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])


                elif array[0][1] == 1:
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;    
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Maltese Dog</h3>
                        <p class="Description">
                        Maltese dog refers both to an ancient variety of dwarf canine from Italy and generally associated also with the island of Malta, and to a modern breed of dog in the toy group. 
                        The contemporary variety is genetically related to the Bichon, Bolognese, and Havanese breeds.
                        </p>

                        <h4 class="tx1">Life Expectancy: 12 – 15 years</h4>
                        <h4 class="tx2">Temperament: Playful, Docile, Easygoing, Intelligent, Lively,</h4>
                        <h4 class="tx3">Hypoallergenic: Yes</h4>
                        <h4 class="tx4">Origin: Italy, Mediterranean Basin</h4>
                        <h4 class="tx5">Weight: Male: 21–25 cm, Female: 20–23 cm</h4>
                        <h4 class="tx6">Colors: Male: 3–4 kg, Female: 3–4 kg</h4>
                        <h4 class="tx7">The Kennel Club: White</h4>
                    </div>
                    """, width=450, height=400)
                    st.write('---')
                    query1 = "Maltese dog"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[1])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])

                elif array[0][2] == 1:
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;    
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Afghan Hound</h3>
                        <p class="Description">
                        The Afghan Hound is a hound that is distinguished by its thick, fine, silky coat and its tail with a ring curl at the end. The breed is selectively bred for its unique features in the cold mountains of Afghanistan. Its local name is Tāžī Spay or Sag-e Tāzī
                        </p>

                        <h4 class="tx1">Life Expectancy: 12 – 15 years</h4>
                        <h4 class="tx2">Temperament: Dignified, Aloof, Independent, Clownish, Happy,</h4>
                        <h4 class="tx3">Hypoallergenic: Yes</h4>
                        <h4 class="tx4">Origin: Afghanistan</h4>
                        <h4 class="tx5">Height: Male: 68–74 cm, Female: 60–69 cm</h4>
                        <h4 class="tx6">Colors: Black, Cream, Red</h4>
                        <h4 class="tx7">The Kennel Club: White</h4>
                    </div>
                    """, width=450, height=400)
                    st.write('---')
                    query1 = "Afghan hound dog"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[1])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])
                elif array[0][3] == 1:
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;    
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Entlebucher</h3>
                        <p class="Description">
                        The Entlebucher Sennenhund or Entlebucher Mountain Dog is a medium-sized herding dog, it is the smallest of the four regional breeds that constitute the Sennenhund dog type. The name Sennenhund refers to people called Senn, herders in the Swiss Alps. Entlebuch is a region in the canton of Lucerne in Switzerland.
                        </p>

                        <h4 class="tx1">Life Expectancy: 12 – 15 years</h4>
                        <h4 class="tx2">Temperament: Dignified, Aloof, Independent, Clownish, Happy,</h4>
                        <h4 class="tx3">Hypoallergenic: Yes</h4>
                        <h4 class="tx4">Origin: Afghanistan</h4>
                        <h4 class="tx5">Height: Male: 68–74 cm, Female: 60–69 cm</h4>
                        <h4 class="tx6">Colors: Black, Cream, Red</h4>
                        <h4 class="tx7">The Kennel Club: White</h4>
                    </div>
                    """, width=450, height=420)
                    st.write('---')
                    query1 = "Entlebucher dog"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[1])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])

                elif array[0][4] == 1:
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;    
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Bernese Mountain Dog</h3>
                        <p class="Description">
                        The Bernese Mountain Dog is a large dog breed, one of the four breeds of Sennenhund-type dogs from Bern, Switzerland and the Swiss Alps. These dogs have roots in the Roman mastiffs. The name Sennenhund is derived from the German Senne and Hund, as they accompanied the alpine herders and dairymen called Senn.
                        </p>

                        <h4 class="tx1">Life expectancy: 6 – 8 years</h4>
                        <h4 class="tx2">Temperament: Intelligent, Affectionate, Loyal, Faithful</h4>
                        <h4 class="tx3">Hypoallergenic: Yes</h4>
                        <h4 class="tx4">Origin: Afghanistan</h4>
                        <h4 class="tx5">Height: Male: 64–70 cm, Female: 58–66 cm</h4>
                        <h4 class="tx6">Male: 38–50 kg, Female: 36–48 kg</h4>
                        <h4 class="tx7">Switzerland</h4>
                    </div>
                    """, width=450, height=400)
                    st.write('---')
                    query1 = "Bernese mountain dog breed"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[1])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])
                elif array[0][5] == 1:
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;    
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Shih Tzu</h3>
                        <p class="Description">
                        The Shih Tzu is a toy dog breed originating from Tibet and was bred from the Pekingese and the Lhasa Apso. Shih Tzus are known for their short snouts and large round eyes, as well as their long coat, floppy ears, and short and stout posture.
                        </p>

                        <h4 class="tx1">Life expectancy: 10 – 16 years</h4>
                        <h4 class="tx2">Temperament: Playful, Clever, Friendly, Intelligent, Lively, Outgoing,</h4>
                        <h4 class="tx3">Hypoallergenic: Yes</h4>
                        <h4 class="tx4">Origin: China, Tibet</h4>
                        <h4 class="tx5">Height: 20 – 28 cm (Female, Adult, At the withers), 20 – 28 cm (Male, Adult, At the withers)</h4>
                        <h4 class="tx6">Mass: 4 – 7.2 kg (Female, Adult), 4 – 7.2 kg (Male, Adult)</h4>
                    </div>
                    """, width=450, height=400)
                    st.write('---')
                    query1 = "Shih Tzu dog breed"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[1])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])
                    st.markdown('shih-tzu')

                elif array[0][6] == 1:
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;
                            padding-bottom:20px;    
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Pyrenean Mountain Dog</h3>
                        <p class="Description">
                        The Pyrenean Mountain Dog is a breed of livestock guardian dog from France, where it is commonly called the Patou. The breed comes from the French side of the Pyrenees Mountains that separate France and Spain.
                        </p>

                        <h4 class="tx1">Life Expectancy: 12 – 15 years</h4>
                        <h4 class="tx2">Temperament: Dignified, Aloof, Independent, Clownish, Happy,</h4>
                        <h4 class="tx3">Hypoallergenic: Yes</h4>
                        <h4 class="tx4">Origin: Afghanistan</h4>
                        <h4 class="tx5">Height: Male: 68–74 cm, Female: 60–69 cm</h4>
                    </div>
                    """, width=450, height=380)
                    st.write('---')
                    query1 = "Pyrenean Mountain Dog"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[1])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])
                elif array[0][7] == 1:
                    com.html("""
                     <style>
                         .info_div{
                             box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                             border-radius:8px;
                             background-color:#fc4949;
                             color:white;
                             margin-top:-20px;    
                         }
                         .Dog_name{
                         font-size:22px;
                         font-family:calibri;
                         padding-top: 20px;
                         padding-left: 20px;
                         font-weight:600;
                         }
                         .Description{
                         font-family:calibri;
                         font-size:17px;
                         text-align:justify;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-5px;
                         }

                         .text1{
                         display:flex;
                         flex-direction:horizontal;
                         }
                         .tx1{
                         font-family: calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top: -5px;
                         }
                         .tx2{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx3{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx4{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }

                         .tx5{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx6{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }

                         .tx7{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         padding-bottom:20px;
                         }

                     </style>

                     <div class="info_div">
                         <h3 class="Dog_name">Pomeranian</h3>
                         <p class="Description">
                           The Pomeranian is a breed of dog of the Spitz type that is named for the Pomerania region in north-west Poland and north-east Germany in Central Europe. Classed as a toy dog breed because of its small size, the Pomeranian is descended from larger Spitz-type dogs, specifically the German Spitz.
                         </p>

                         <h4 class="tx1">Life Expectancy: 12 – 15 years</h4>
                         <h4 class="tx2">Temperament: Playful, Docile, Easygoing, Intelligent, Lively,</h4>
                         <h4 class="tx3">Hypoallergenic: Yes</h4>
                         <h4 class="tx4">Origin: Italy, Mediterranean Basin</h4>
                         <h4 class="tx5">Weight: Male: 21–25 cm, Female: 20–23 cm</h4>
                         <h4 class="tx6">Colors: Male: 3–4 kg, Female: 3–4 kg</h4>
                         <h4 class="tx7">The Kennel Club: White</h4>
                     </div>
                     """, width=450, height=400)
                    st.write('---')
                    query1 = "Pomeranian dog breed"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[3])
                    with tab2:
                        st.image(list_img_urls[4])
                    with tab3:
                        st.image(list_img_urls[5])

                elif array[0][8] == 1:
                    com.html("""
                     <style>
                         .info_div{
                             box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                             border-radius:8px;
                             background-color:#fc4949;
                             color:white;
                             margin-top:-20px;    
                         }
                         .Dog_name{
                         font-size:22px;
                         font-family:calibri;
                         padding-top: 20px;
                         padding-left: 20px;
                         font-weight:600;
                         }
                         .Description{
                         font-family:calibri;
                         font-size:17px;
                         text-align:justify;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-5px;
                         }

                         .text1{
                         display:flex;
                         flex-direction:horizontal;
                         }
                         .tx1{
                         font-family: calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top: -5px;
                         }
                         .tx2{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx3{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx4{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }

                         .tx5{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx6{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }

                         .tx7{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         padding-bottom:20px;
                         }

                     </style>

                     <div class="info_div">
                         <h3 class="Dog_name">Basenji</h3>
                         <p class="Description">
                            The Basenji is a breed of hunting dog. It was bred from stock that originated in central Africa. The Fédération Cynologique Internationale places the breed in the Spitz and primitive types. 
                            The Basenji produces an unusual yodel-like sound, due to its unusually shaped larynx.
                         </p>

                         <h4 class="tx1">Life Expectancy: 12 – 15 years</h4>
                         <h4 class="tx2">Temperament: Playful, Docile, Easygoing, Intelligent, Lively,</h4>
                         <h4 class="tx3">Hypoallergenic: Yes</h4>
                         <h4 class="tx4">Origin: Italy, Mediterranean Basin</h4>
                         <h4 class="tx5">Weight: Male: 21–25 cm, Female: 20–23 cm</h4>
                         <h4 class="tx6">Colors: Male: 3–4 kg, Female: 3–4 kg</h4>
                         <h4 class="tx7">The Kennel Club: White</h4>
                     </div>
                     """, width=450, height=400)
                    st.write('---')
                    query1 = "Basenji dog breed"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[3])
                    with tab2:
                        st.image(list_img_urls[4])
                    with tab3:
                        st.image(list_img_urls[5])
                elif array[0][9] == 1:
                    com.html("""
                     <style>
                         .info_div{
                             box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                             border-radius:8px;
                             background-color:#fc4949;
                             color:white;
                             margin-top:-20px;    
                         }
                         .Dog_name{
                         font-size:22px;
                         font-family:calibri;
                         padding-top: 20px;
                         padding-left: 20px;
                         font-weight:600;
                         }
                         .Description{
                         font-family:calibri;
                         font-size:17px;
                         text-align:justify;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-5px;
                         }

                         .text1{
                         display:flex;
                         flex-direction:horizontal;
                         }
                         .tx1{
                         font-family: calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top: -5px;
                         }
                         .tx2{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx3{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx4{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }

                         .tx5{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }
                         .tx6{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         }

                         .tx7{
                         font-family:calibri;
                         margin-left:20px;
                         margin-right:20px;
                         margin-top:-15px;
                         padding-bottom:20px;
                         }

                     </style>

                     <div class="info_div">
                         <h3 class="Dog_name">Samoyed</h3>
                         <p class="Description">
                            The Samoyed is a breed of medium-sized herding dogs with thick, white, double-layer coats. They are a spitz-type dog which takes its name from the Samoyedic peoples of Siberia. Descending from the Nenets Herding Laika, 
                            they are a domesticated animal that assists in herding, hunting, protection and sled-pulling.
                         </p>

                         <h4 class="tx1">Life Expectancy: 12 – 15 years</h4>
                         <h4 class="tx2">Temperament: Playful, Docile, Easygoing, Intelligent, Lively,</h4>
                         <h4 class="tx3">Hypoallergenic: Yes</h4>
                         <h4 class="tx4">Origin: Italy, Mediterranean Basin</h4>
                         <h4 class="tx5">Weight: Male: 21–25 cm, Female: 20–23 cm</h4>
                         <h4 class="tx6">Colors: Male: 3–4 kg, Female: 3–4 kg</h4>
                         <h4 class="tx7">The Kennel Club: White</h4>
                     </div>
                     """, width=450, height=400)
                    st.write('---')
                    query1 = "Samoyed dog breed"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[3])
                    with tab2:
                        st.image(list_img_urls[4])
                    with tab3:
                        st.image(list_img_urls[5])
            st.write('---')
    elif selected == "Take Camera":
        camera_input = st.camera_input("Take a photo")
        if camera_input is not None:
            loaded_model = tf.keras.models.load_model('C:/Users/GL/PycharmProjects/VirtualAssistant/DBC.h5')

            img1 = image.load_img(camera_input, target_size=(224, 224))
            img2 = image.load_img(camera_input)

            conn = sqlite3.connect('Camera_taken_photos.db')
            cursor = conn.cursor()
            name = "Selfie Camera Photos"
            conn.execute("""
            INSERT INTO my_table1 (name,data) VALUES(?,?)""", (name, camera_input.name))
            conn.commit()
            cursor.close()
            conn.close()
            # test_image = image.img_to_array(img1)
            # test_image = np.expand_dims(test_image, axis=0)
            # images = np.vstack([test_image])
            # val = loaded_model.predict(images)
            mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()
            img = image.load_img(camera_input, target_size=(224, 224))
            resized_img = image.img_to_array(img)
            final_img = np.expand_dims(resized_img, axis=0)
            final_img = tf.keras.applications.mobilenet_v2.preprocess_input(final_img)
            prediction = mobile.predict(final_img)
            result = imagenet_utils.decode_predictions(prediction)
            numpy_result = np.array(result)
            label = numpy_result[0][0][1]
            print(label)

            coll1, coll2 = st.columns(2)
            with coll1:
                st.image(img2)

                if label== "Scottish_deerhound":
                    with st.expander("Size"):
                        st.markdown('''#### 15 to 20 inches tall ''')
                        st.markdown(
                            "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                    with st.expander('Shadding'):
                        st.markdown(
                            "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                    with st.expander("Purpose"):
                        st.markdown(
                            "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                    with st.expander("Health"):
                        st.markdown(
                            "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
                elif label=="Afghan_hound":
                    if label == "Afghan_hound":
                        with st.expander("Size"):
                            st.markdown('''#### 15 to 20 inches tall ''')
                            st.markdown(
                                "Depending on gender, the Irish wolfhound ranges in height from 30 to 36 inches tall, while the Scottish deerhound only measures 30 to 32 inches tall. While both of these dogs look extraordinarily similar to one another, there are some key differences in their sizes.")
                        with st.expander('Shadding'):
                            st.markdown(
                                "The deerhound coat does not shed, but it needs weekly brushing or combing, and the dead hairs need to be pulled out by hand twice a year. The beard tends to drip water after drinking, and it should be washed frequently.")
                        with st.expander("Purpose"):
                            st.markdown(
                                "The crisply coated Scottish Deerhound, 'Royal Dog of Scotland,' is a majestically large coursing hound struck from the ancient Greyhound template. Among the tallest of dog breeds, the Deerhound was bred to stalk the giant wild red deer.")
                        with st.expander("Health"):
                            st.markdown(
                                "The Scottish Deerhound breed, which has an average lifespan of 7 to 9 years, is susceptible to major health issues such as cardiomyopathy, gastric torsion, and osteosarcoma. Hypothyroidism, neck pain, atopy, and cystinuria may also plague this dog.Oct 6, 2009")
            with coll2:

                if label == "Scottish_deerhound":
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Scottish Deerhound</h3>
                        <p class="Description">
                            The Scottish Deerhound, or simply the Deerhound, is a large breed of sighthound,
                             once bred to hunt the red deer by coursing. In outward appearance, the Scottish Deerhound
                             is similar to the Greyhound, but larger and more heavily boned with a rough-coat.
                        </p>

                        <h4 class="tx1">Life Expectancy: 8-11 years.</h4>
                        <h4 class="tx2">Temperament: Dignified, Docile, Friendly, Gentle</h4>
                        <h4 class="tx3">Hypoallergenic: No</h4>
                        <h4 class="tx4">Origin: Scotland</h4>
                        <h4 class="tx5">Weight: Male: 39–50 kg, Female: 34–43 kg</h4>
                        <h4 class="tx6">Colors: Brindle, Fawn, Red Fawn, Blue, Grey, Yellow</h4>
                        <h4 class="tx7">The Kennel Club: standard</h4>
                    </div>
                    """, width=450, height=400)
                    st.write('---')
                    query1 = "Scottish Deerhound"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[5])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])
                elif label == "Afghan_hound":
                    com.html("""
                    <style>
                        .info_div{
                            box-shadow:2px 2px 10px rgba(0,0,0,0.25);
                            border-radius:8px;
                            background-color:#fc4949;
                            color:white;
                            margin-top:-20px;
                        }
                        .Dog_name{
                        font-size:22px;
                        font-family:calibri;
                        padding-top: 20px;
                        padding-left: 20px;
                        font-weight:600;
                        }
                        .Description{
                        font-family:calibri;
                        font-size:17px;
                        text-align:justify;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-5px;
                        }

                        .text1{
                        display:flex;
                        flex-direction:horizontal;
                        }
                        .tx1{
                        font-family: calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top: -5px;
                        }
                        .tx2{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx3{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx4{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx5{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }
                        .tx6{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        }

                        .tx7{
                        font-family:calibri;
                        margin-left:20px;
                        margin-right:20px;
                        margin-top:-15px;
                        padding-bottom:20px;
                        }

                    </style>

                    <div class="info_div">
                        <h3 class="Dog_name">Afghan Hound</h3>
                        <p class="Description">
                        The Afghan Hound is a hound that is distinguished by its thick, fine, silky coat and its tail with a ring curl at the end. The breed is selectively bred for its unique features in the cold mountains of Afghanistan. Its local name is Tāžī Spay or Sag-e Tāzī
                        </p>

                        <h4 class="tx1">Life Expectancy: 12 – 15 years</h4>
                        <h4 class="tx2">Temperament: Dignified, Aloof, Independent, Clownish, Happy,</h4>
                        <h4 class="tx3">Hypoallergenic: Yes</h4>
                        <h4 class="tx4">Origin: Afghanistan</h4>
                        <h4 class="tx5">Height: Male: 68–74 cm, Female: 60–69 cm</h4>
                        <h4 class="tx6">Colors: Black, Cream, Red</h4>
                        <h4 class="tx7">The Kennel Club: White</h4>
                    </div>
                    """, width=450, height=400)
                    st.write('---')
                    query1 = "Afghan hound dog"
                    wp_page = wikipedia.page(query1)
                    list_img_urls = wp_page.images

                    tab1, tab2, tab3 = st.tabs(['Related Dog image 1', 'Related Dog image 2', 'Related Dog images 3'])
                    with tab1:
                        st.image(list_img_urls[1])
                    with tab2:
                        st.image(list_img_urls[2])
                    with tab3:
                        st.image(list_img_urls[3])

if selected == "IES":
    st.sidebar.image("https://lh3.googleusercontent.com/-cE0zxnMLK8w/V08TjpafjII/AAAAAAAAARQ/HVrjhBvqJXA/s640/1464798640719.png")
    upload_image_file = st.file_uploader("Upload image",'jpg',key=1555)
    if upload_image_file is not None:
        image = np.array(Image.open(upload_image_file, "r"))
        image2 = np.copy(image)

    else:
        st.info("Waiting for the image to be uploaded")

    edit_options = st.sidebar.selectbox("Edit Options:", ["Select Option","Basic Image Info", "Rotate", "Flip", "Crop"], 0)

    if edit_options != "Select Option":
        if edit_options == "Basic Image Info":
            coll1, coll2 = st.columns(2)
            with coll1:
                st.image(upload_image_file)
            with coll2:

                st.markdown("""##### Basic Information about Uploaded Image """)
                st.markdown(f"File Name:    {upload_image_file.name}")
                st.markdown(f"File type:    {upload_image_file.type}")
                st.markdown(f"File size:    {upload_image_file.size} kb")
                st.markdown(f"File type:    {image.dtype}")

                width = int(image.shape[0])
                height = int(image.shape[1])
                color_val = int(image.shape[2])
                color_type = " Refers RGB image"
                st.markdown(f"Dimension of the image:{width} * {height}")
                if color_val == 3:
                    st.markdown(f"Color val: {color_val} ({color_type})")
        if edit_options == "Rotate":
            coll1, coll2 = st.columns(2)
            with coll1:

                st.write("Uploaded image")
                st.image(upload_image_file)

            with coll2:
                choice = st.selectbox("Choose rotation:", ['Select', '90deg', '180deg', '270deg'], 0)
                if choice != 'Select':

                    if choice == "90deg":
                        image2 = np.rot90(image, k=1)
                        st.image(image2)
                        st.download_button("Download an image",image2,mime="image/jpeg")
                    elif choice == "180deg":
                        image3 = np.rot90(image, k=2)
                        st.image(image3)
                    elif choice == "270deg":
                        image4 = np.rot90(image, k=3)
                        st.image(image4)
        elif edit_options == "Flip":
            coll1, coll2 = st.columns(2)
            with coll1:
                st.image(upload_image_file)
            with coll2:
                st.write("Fliped Image")
                st.image(np.fliplr(image))
        elif edit_options == "Crop":
            coll1, coll2 = st.columns(2)
            with coll1:
                width_value_right = st.number_input("Enter right width value to crop an image", 32, 255)
                width_value_left = st.number_input("Enter left value to crop an image", 32, 255)
                height_value_right = st.number_input("Enter the right height value to crop an image", 32, 255)
                height_value_left = st.number_input("Enter the left height value to crop an image", 32, 255)

            with coll2:
                if width_value_right and width_value_left and height_value_left and height_value_right:
                    img = np.array(image)
                    img0 = img[height_value_right:-height_value_left, width_value_right:-width_value_left, :]
                    st.image(img0)

    filter_options = st.sidebar.selectbox("Choose Filter:",
                                          ["Select option", "Filters", "Color Controller", "Blend Images","Noise Removal"], 0)
    if filter_options != "Select":
        if filter_options == "Filters":
            choice2 = st.selectbox("Filter type",
                                   ("filter_type1", "filter_type2", "filter_type3", "filter_type4", "filter_type5","filter_type6"))
            if choice2 == "filter_type1":
                negative_image = 255 - image
                st.image(negative_image)
            if choice2 == "filter_type2":
                for i in range(len(image)):
                    for j in range(len(image[i])):
                        blue = image[i, j, 0]
                        green = image[i, j, 1]
                        red = image[i, j, 2]

                        grey_scale = (red / 3 + green / 3 + blue / 3)
                        image[i, j] = grey_scale

                grey_scale_img = Image.fromarray(image)
                image2 = np.array(grey_scale_img)
                st.image(image2)
            if choice2 == "filter_type3":
                kernel = np.ones((3, 3), np.float32) / 9
                img = cv2.filter2D(image, -1, kernel)
                st.image(img)
            if choice2 == "filter_type4":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Emboss_Kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
                Emboss_Effect_Img = cv2.filter2D(src=image, kernel=Emboss_Kernel, ddepth=-1)
                st.image(Emboss_Effect_Img)
            if choice2 == "filter_type5":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Sharpen_Kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                Sharpen_Effect_Img = cv2.filter2D(src=image, kernel=Sharpen_Kernel, ddepth=-1)
                st.image(Sharpen_Effect_Img)
            if choice2 == "filter_type6":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                Sepia_Kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
                Sepia_Effect_Img = cv2.filter2D(src=image, kernel=Sepia_Kernel, ddepth=-1)
                st.image(Sepia_Effect_Img)



        if filter_options == "Color Controller":
            number = st.number_input("Enter numerical value", 1, 255)
            if number:
                image2 = (image // number) * number
                st.image(image2)
        if filter_options == "Blend Images":
            uploaded_file2 = st.file_uploader('Upload image file', "jpg",key=1333)
            if uploaded_file2 is not None:
                added_image = np.array(Image.open(uploaded_file2, "r"))
                img1 = np.array(image)
                res_img = cv2.resize(added_image, img1.shape[1::-1])
                New_Image = (img1 * 0.8 + res_img * 0.2).astype(np.uint8)
                st.image(New_Image)
            else:
                st.info("Waiting for the image to be uploaded")
        if filter_options == "Noise Removal":
            noiseless_image_colored = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
            st.image(noiseless_image_colored)

elif selected == "CRUD":
    crud_uploaded_file1 = st.file_uploader("Upload a csv file",'csv',key=1777)
    if crud_uploaded_file1 is not None:
        coldd1,coldd2 = st.columns(2)
        dataframe = pd.read_csv(crud_uploaded_file1)
        with coldd1:

            st.markdown('''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Uploaded Dataset</span>''',unsafe_allow_html=True)
            st.write(dataframe)
            st.write("---")
    else:
        st.info("Waiting for the file to be uploaded")

    info_choices = st.sidebar.selectbox("Choose basic info options:",["Select","Describe","isnull","Value_counts"],0)
    if info_choices != "Select":
        if info_choices == "Describe":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Basic statistical information</span>''',
                    unsafe_allow_html=True)

                st.write(dataframe.describe())

        elif info_choices == "isnull":
            with coldd2:
                st.markdown('''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Statistics for null values </span>''',unsafe_allow_html=True)

                st.write(dataframe.isna().sum())

        elif info_choices == "Value_counts":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Value count information</span>''',
                    unsafe_allow_html=True)

                text_input1 = st.text_input("Enter column name:")
                if text_input1:
                    st.write(dataframe[text_input1].value_counts())
    view_choices = st.sidebar.selectbox("Choose view options:",["Select","Get_Data_From_Database","View top5","View bottom5","View random5","view specific cell","view single row","View single column"],0)
    if view_choices != "Select":
        if view_choices == "Get_Data_From_Database":
            conn = sqlite3.connect('MedicalPatientRecord.db')
            loaded_data=pd.read_sql_query('SELECT * FROM patient_record_table',conn)
            database_dataframe = pd.DataFrame(loaded_data)
            st.write(loaded_data)
            st.download_button("Download CSV", database_dataframe.to_csv(), mime="text/csv")

        if view_choices == "View top5":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Top 5 data lists from the uploaded dataset</span>''',
                    unsafe_allow_html=True)

                st.write(dataframe.head(5))
        elif view_choices == "View bottom5":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Bottom 5 data lists from the dataframe</span>''',
                    unsafe_allow_html=True)

                st.write(dataframe.tail(5))
        elif view_choices == "View random5":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Random data lists from the dataframe</span>''',
                    unsafe_allow_html=True)

                value1 = st.number_input("Enter initial index value:")
                value2 = st.number_input("Enter final index value:")

                value1_int = int(value1)
                value2_int = int(value2)

                st.write(dataframe[value1_int:value2_int])
        elif view_choices == "view specific cell":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">View single cell data list from the dataframe</span>''',
                    unsafe_allow_html=True)

                index_value1 = st.number_input("Enter index value of the cell:")
                name_of_col = st.text_input("Enter the name of the column the data is located:")
                if name_of_col:
                    index_value1_int = int(index_value1)
                    st.write(f"The value within the given cell is : {dataframe.loc[index_value1_int,name_of_col]}")
        elif view_choices == "view single row":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Search datalist from the uploaded dataset using index Value</span>''',
                    unsafe_allow_html=True)

                index_value2 = st.number_input("Enter index value of the row:")

                index_value2_int = int(index_value2)
                list_of_index = dataframe.index.tolist()
                for i in range(len(list_of_index)):
                    if index_value2_int == int(list_of_index[i]):
                        st.write(dataframe[dataframe.index == index_value2_int])
        # elif view_choices == "View single column":
        #     column_name1 = st.text_input("Enter the name of the column:")
        #
        #     list_of_columns = dataframe.columns.tolist()
        #
        #     for i in range(len(list_of_columns)):
        #         if column_name1 == str(list_of_columns[i]):
        #             st.write(dataframe[column_name1].where[column_name1 == list_of_columns[i]])

    update_choices = st.sidebar.selectbox("Choose update options:",["Select","Fill null value using ffill","Fill null value using mean",
                                                                    "Fill null values using median","Fill null values using mode"],0)
    if update_choices != "Select":
        if update_choices == "Fill null value using ffill":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Fill null datas within the uploaded dataset</span>''',
                    unsafe_allow_html=True)

                # col_name = st.text_input("Enter the name of the column with null values:")
                # if col_name:
                dataframe.fillna(method = "ffill",inplace = True)
                st.write(dataframe)
                st.download_button("Download CSV",dataframe.to_csv(),mime = "text/csv")

        elif update_choices == "Fill null value using mean":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Fill null datas within the uploaded dataset</span>''',
                    unsafe_allow_html=True)
                dataframe = dataframe.fillna(dataframe.mean(numeric_only=True))
                st.write(dataframe)
                st.download_button("Download CSV",dataframe.to_csv(),mime="text/csv")

        elif update_choices == "Fill null values using median":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Fill null datas within the uploaded dataset</span>''',
                    unsafe_allow_html=True)
                dataframe = dataframe.fillna(dataframe.median(numeric_only=True))
                st.write(dataframe)
                st.download_button("Download CSV",dataframe.to_csv(),mime="text/csv")
        elif update_choices == "Fill null values using mode":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Fill null datas within the uploaded dataset</span>''',
                    unsafe_allow_html=True)
                col_name2 = st.text_input("Enter the colummn name with null values:")
                if col_name2:
                    dataframe[col_name2].fillna(dataframe[col_name2].mode()[0],inplace = True)
                    st.write(dataframe)
                    st.download_button("Download CSV",dataframe.to_csv(),mime="text/csv")

    remove_sign_choices = st.sidebar.selectbox("Choose sign removal options:",["Select","remove signs"],0)
    if remove_sign_choices != "Select":
        if remove_sign_choices == "remove signs":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Remove unseccary signs from the uploaded dataset</span>''',
                    unsafe_allow_html=True)
                col_name3 = st.text_input("Enter the column with unwanted signs:")
                if col_name3:
                    dataframe[col_name3] = dataframe[col_name3].str.replace(',', '').str.replace('$', '').astype(float)
                    dataframe = dataframe.fillna(dataframe.mean(numeric_only=True))
                    st.write(dataframe)
                    st.download_button("Download CSV", dataframe.to_csv(),mime="text/csv")
    create_dataset_choices = st.sidebar.selectbox("Choose option:",["Select","create"],0)
    if create_dataset_choices != "Select":
        if create_dataset_choices == "create":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Create a new dataset</span>''',
                    unsafe_allow_html=True)
                added_file = st.file_uploader("Upload a csv file",'csv')
                if added_file is not None:
                    added_csv = pd.read_csv(added_file)
                    created_data = pd.concat([dataframe,added_csv],axis=1,join = "outer")
                    st.write(created_data)
                    st.download_button("Download CSV",created_data.to_csv(),mime="text/csv")
                else:
                    st.info("Waiting for the file to be uploaded")
    delete_choices = st.sidebar.selectbox("Choose delete options:",["Select","Delete Null Values","Delete by name of column","Delete by Row"],0)
    if delete_choices != "Select":
        if delete_choices == "Delete Null Values":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Delete all null values from the dataset</span>''',
                    unsafe_allow_html=True)
                dataframe.dropna(axis = 1,inplace=True)
                st.success("Columns which holds null values are deleted succesfuly")
                st.write(dataframe)
                st.download_button("Download csv",dataframe.to_csv(),mime="text/csv")
        elif delete_choices == "Delete by name of column":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Delete a data from a dataset using column</span>''',
                    unsafe_allow_html=True)
                first_list = dataframe.columns.tolist()
                column_name = st.text_input("Enter the name of the column:")
                for i in range(len(first_list)):
                    if first_list[i] == str(column_name):
                        dataframe.drop(first_list[i],axis=1,inplace= True)
                        st.success("Column deleted Succesfuly")
                st.write(dataframe)
                st.download_button("Download CSV",dataframe.to_csv(),mime="text/csv")
        elif delete_choices  == "Delete by Row":
            with coldd2:
                st.markdown(
                    '''<span style = "color:#FF4B4B;font-family:Calibri;font-weight:600;font-size:20px;">Delete a data from the given dataset using row</span>''',
                    unsafe_allow_html=True)
                second_list = dataframe.index.tolist()
                index_value = st.number_input("Enter the index value of the row:")
                for i in range(len(second_list)):
                    if int(index_value) == int(second_list[i]):
                        dataframe.drop(second_list[i],axis=0,inplace=True)
                        st.success("Row deleted Successfuly")
                st.write(dataframe)
                st.download_button("Download CSV",dataframe.to_csv(),mime="text/csv")
elif selected == "EDA Page":
    st.sidebar.image("https://miro.medium.com/max/812/1*bpCiEjjuj42XwPSL6EMvtA.png")
    visualised_data = st.sidebar.file_uploader("Upload any csv file to visualise",'csv')
    sidebar_options = st.sidebar.selectbox("Choose among listed options:",
                                           ["Select","EDA Report Page", "Pie Chart", "Bar Chart","Bar Chart2","Box Plot","Scatter Plot", "Line Chart", "Area Chart"], 0)

    if visualised_data is not None:
        visualised_dataframe = pd.read_csv(visualised_data)
        st.write('---')
        st.subheader("Loaded Dataframe")
        st.write(visualised_dataframe)
        st.write("---")

        if sidebar_options != "Select":
            if sidebar_options == "EDA Report Page":
                if visualised_data is not None:
                    create_report(visualised_dataframe).show_browser()
                else:
                    st.info("Waiting for any csv file to be uploaded.")
            if sidebar_options == "Pie Chart":
                if visualised_data is not None:


                    cold1, cold2 = st.columns(2)
                    with cold1:
                        col_name = st.text_input("Enter the name of the column to visualise:")
                        title = st.text_input("Enter any kind of title:")
                    with cold2:
                        if col_name and title:
                            fig = plt.figure(figsize=(15, 5))
                            plt.pie(visualised_dataframe[col_name], labels=np.array(visualised_dataframe[col_name].values))
                            plt.title(title)
                            plt.legend("value1","value2")
                            st.pyplot(fig)
                    st.write("---")
            if sidebar_options == "Scatter Plot":
                col_name1 = st.text_input("Enter the name of the first column:")
                col_name2 = st.text_input("Enter the name of the second column:")
                title_name = st.text_input("Enter any kind of title:")
                if col_name1 and col_name2 and title_name:
                    fig3 = plt.figure(figsize=(15,5))
                    plt.scatter(visualised_dataframe[col_name1],visualised_dataframe[col_name2])
                    plt.title(title_name)
                    plt.xlabel(col_name2)
                    plt.ylabel(col_name1)
                    st.pyplot(fig3)
            if sidebar_options == "Bar Chart":
                col_name2 = st.text_input("Enter the name of the first column:")
                col_name3 = st.text_input("Enter the name of the second column:")
                col_name4 =st.text_input("Enter any kind of title:")
                if col_name2 and col_name3 and col_name4:
                    fig2 = plt.figure(figsize=(15, 5))
                    plt.bar(visualised_dataframe[col_name2], visualised_dataframe[col_name3])
                    plt.title(col_name4)
                    plt.xlabel(col_name2)
                    plt.ylabel(col_name3)
                    st.pyplot(fig2)

            if sidebar_options == "Bar Chart2":
                name_of_col2 = st.text_input("Enter the name of column:")
                title_name2 = st.text_input("Enter ant kind of title:")
                if name_of_col2 and title_name2:
                    st.write(title_name2)
                    st.bar_chart(visualised_dataframe[name_of_col2])
            if sidebar_options == "Line Chart":
                name_of_col1 = st.text_input("Enter the name of the column to visulaise:",key = 1999)
                title_name3 = st.text_input("Enter any kind of title:")
                if name_of_col1 and title_name3:
                    st.write(f"{title_name3} : {name_of_col1} ")
                    st.line_chart(visualised_dataframe[name_of_col1])
            if sidebar_options == "Area Chart":
                name_of_col2 = st.text_input("Enter the name of the column to visualise:",key = 1899)
                title_name3 = st.text_input("Enter any kind of title:")
                if name_of_col2 and title_name3:
                    st.write(f"{title_name3} : {name_of_col2}")
                    st.area_chart(visualised_dataframe[name_of_col2])
            if sidebar_options == "Box Plot":
                name_of_coll1 = st.text_input("Enter the name of the column:")
                if name_of_coll1:
                    fig5 = plt.figure(figsize=(15,5))
                    plt.boxplot(visualised_dataframe[name_of_coll1])
                    st.pyplot(fig5)


    else:
        st.info("Waiting for a csv file to be uploaded")