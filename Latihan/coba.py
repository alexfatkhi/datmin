import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import warnings
import os
warnings.filterwarnings("ignore")
import datetime


with st.sidebar :
    selected = option_menu('MENU',
    ['Testing data baru',
    'Tentang data training'],
    default_index=0) 


if (selected == 'Testing data baru') : 

    st.title('Data Mining')

    def run_code():

        data=pd.read_csv('data.csv')
        #==============================================================================================
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.write("Melihat data testing yang diinput : ")
        st.write(data_new)



        data.drop('Unnamed: 32', axis = 1, inplace = True)#menghapus kolom dengan nama 'unnamed' dari data

        x_train = data.drop(columns = 'diagnosis') #berisi fitur-fitur dari DataFrame data,
        #kecuali kolom 'diagnosis'. Fungsi drop digunakan untuk menghapus kolom 'diagnosis' dari dataset.

        # Getting Predicting Value
        y_train = data['diagnosis']

        data_new.drop('Unnamed: 32', axis = 1, inplace = True)

        data_new.head()

        # Getting Features

        x_test = data_new.drop(columns = 'diagnosis') #berisi fitur-fitur dari DataFrame data,
        #kecuali kolom 'diagnosis'. Fungsi drop digunakan untuk menghapus kolom 'diagnosis' dari dataset.

        # Getting Predicting Value
        y_test = data_new['diagnosis'] #berisi nilai yang akan diprediksi, yaitu kolom 'diagnosis' dari dataset data.

        #==============================================================================================
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.success(f"Jumlah data yang di test: {len(x_test)} baris")

        

        from sklearn.tree import DecisionTreeClassifier
        dtree = DecisionTreeClassifier(max_depth=6, random_state=123)

        dtree.fit(x_train,y_train)

        from sklearn import tree


        import matplotlib.pyplot as plt
        # Misalkan ada tambahan fitur 'texture_mean' dan 'area_mean'
        additional_features = ['radius_mean', 'area_mean']

        # Gabungkan fitur tambahan dengan fitur yang sudah ada
        all_features = list(data.columns) + additional_features
        fig= plt.figure(figsize=(50,30))
        _ = tree.plot_tree(dtree, feature_names=all_features, class_names=data.diagnosis, filled=True)

        #==============================================================================================
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.write("Decision Tree : ")
        st.pyplot(fig)

        y_pred=dtree.predict(x_test)
        a1 = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        #==============================================================================================
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.write("Tabel Confusion Matrix : ")
        st.write(a1)

        from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error


        conf_matrix = confusion_matrix(y_test, y_pred)


        # Create the heatmap
        fig, ax = plt.subplots()
        p = sns.heatmap(conf_matrix, annot=True, cmap="Pastel1", fmt='g', ax=ax)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        #==============================================================================================
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.write("Confusion Matrix : ")
        # Display the heatmap in Streamlit
        st.pyplot(fig)


        #==============================================================================================
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.write("Tabel presisi, recall, f1, akurasi : ")
        st.text(classification_report(y_test,y_pred))
        st.write("Training Score: ",dtree.score(x_train,y_train)*100)

        #==============================================================================================
        st.write("<br><br><br>", unsafe_allow_html=True)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {accuracy:.2f}") #mencetak akurasi model secara desimal
        st.success(f"Accuracy: {accuracy * 100:.3f}%") #mencetak akurasi model (dalam persentase) terhadap data uji

        #==============================================================================================
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.write("Heatmap Accuracy Score : ")
        plt.figure(figsize=(10, 6))
        sns.heatmap([[accuracy]], annot=True, cmap="Pastel1", fmt='.3%',)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Accuracy Score')
        plt.show()
        st.pyplot(plt.gcf())

    uploaded_file = st.file_uploader("Upload data testing:", type=["csv"])

    if uploaded_file is not None:
        # Membaca file CSV yang diunggah menjadi DataFrame
        data_new = pd.read_csv(uploaded_file)

    # Tombol "Run" untuk menjalankan kode setelah upload file
    if st.button("Jalankan") and uploaded_file is not None:
        run_code()



if (selected == 'Tentang data training') :
    data=pd.read_csv('data.csv')

    st.write(data.head())
    st.write("<br><br><br>", unsafe_allow_html=True)

    st.write(data.describe())
    st.write("<br><br><br>", unsafe_allow_html=True)

    st.write(data.info())
    st.write("<br><br><br>", unsafe_allow_html=True)

    st.write(data.shape)
    st.write("<br><br><br>", unsafe_allow_html=True)
    
    st.write(data.columns)
    st.write("<br><br><br>", unsafe_allow_html=True)

    st.write(data.value_counts)
    st.write("<br><br><br>", unsafe_allow_html=True)

    st.write(data.dtypes)
    st.write("<br><br><br>", unsafe_allow_html=True)

    st.write(data.drop('Unnamed: 32', axis = 1, inplace = True))
    st.write("<br><br><br>", unsafe_allow_html=True)


    #daftar kolom yang akan divisualisasikan dalam pairplot.
    mean_col = ['diagnosis','radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

    pair_plot = sns.pairplot(data[mean_col], hue='diagnosis', palette='Reds')

    # Menampilkan plot di Streamlit dengan st.pyplot()
    st.pyplot(pair_plot.fig)
    st.write("<br><br><br>", unsafe_allow_html=True)

    violin_plot = sns.violinplot(x="smoothness_mean", y="perimeter_mean", data=data)

    # Menampilkan plot di Streamlit dengan st.pyplot()
    st.pyplot(violin_plot.figure)

