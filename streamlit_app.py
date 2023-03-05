import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

def df_aff(df):
    st.header('Affichage de la base de données')
    st.write(df)
    st.info("La base de données contient " + str(len(df)) + " enregistrements et contient " + str(len(df.columns)) + " colonnes.")


def df_desc(df):
    st.header('Description du jeu de données')
    st.write(df.describe())

def df_repartition(df):
    # Titre de l'application
    st.header('Camembert des occurences dans une colonne')
    # Sélection de la colonne
    selected_column = st.selectbox('Sélectionnez une colonne pour afficher les occurences (on affiche le camembert s il y a moins de 20 occurences différentes)', list(df.columns))

    # Affichage des occurences en camembert
    if selected_column is not None:
        nombre_differents = df[selected_column].nunique()
        if nombre_differents < 20:
            st.write('Voici le camembert des occurences dans la colonne :', selected_column)
            fig, ax = plt.subplots()
            df[selected_column].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_title(selected_column)
            st.pyplot(fig)
        else:
            st.write("Trop de valeurs uniques")



def interactive_plot(df):
    col1, col2 = st.columns(2)

    x_axis_val = col1.selectbox("Sélectionnez l'axe X", options=df.columns)
    y_axis_val = col2.selectbox("Sélectionnez l'axe Y", options=df.columns)
    plot = px.scatter(df, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot, use_container_width=True)



def reg_lin_multiple(df):
    df_quantitatif = df.select_dtypes(exclude=['object'])
    selected_columns = st.multiselect('Sélectionnez les colonnes à inclure dans la régression (variables quantitatives)', list(df_quantitatif.columns))
    if len(selected_columns) >= 2:
        # Sélection de la variable dépendante
        target_column = st.selectbox('Sélectionnez la variable dépendante (Y)', selected_columns)

        # Sélection des variables indépendantes
        feature_columns = [col for col in selected_columns if col != target_column]

        # Construction du modèle
        X = df[feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Création d'un modèle de régression linéaire
        reg = LinearRegression().fit(X_train, y_train)

        # Prédiction des valeurs de y pour les données de test
        y_pred = reg.predict(X_train)

        # Affichage des résultats
        st.write("Coefficients : ")
        for i in range(len(feature_columns)):
            st.info(str(feature_columns[i]) + " : " +str(round(reg.coef_[i],3)))
        st.write("Moyenne quadratique des résidus (MQ) : %.2f" % np.mean((reg.predict(X_train) - y_train) ** 2))
        st.write("Score de prédiction (R²) : %.2f" % reg.score(X_train, y_train))
        st.write("Prédictions pour les données de test : ", y_pred)

        st.write("Courbe de régression et de résidus pour la base test :")
        #Traçage reg

        fig_test, (ax1_test, ax2_test) = plt.subplots(ncols=2, figsize=(10, 4))
        sns.regplot(x=y_test, y=reg.predict(X_test), ax=ax1_test)
        ax1_test.set_xlabel(target_column)
        ax1_test.set_ylabel('Prédiction')
        ax1_test.set_title('Régression linéaire multiple')

        # Tracé des résidus
        residuals_test = y_test - reg.predict(X_test)
        sns.residplot(x=y_test, y=residuals_test, ax=ax2_test)
        ax2_test.set_xlabel(target_column)
        ax2_test.set_ylabel('Résidus')
        ax2_test.set_title('Résidus de la régression linéaire multiple')
        st.pyplot(fig_test)

        st.write("Courbe de régression et de résidus pour la base train :")
        #Traçage reg

        fig_train, (ax1_train, ax2_train) = plt.subplots(ncols=2, figsize=(10, 4))
        sns.regplot(x=y_train, y=reg.predict(X_train), ax=ax1_train)
        ax1_train.set_xlabel(target_column)
        ax1_train.set_ylabel('Prédiction')
        ax1_train.set_title('Régression linéaire multiple')

        # Tracé des résidus
        residuals_train = y_train - reg.predict(X_train)
        sns.residplot(x=y_train, y=residuals_train, ax=ax2_train)
        ax2_train.set_xlabel(target_column)
        ax2_train.set_ylabel('Résidus')
        ax2_train.set_title('Résidus de la régression linéaire multiple')
        st.pyplot(fig_train)


def correlation(df):
    st.title('Corrélation entre deux variables quantitatives du jeu de données')
    # Sélection des colonnes
    df_quantitatif = df.select_dtypes(exclude=['object'])
    selected_columns = st.multiselect('Sélectionnez deux colonnes pour calculer la corrélation', list(df_quantitatif.columns))

    # Vérification que deux colonnes ont été sélectionnées
    if len(selected_columns) == 2:
        # Calcul de la corrélation
        corr = df[selected_columns].corr()
        # Tracé de la heatmap de corrélation
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Heatmap de corrélation')
        st.pyplot(fig)
        #Tracé de la courbe de corrélation
        fig, ax = plt.subplots()
        ax.plot(df[selected_columns[0]], df[selected_columns[1]], 'o')
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])
        ax.set_title('Courbe de corrélation')
        st.pyplot(fig)
    else:
        st.write('Veuillez sélectionner exactement deux colonnes.')


def reg_logistique(df):
    selected_columns = st.multiselect(
        'Sélectionnez les colonnes à inclure dans la régression (variables quantitatives)',list(df.columns))
    if len(selected_columns) >= 2:
        # Sélection de la variable dépendante
        target_column = st.selectbox('Sélectionnez la variable dépendante (Y)', selected_columns)

        # Sélection des variables indépendantes
        feature_columns = [col for col in selected_columns if col != target_column]

        # Construction du modèle
        X = df[feature_columns]
        y = df[target_column]


        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        # Créer un objet de modèle de régression logistique
        model = LogisticRegression()

        # Entraîner le modèle sur les données d'entraînement
        model.fit(X_train, y_train)

        # Prédire les valeurs de sortie pour les données de test
        y_pred = model.predict(X_test)

        # Évaluer les performances du modèle
        accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write("Accuracy (qualité de la prédiction):", accuracy)

        st.write("Coefficients : ")
        for i in range(len(feature_columns)):
            st.info(str(feature_columns[i]) + " : " + str(model.coef_[0][i]))

        # Linear
        y_test_logistic = model.predict_proba(X=X_test)[:, 1]
        fpr_log_test, tpr_log_test, _ = metrics.roc_curve(y_test.values, y_test_logistic)
        roc_auc_log_test = metrics.auc(fpr_log_test, tpr_log_test)
        st.write('Aire sous la courbe pour la base test :', roc_auc_log_test)

        # %%ROC

        # ROC plot
        lw = 2
        fig, ax = plt.subplots()
        ax.plot(fpr_log_test, tpr_log_test, color='darkorange',
                 lw=lw, label='ROC curve RF (area = %0.2f)' % roc_auc_log_test)
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Courbe ROC')
        ax.legend(loc="lower right")
        st.pyplot(fig)







st.title("Analyse d'un jeu de données")


st.sidebar.title('Importation')
choix_separateur = st.sidebar.radio(label='Format séparateur', options=[',',';',' ','/','/t'], horizontal=True)
choix_decimal = st.sidebar.radio(label='Format décimal', options=['.',','], horizontal=True)

upload_file = st.sidebar.file_uploader('Importez votre base de données')
st.sidebar.title('Actions')
#Sélection
options = st.sidebar.radio('Quelle analyse voulez vous faire ?',
                           ['Affichage de la base de données', 'Description du jeu de données','Répartition des valeurs de chaque colonne','Scatter Plot','Régression linéaire','Correlation','Régression logistique'])
if upload_file is None:

    st.text("Pour commencer importez une base de données")
else:
    df = pd.read_csv(upload_file,sep=choix_separateur, encoding='latin-1',decimal= choix_decimal)

    if options == 'Affichage de la base de données':
        df_aff(df)
    elif options == 'Description du jeu de données':
        df_desc(df)
    elif options == 'Répartition des valeurs de chaque colonne':
        df_repartition(df)
    elif options == 'Scatter Plot':
        interactive_plot(df)
    elif options == 'Régression linéaire':
        reg_lin_multiple(df)
    elif options == 'Correlation':
        correlation(df)
    elif options == 'Régression logistique':
        reg_logistique(df)
