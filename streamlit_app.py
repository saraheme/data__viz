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
import altair as alt

def df_aff(df):
    st.header('Affichage de la table')
    st.write(df)
    st.info("La table contient " + str(len(df)) + " enregistrements et contient " + str(len(df.columns)) + " colonnes.")


def df_desc(df):
    st.header('Description du jeu de données')
    st.write(df.describe()) #Description
    #Sélection d'une variable pour afficher son histogramme et boxplot
    st.write("Affichage histogramme et boxplot pour une variable donnée")
    var_select = st.selectbox('Sélectionner une variable :', [col for col in df.columns if df[col].dtype != 'object']) #if df[col].dtype != 'object'

    hist = alt.Chart(df).mark_bar().encode(
        alt.X(var_select, bin=True),
        y='count()',
    ).properties(
        width=600,
        height=400,
        title='Histogramme de {}'.format(var_select)
    )
    boxplot = alt.Chart(df).mark_boxplot().encode(
        y=alt.Y(var_select),
    ).properties(
        width=600,
        height=400,
        title='Boxplot de {}'.format(var_select)
    )
    st.altair_chart(hist)
    st.altair_chart(boxplot)


def df_repartition(df):
    st.header('Camembert des occurences dans une colonne')
    #Sélection de la colonne
    selected_column = st.selectbox('Sélectionnez une colonne pour afficher les occurences (on affiche le camembert s il y a moins de 20 occurences différentes)', list(df.columns))

    #Affichage des occurences en camembert
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
    x_axis_val = col1.selectbox("Sélectionnez l'axe X", options=df.columns,key="plot_var1")
    y_axis_val = col2.selectbox("Sélectionnez l'axe Y", options=df.columns,key="plot_var2")
    st.text("La colonne " + str(x_axis_val) + " est de type " + str(df[x_axis_val].dtypes))
    st.text("La colonne " + str(y_axis_val) + " est de type " + str(df[y_axis_val].dtypes))




    if (df[x_axis_val].dtypes == object and df[y_axis_val].dtypes != object ):
        select_categories = st.multiselect(label = 'Sélectionner les variables pour lesquelles on affiche les boxplot :', options = ["All"] + df[x_axis_val].unique().tolist(),default="All")
        if "All" in select_categories:
            categories = df[x_axis_val].unique().tolist()
        else:
            categories = select_categories
        data = [df[df[x_axis_val] == category][y_axis_val] for category in categories]
        fig, ax = plt.subplots()
        sns.boxplot(data=data, ax=ax)
        ax.set_xticklabels(categories)
        ax.set_title(f"Boxplot de '{x_axis_val}' par catégorie")
        st.pyplot(fig)

    elif (df[y_axis_val].dtypes == object and df[x_axis_val].dtypes != object):
        select_categories = st.multiselect(label='Sélectionner les variables pour lesquelles on affiche les boxplot :',
                                           options=["All"] + df[y_axis_val].unique().tolist(), default="All")
        if "All" in select_categories:
            categories = df[y_axis_val].unique().tolist()
        else:
            categories = select_categories
        data = [df[df[y_axis_val] == category][x_axis_val] for category in categories]
        fig, ax = plt.subplots()
        sns.boxplot(data=data, ax=ax)
        ax.set_xticklabels(categories)
        ax.set_title(f"Boxplot de '{y_axis_val}' par catégorie")
        st.pyplot(fig)
    else:
        plot = px.scatter(df, x=x_axis_val, y=y_axis_val)
        st.plotly_chart(plot, use_container_width=True)



def reg_lin_multiple(df):
    df_quantitatif = df.select_dtypes(exclude=['object']) #On garde que les variables quantitatives
    selected_columns = st.multiselect('Sélectionnez les colonnes à inclure dans la régression (variables quantitatives)', list(df_quantitatif.columns))
    if len(selected_columns) >= 2:
        #Sélection de la variable à expliquer
        variable_expliquee = st.selectbox('Sélectionnez la variable à expliquer (Y)', selected_columns)

        #On stocke les variables explicatives
        variables_explicatives = [col for col in selected_columns if col != variable_expliquee]

        #Modèle
        X = df[variables_explicatives]
        y = df[variable_expliquee]
        select_test_size = st.number_input(label = "Sélectionnez la taille de la base test souhaitée (en pourcentage)",value=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=select_test_size/100, random_state=0)

        # execution de la Régression linéaire
        reg = LinearRegression().fit(X_train, y_train)

        # Prédiction des valeurs de y pour les données de test
        y_pred = reg.predict(X_train)

        # Affichage des résultats
        st.write("Coefficients : ")
        for i in range(len(variables_explicatives)):
            st.info(str(variables_explicatives[i]) + " : " +str(round(reg.coef_[i],3)))
        st.write("Moyenne quadratique des résidus (MQ) : %.2f" % np.mean((reg.predict(X_train) - y_train) ** 2))
        st.write("Score de prédiction (R²) : %.2f" % reg.score(X_train, y_train))
        st.write("Prédictions pour les données de test : ", y_pred)

        st.write("Courbe de régression et de résidus pour la base test :")
        #Traçage reg

        fig_test, (ax1_test, ax2_test) = plt.subplots(ncols=2, figsize=(10, 4))
        sns.regplot(x=y_test, y=reg.predict(X_test), ax=ax1_test)
        ax1_test.set_xlabel(variable_expliquee)
        ax1_test.set_ylabel('Prédiction')
        ax1_test.set_title('Régression linéaire multiple')

        # Tracé des résidus
        residuals_test = y_test - reg.predict(X_test)
        sns.residplot(x=y_test, y=residuals_test, ax=ax2_test)
        ax2_test.set_xlabel(variable_expliquee)
        ax2_test.set_ylabel('Résidus')
        ax2_test.set_title('Résidus de la régression linéaire multiple')
        st.pyplot(fig_test)

        st.write("Courbe de régression et de résidus pour la base train :")
        #Traçage reg

        fig_train, (ax1_train, ax2_train) = plt.subplots(ncols=2, figsize=(10, 4))
        sns.regplot(x=y_train, y=reg.predict(X_train), ax=ax1_train)
        ax1_train.set_xlabel(variable_expliquee)
        ax1_train.set_ylabel('Prédiction')
        ax1_train.set_title('Régression linéaire multiple')

        # Tracé des résidus
        residuals_train = y_train - reg.predict(X_train)
        sns.residplot(x=y_train, y=residuals_train, ax=ax2_train)
        ax2_train.set_xlabel(variable_expliquee)
        ax2_train.set_ylabel('Résidus')
        ax2_train.set_title('Résidus de la régression linéaire multiple')
        st.pyplot(fig_train)


def correlation(df):
    st.title('Corrélation entre deux variables quantitatives du jeu de données')
    #Sélection des colonnes
    df_quantitatif = df.select_dtypes(exclude=['object']) #On garde que les variables quantitatives
    st.info("Sélectionnez deux colonnes pour calculer la corrélation")
    col1, col2 = st.columns(2)
    x_axis_val = col1.selectbox("Choix variable : ", options=df_quantitatif.columns, key="corr_var1")
    y_axis_val = col2.selectbox("Choix variable : ", options=df_quantitatif.columns, key="corr_var2")
    selected_columns = [x_axis_val,y_axis_val]
    #Sélection du type de correlation
    correlation_type = st.radio("Sélectionnez le type de corrélation souhaité :",options=['pearson','kendall','spearman'],horizontal=True)

    #Calcul de la corrélation
    corr = df[selected_columns].corr(method=correlation_type)
    #Tracé de la heatmap de corrélation
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


st.title("Analyse d'un jeu de données")


st.sidebar.title('Importation')
choix_separateur = st.sidebar.radio(label='Format séparateur', options=[',',';',' ','/','/t'], horizontal=True)
choix_decimal = st.sidebar.radio(label='Format décimal', options=['.',','], horizontal=True)

upload_file = st.sidebar.file_uploader('Importez votre table')
st.sidebar.title('Actions')
#Sélection
options = st.sidebar.radio('Quelle analyse voulez vous faire ?',
                           ['Affichage de la table', 'Description du jeu de données','Répartition des valeurs de chaque colonne','Visualisation','Correlation','Régression linéaire'])
if upload_file is None:
    st.text("Pour commencer importez une table.")
else:
    df = pd.read_csv(upload_file,sep=choix_separateur, encoding='latin-1',decimal= choix_decimal)

    if options == 'Affichage de la table':
        df_aff(df)
    elif options == 'Description du jeu de données':
        df_desc(df)
    elif options == 'Répartition des valeurs de chaque colonne':
        df_repartition(df)
    elif options == 'Visualisation':
        interactive_plot(df)
    elif options == 'Régression linéaire':
        reg_lin_multiple(df)
    elif options == 'Correlation':
        correlation(df)

