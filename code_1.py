import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set dataset path
DATA_PATH = os.path.join(os.getcwd(), "Dataset")

# Load data
@st.cache_data
def load_data():
    files = {
        "City": "City.xlsx",
        "Continent": "Continent.xlsx",
        "Country": "Country.xlsx",
        "Item": "Item.xlsx",
        "Mode": "Mode.xlsx",
        "Region": "Region.xlsx",
        "Transaction": "Transaction.xlsx",
        "Type": "Type.xlsx",
        "User": "User.xlsx"
    }
    return {name: pd.read_excel(os.path.join(DATA_PATH, file)) for name, file in files.items()}

# Data Preprocessing
def preprocess_data(df):
    city = df["City"]
    user = df["User"]
    trans = df["Transaction"]
    item = df["Item"]
    typ = df["Type"]

    merged = trans.merge(user, on="UserId", how="left") \
                  .merge(item, on="AttractionId", how="left") \
                  .merge(typ, on="AttractionTypeId", how="left") \
                  .merge(city.add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left")

    merged = merged.dropna()
    return merged

# Regression Model
def train_regression(df):
    X = df[['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']]
    y = df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

# Classification Model
def train_classification(df):
    X = df[['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId']]
    y = df['VisitMode']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, acc

# Recommendation System
def recommend(df, user_id):
    pivot = df.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)
    sim_matrix = cosine_similarity(pivot)
    sim_df = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)

    if user_id not in sim_df.index:
        return pd.DataFrame({'Attraction': ["Not enough data for this user."], 'Rating': [None]})

    similar_users = sim_df[user_id].sort_values(ascending=False)[1:4].index
    attractions = df[df['UserId'].isin(similar_users)].groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(5)
    return attractions.reset_index()

# Main Streamlit App
def main():
    st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")
    st.title("🧳 Tourism Experience Analytics")

    data = load_data()
    df = preprocess_data(data)

    # Sidebar Inputs
    st.sidebar.header("User Input")
    visit_year = st.sidebar.selectbox("Visit Year", sorted(df['VisitYear'].unique()))
    visit_month = st.sidebar.selectbox("Visit Month", sorted(df['VisitMonth'].unique()))
    continent = st.sidebar.selectbox("Continent", sorted(df['ContinentId'].unique()))
    country = st.sidebar.selectbox("Country", sorted(df['CountryId'].unique()))
    attraction_type = st.sidebar.selectbox("Attraction Type", sorted(df['AttractionTypeId'].unique()))
    user_id = st.sidebar.selectbox("User ID", sorted(df['UserId'].unique()))

    st.header("📊 Predictions and Recommendations")

    reg_model, r2 = train_regression(df)
    class_model, label_enc, acc = train_classification(df)

    # Prepare inputs as DataFrames (Fix warning)
    input_data_reg = pd.DataFrame([{
        'VisitYear': visit_year,
        'VisitMonth': visit_month,
        'AttractionTypeId': attraction_type,
        'ContinentId': continent,
        'CountryId': country
    }])

    input_data_class = pd.DataFrame([{
        'VisitYear': visit_year,
        'VisitMonth': visit_month,
        'ContinentId': continent,
        'CountryId': country
    }])

    # Predictions
    pred_rating = reg_model.predict(input_data_reg)[0]
    pred_visit_mode = label_enc.inverse_transform(class_model.predict(input_data_class))[0]

    st.subheader("🎯 Predicted Rating")
    st.success(f"{pred_rating:.2f} / 5.0")

    st.subheader("🧭 Predicted Visit Mode")
    st.info(pred_visit_mode)

    st.subheader("🧪 Model Performance")
    st.write(f"**Regression R² Score:** {r2:.2f}")
    st.write(f"**Classification Accuracy:** {acc:.2f}")

    # Recommendations
    st.subheader("🌟 Recommended Attractions")
    recommendations = recommend(df, user_id)
    st.dataframe(recommendations)

    # Export Results
    result_df = pd.DataFrame({
        'Predicted Rating': [round(pred_rating, 2)],
        'Predicted Visit Mode': [pred_visit_mode]
    })
    result_with_recommendations = pd.concat([
        result_df,
        recommendations.rename(columns={"Attraction": "Recommended Attraction", "Rating": "Avg Rating"})
    ], axis=1)

    csv = result_with_recommendations.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions & Recommendations", csv,
                       file_name='tourism_recommendations.csv', mime='text/csv')

    # Visualization: Attraction Types
    st.subheader("📈 Top Attraction Types")
    fig, ax = plt.subplots()
    sns.countplot(y='AttractionType', data=df, order=df['AttractionType'].value_counts().index[:10], ax=ax)
    st.pyplot(fig)

    # Visualization: Most Visited Cities
    st.subheader("🗺️ Most Visited Cities")
    city_counts = df['User_CityName'].value_counts().head(10)
    st.bar_chart(city_counts)

if __name__ == '__main__':
    main()
