import streamlit as st
import pandas as pd
from collections import OrderedDict
from pathlib import Path
import plotly.express as px
from PIL import Image
from utils.predict import get_predictions


st.title("Costa Rica Household Poverty Predictor")
st.subheader("A Web App by [Henry Fung](https://github.com/hfung4)")
st.write(
    """Using data on the characterisitics of Costa Rican households, 
         we use a trained Random Forest model to predict the poverty level of households.
         There are four levels of poverty: 1: Extreme Poverty; 2: Moderate Poverty; 3: Vulnerable; 4: Not Vulnerable.
         """
)

st.text("")
st.write(
    """The train dataset was provided by the [Inter-American Development Bank](https://www.iadb.org/en).
        The purpose of this model is to predict the poverty level of the world's poorest families
         in order to make sure  aid reaches families who are the most in need. In this region of the world,
         households typically can’t provide the necessary income and expense records to prove that they qualify. Thus, we need
         to rely other household observable attributes to predict their level of need.   
         """
)

st.text("")
st.text("")

st.header("Upload the household characteristics data")
st.write(
    "The input data must contains the same observed household attributes as described [here]('https://www.kaggle.com/competitions/costa-rican-household-poverty-prediction/data')."
)

test_data_file = st.file_uploader("Upload your household attributes data")

# Do this only after the user inputs test data
if test_data_file is not None:
    X_test = pd.read_csv(test_data_file)

    st.markdown(
        f"**Your data is loaded! It has {X_test.shape[0]} individuals and {len(X_test.idhogar.unique())} households.**"
    )
else:
    # Load default test data
    X_test = pd.read_csv(Path("data", "test.csv"))
    st.markdown(
        f"**If you choose not to load a dataset, we will use a default test dataset. It has {X_test.shape[0]} individuals and {len(X_test.idhogar.unique())} households.**"
    )

st.text("")
st.text("")

st.header("Predictions")
st.write("We will use a previously trained Random Forest model to make predictions.")
st.write(
    "Predictions are made at the household level, and then cross-walked to each individual (i.e. each individual in the same household gets the same prediction)."
)


# Distribution of predictions

test_res = get_predictions(X_test)

# Create final predictions dataframe
df_final_predictions_test = test_res["test_ids"].copy()
df_final_predictions_test["predicted_poverty_level"] = test_res["predictions"]

# Output the final predictions
df_final_predictions_test.to_csv(Path("outputs", "final_predictions.csv"))

# Distribution of predictions

# Recode the predictions
# Poverty label mapping
poverty_label_mapping = OrderedDict(
    {1: "extreme", 2: "moderate", 3: "vulnerable", 4: "not vulnerable"}
)

df_plot = df_final_predictions_test.copy()
df_plot["predicted_poverty_level"] = df_plot.predicted_poverty_level.astype(
    "int"
).replace(poverty_label_mapping)


fig = px.histogram(
    df_plot,
    x="predicted_poverty_level",
    barmode="overlay",
    marginal="box",
    labels="Poverty level",
)

fig.update_layout(
    autosize=False,
    width=800,
    height=600,
    font={"size": 18},
)

st.plotly_chart(fig)


st.text("")
st.text("")
st.text("")
st.text("")


st.header("Trained model performance and characteristics")
st.write(
    "Several evaluation metrics and interpretation of the model (e.g., feature importance) is provided below."
)

st.text("")
st.text("")

st.subheader("Feature importance")
st.write(
    """Top 10 feature importance from the trained model.
         Feature importance is computed from the mean decrease in
         GINI due to a split using the feature over the bagged trees in the 
         ensemble. A large value indicates that the feature is important."""
)

# Get image
feature_importance_image = Image.open(
    Path("utils", "artifacts", "feature_importance.png")
)
st.image(
    feature_importance_image,
    caption="""The most important predictors of poverty 
         are related to cellphone ownership, education (inst, and escolari), number of dependencies (children/elderly)
         age statistics in the household, and living conditions (house_structure_score, rooms_per_capita).""",
    width=800,
)

st.text("")
st.text("")

# Confusion matrix
st.subheader("Confusion matrix")

# Get image
cm_image = Image.open(Path("utils", "artifacts", "confusion_matrix.png"))
st.image(
    cm_image,
    caption="""The model seems to be only good in predicting the majority class (not vulnerable), 
    despite my attempt in using methods to address class imbalanced in training (e.g., SMOTE, cost-sensitive learning).
    Let's hone in on the more interesting class--extreme poverty. The model only correctly predicts 0.31 of all observations that actually have extreme poverty. 
    The model has 31% recall for the extreme poverty class. (Not shown in this confusion matrix): 
    The model makes 216 predictions that are extreme poverty. Of those, 69 are correct. The model therefore has 39/216 = 0.32 precision.
    """,
)


st.text("")
st.text("")

# Precision-Recall curve and average precision

# PRC
st.subheader("Precision-Recall Curve and Average Precision")

st.write(
    """Due to class imbalance (>60% of samples in the train set is 'Not vulnerable'), I
         used precision-recall curve and average precision to evaluate model performance.
         """
)

# Get image
prc_image = Image.open(Path("utils", "artifacts", "prc.png"))

st.image(
    prc_image,
    caption="""Again, the precision-recall curve and average precision indicates that 
    the model performs well only for the majority class (not vulnerable). Note that the 
    baseline value of average precision 'no better than random guessing' is the proportion 
    of the class within the sample, indicated in the parenthesis in the above plot.
    """,
    width=1000,
)
