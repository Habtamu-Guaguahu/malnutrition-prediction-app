import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Acute Malnutrition Risk Prediction",
    layout="wide"
)

# =====================================================
# LOAD MODEL AND THRESHOLD
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("malnutrition_rf_pipeline.pkl")

@st.cache_resource
def load_threshold():
    return joblib.load("youden_threshold.pkl")

model = load_model()
threshold = load_threshold()

# =====================================================
# HEADER
# =====================================================
st.title("🌍 Acute Malnutrition Risk Prediction Tool")
st.markdown("### Random Forest Machine Learning Model")
st.markdown("Predicting Acute Malnutrition in Sub-Saharan Africa")
st.markdown("---")

# =====================================================
# INPUT FORM
# =====================================================
with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # -----------------------
    # COLUMN 1
    # -----------------------
    with col1:
        matage = st.selectbox("Maternal Age",
            ['15-19','20-24','25-29','30-34','35-39','40-44','45-49'])
        placeresid = st.selectbox("Place of Residence",
            ['urban','rural'])
        latrine = st.selectbox("Latrine Available",
            ['no','yes'])
        famsize = st.selectbox("Family Size",
            ['1-5','6 and more'])
        num5child = st.selectbox("Under-5 Children",
            ['less than 2','3 and more'])
        sexofhead = st.selectbox("Sex of Household Head",
            ['male','female'])
        litracy = st.selectbox("Maternal Literacy",
            [' cannot read at all','able to read only parts of sentence','able to read whole sentence'])
        bottlefeeding = st.selectbox("Bottle Feeding",
            ['no','yes'])

    # -----------------------
    # COLUMN 2
    # -----------------------
    with col2:
        disposalofchild = st.selectbox("Child Stool Disposal",
            ['appropraite','inappropraite'])
        maritalstatus = st.selectbox("Marital Status",
            ['unmarried/widowed/divorced','married'])
        working = st.selectbox("Mother Working",
            ['no','yes'])
        twin = st.selectbox("Birth Type",
            ['sinltone','multiple'])
        childgender = st.selectbox("Child Gender",
            ['male','female'])
        childage = st.selectbox("Child Age",
            ['0-11 months','12-23 months','24-35 months','36-47 months','48-59 months'])
        birthinterval = st.selectbox("Birth Interval",
            ['less than 2 years','2 years','3 years','4 years and above'])
        breastfed = st.selectbox("Breastfeeding Status",
            ['never breastfed','ever breastfed, not currently breastfee',' still breastfeeding'])

    # -----------------------
    # COLUMN 3
    # -----------------------
    with col3:
        anc = st.selectbox("Antenatal Care",
            ['adequate','inadequate','not at all'])
        birthsize = st.selectbox("Birth Size",
            ['very large','larger than average','average','smaller than average','very small'])
        diarrhea = st.selectbox("Recent Diarrhea",
            ['no','yes'])
        fever = st.selectbox("Recent Fever",
            ['no','yes'])
        vitamina = st.selectbox("Vitamin A Supplement",
            ['no','yes'])
        maternaldecision = st.selectbox("Maternal Decision Power",
            ['no','yes'])
        dietdivers = st.selectbox("Diet Diversity",
            ['adequate','inadequate'])
        watersource = st.selectbox("Water Source",
            ['unimproved','improved'])
        ari = st.selectbox("Acute Respiratory Infection",
            ['no','yes'])
        country = st.selectbox("Country",
            ['brundi','comoros','ethiopia','kenya','madagascar','malawi',
             'mozambique','rwanda','tanzania','uganda','zambia','zimbabwe'])

    submitted = st.form_submit_button("Predict Risk")

# =====================================================
# PREDICTION
# =====================================================
if submitted:
    # Create a DataFrame with raw categorical strings
    input_df = pd.DataFrame([{
        'matage': matage,
        'placeresid': placeresid,
        'latrine': latrine,
        'famsize': famsize,
        'num5child': num5child,
        'sexofhead': sexofhead,
        'litracy': litracy,
        'bottlefeeding': bottlefeeding,
        'disposalofchild': disposalofchild,
        'maritalstatus': maritalstatus,
        'working': working,
        'twin': twin,
        'childgender': childgender,
        'childage': childage,
        'birthinterval': birthinterval,
        'breastfed': breastfed,
        'anc': anc,
        'birthsize': birthsize,
        'diarrhea': diarrhea,
        'fever': fever,
        'vitamina': vitamina,
        'maternaldecision': maternaldecision,
        'dietdivers': dietdivers,
        'watersource': watersource,
        'ari': ari,
        'country': country
    }])

    try:
        probability = model.predict_proba(input_df)[0][1]
        st.markdown("---")
        st.subheader(f"Predicted Probability of Acute Malnutrition: {probability:.3f}")

        if probability >= threshold:
            st.error("⚠️ High Risk of Acute Malnutrition")
        else:
            st.success("✅ Low Risk of Acute Malnutrition")

        st.info(f"Decision Threshold (Youden Index): {threshold:.3f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =====================================================
# MODEL PERFORMANCE PANEL
# =====================================================
st.markdown("---")
st.markdown("### Model Performance")
st.write("Random Forest Classifier")
st.write("Feature Selection: Consensus (≥2 methods)")
st.write("Threshold Optimization: Youden Index")
# Optional metrics (replace with your real ones)
# st.write("ROC-AUC: 0.91")
# st.write("Sensitivity: 0.85")
# st.write("Specificity: 0.83")

# =====================================================
# DISCLAIMER
# =====================================================
st.markdown("---")
st.caption(
    "This tool is developed for research and screening purposes. "
    "It does not replace clinical judgment or formal nutritional assessment."
)