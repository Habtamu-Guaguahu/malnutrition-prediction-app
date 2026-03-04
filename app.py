import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Acute Malnutrition Risk Prediction",
    page_icon="🌍",
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
# SIDEBAR
# =====================================================
st.sidebar.title("Model Overview")
st.sidebar.markdown("""
**Algorithm:** Random Forest  
**Feature Selection:** Consensus (≥2 methods)  
**Threshold Selection:** Youden Index  

Predicting acute malnutrition among children under five  
in Sub-Saharan Africa.
""")

# =====================================================
# HEADER
# =====================================================
st.title("🌍 Acute Malnutrition Risk Prediction")
st.markdown("""
This tool estimates the probability of **acute malnutrition (wasting)**  
using a trained Random Forest machine learning model.
""")
st.markdown("---")

# =====================================================
# INPUT FORM
# =====================================================
with st.form("prediction_form"):

    # ==============================
    # MATERNAL CHARACTERISTICS
    # ==============================
    st.markdown("## 👩 Maternal Characteristics")
    col1, col2 = st.columns(2)

    with col1:
        matage = st.selectbox("Maternal Age",
            ['15-19','20-24','25-29','30-34','35-39','40-44','45-49'])
        litracy = st.selectbox("Maternal Literacy",
            [' cannot read at all','able to read only parts of sentence','able to read whole sentence'])
        maritalstatus = st.selectbox("Marital Status",
            ['unmarried/widowed/divorced','married'])
        working = st.selectbox("Mother Working",
            ['no','yes'])

    with col2:
        anc = st.selectbox("Antenatal Care",
            ['adequate','inadequate','not at all'])
        maternaldecision = st.selectbox("Maternal Decision Power",
            ['no','yes'])
        birthinterval = st.selectbox("Birth Interval",
            ['less than 2 years','2 years','3 years','4 years and above'])

    # ==============================
    # CHILD CHARACTERISTICS
    # ==============================
    st.markdown("## 👶 Child Characteristics")
    col3, col4 = st.columns(2)

    with col3:
        childgender = st.selectbox("Child Gender",
            ['male','female'])
        childage = st.selectbox("Child Age",
            ['0-11 months','12-23 months','24-35 months','36-47 months','48-59 months'])
        twin = st.selectbox("Birth Type",
            ['sinltone','multiple'])
        birthsize = st.selectbox("Birth Size",
            ['very large','larger than average','average','smaller than average','very small'])

    with col4:
        diarrhea = st.selectbox("Recent Diarrhea",
            ['no','yes'])
        fever = st.selectbox("Recent Fever",
            ['no','yes'])
        ari = st.selectbox("Acute Respiratory Infection",
            ['no','yes'])
        breastfed = st.selectbox("Breastfeeding Status",
            ['never breastfed','ever breastfed, not currently breastfee',' still breastfeeding'])
        bottlefeeding = st.selectbox("Bottle Feeding",
            ['no','yes'])
        vitamina = st.selectbox("Vitamin A Supplement",
            ['no','yes'])
        dietdivers = st.selectbox("Diet Diversity",
            ['adequate','inadequate'])

    # ==============================
    # HOUSEHOLD CHARACTERISTICS
    # ==============================
    st.markdown("## 🏠 Household Characteristics")
    col5, col6 = st.columns(2)

    with col5:
        placeresid = st.selectbox("Place of Residence",
            ['urban','rural'])
        famsize = st.selectbox("Family Size",
            ['1-5','6 and more'])
        num5child = st.selectbox("Under-5 Children",
            ['less than 2','3 and more'])
        sexofhead = st.selectbox("Sex of Household Head",
            ['male','female'])

    with col6:
        latrine = st.selectbox("Latrine Available",
            ['no','yes'])
        disposalofchild = st.selectbox("Child Stool Disposal",
            ['appropraite','inappropraite'])
        watersource = st.selectbox("Water Source",
            ['unimproved','improved'])
        country = st.selectbox("Country",
            ['brundi','comoros','ethiopia','kenya','madagascar','malawi',
             'mozambique','rwanda','tanzania','uganda','zambia','zimbabwe'])

    submitted = st.form_submit_button("🔍 Predict Risk")

# =====================================================
# PREDICTION SECTION
# =====================================================
if submitted:

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
        st.subheader("📊 Prediction Result")

        colA, colB = st.columns(2)

        with colA:
            st.metric(
                label="Predicted Probability",
                value=f"{probability*100:.1f} %"
            )

            if probability >= threshold:
                st.error("⚠️ High Risk of Acute Malnutrition")
            else:
                st.success("✅ Low Risk of Acute Malnutrition")

        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Risk Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, threshold*100], 'color': "lightgreen"},
                        {'range': [threshold*100, 100], 'color': "salmon"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =====================================================
# MODEL PERFORMANCE
# =====================================================
st.markdown("---")
st.markdown("## 📈 Model Performance")

st.write("""
- **Area Under the Curve (AUC):** 74.6% (95% CI: 73.0–76.2)  
- **Accuracy:** 71.2%  
- **Precision:** 13.1%  
- **Recall (Sensitivity):** 66.2%  
- **Specificity:** 71.5%  
- **F1-Score:** 0.218  
""")