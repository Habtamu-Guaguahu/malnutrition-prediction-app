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
st.sidebar.title("About This Tool")
st.sidebar.markdown("""
**Model:** Random Forest  
**Feature Selection:** Consensus (≥2 methods)  
**Threshold Optimization:** Youden Index  

Developed for screening of acute malnutrition risk among  
children under five in Sub-Saharan Africa.
""")

st.sidebar.markdown("---")
st.sidebar.info("For research and public health screening use only.")

# =====================================================
# HEADER
# =====================================================
st.title("🌍 Acute Malnutrition Risk Prediction")
st.markdown("""
This tool predicts the probability of **acute malnutrition (wasting)**  
using a trained Random Forest machine learning model.
""")
st.markdown("---")

# =====================================================
# INPUT FORM
# =====================================================
with st.form("prediction_form"):

    st.markdown("### 👩 Maternal & Household Characteristics")
    col1, col2, col3 = st.columns(3)

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

    st.markdown("### 👶 Child Characteristics")
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

    st.markdown("### 🏥 Health & Nutrition Factors")
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

        # Probability Metric
        with colA:
            st.metric(
                label="Predicted Probability",
                value=f"{probability*100:.1f} %"
            )

            st.write(f"Decision Threshold (Youden Index): {threshold:.3f}")

            if probability >= threshold:
                st.error("⚠️ High Risk of Acute Malnutrition")
            else:
                st.success("✅ Low Risk of Acute Malnutrition")

        # Gauge Chart
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
st.markdown("### 📈 Model Information")
st.write("Random Forest Classifier")
st.write("Feature Selection: Consensus (≥2 methods)")
st.write("Threshold Optimization: Youden Index")

# =====================================================
# DISCLAIMER
# =====================================================
st.markdown("---")
st.caption(
    "This tool is developed for research and screening purposes. "
    "It does not replace clinical judgment or formal nutritional assessment."
)