import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import gdown
import pandas as pd
import base64
import io
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

st.set_page_config(page_title="מערכת לזיהוי מחלות צמחים 🌾", page_icon="🌾", layout="wide")

# ─────────────────────────────────────────
# MongoDB Connection
# ─────────────────────────────────────────

@st.cache_resource
def get_db():
    uri = st.secrets["MONGODB_URI"]
    client = MongoClient(uri)
    try:
        client.admin.command("ping")
    except ConnectionFailure:
        st.error("❌ שגיאה בחיבור למערכת. אנא פנה למנהל המערכת.")
        st.stop()
    return client["wheat_disease_db"]

db            = get_db()
plants_col    = db["plants"]
diagnoses_col = db["diagnoses"]

# ─────────────────────────────────────────
# Page Navigation
# ─────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "home"
if "current_plant_idx" not in st.session_state:
    st.session_state.current_plant_idx = 0

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
.stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, label,
[data-testid="stWidgetLabel"] {
    text-align: right !important; direction: rtl !important;
}
.stButton>button, .stSelectbox, .stTextArea {
    direction: rtl !important; text-align: right !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    text-align: right !important; direction: rtl !important;
}
[data-testid="stDataFrame"] { direction: rtl !important; }
.custom-card {
    background: #ffffff; padding: 24px; border-radius: 12px;
    border-right: 6px solid #2e7d32; margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}
.home-box {
    background: #ffffff; padding: 40px 30px; border-radius: 16px;
    border: 1px solid #eaeaea; text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.03);
    transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
    height: 100%;
}
.home-box:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(46,125,50,.12);
    border-color: #2e7d32;
}
.main-title {
    font-size: 3rem !important; font-weight: 800 !important;
    background: linear-gradient(45deg, #2e7d32, #1565c0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center !important; margin-bottom: 10px !important;
}
.subtitle {
    font-size: 1.25rem !important; color: #666 !important;
    text-align: center !important; margin-bottom: 40px !important;
}
.db-badge {
    display: inline-block; background: #e8f5e9; color: #2e7d32;
    border-radius: 20px; padding: 4px 14px; font-size: 0.85rem;
    font-weight: 600; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────

FILE_ID = '161ysydHCyvLOoVWkwWqJT5RpcMn_0rVu'
MODEL_PATH = 'best_resnet18_wheat.pt'
CONFIDENCE_THRESHOLD = 0.25

DISEASE_INFO = {
    "BlackPoint": {
        "heb": "חוד שחור (Black Point)",
        "desc": "מחלה הנגרמת על ידי קומפלקס פטריות בשלבי הבשלת הגרעין. מתאפיינת בהשחרה של קצה הגרעין, ומתפתחת בעיקר בעקבות לחות גבוהה.",
        "tip": "מומלץ להפחית את משטר ההשקיה בשלבי ההבשלה. שימוש בזרעים נקיים ומחוטאים בעונה הבאה."
    },
    "FusariumFootRot": {
        "heb": "ריקבון בסיס הקנה (Fusarium)",
        "desc": "מחלה פטרייתית קרקעית התוקפת את השורשים ובסיס הקנה. חוסמת צינורות הובלה ומביאה לנבילה וריקבון.",
        "tip": "יש ליישם מחזור זרעים קפדני עם גידולים שאינם דגניים למשך שנתיים. להימנע מהשקיית יתר."
    },
    "HealthyLeaf": {
        "heb": "עלה בריא (Healthy)",
        "desc": "העלה מציג חיוניות גבוהה, צבע ירוק אחיד ושטח פנים נקי. הפוטוסינתזה מתנהלת בצורה אופטימלית.",
        "tip": "מצב מצוין! יש להמשיך במשטר הטיפוח הנוכחי ולשמור על ניטור שבועי."
    },
    "LeafBlight": {
        "heb": "קמלת עלים (Leaf Blight)",
        "desc": "מחלה פטרייתית המתבטאת בכתמים חומים-אפרפרים על העלים. מפחיתה דרסטית את כושר הפוטוסינתזה.",
        "tip": "מומלץ לרסס בקוטלי פטריות עם זיהוי הסימנים הראשונים. יש להשמיד שאריות צמחים נגועות."
    },
    "WheatBlast": {
        "heb": "פיריקורליית החיטה (Wheat Blast)",
        "desc": "מחלה פטרייתית הרסנית — השיבולת הופכת ללבנה ויבשה תוך ימים ומונעת פיתוח גרגרים.",
        "tip": "מצב חירום חקלאי. יש לבודד את האזור הנגוע מיד ולרסס בקוטלי פטריות סיסטמיים בדחיפות."
    }
}

# ─────────────────────────────────────────
# DB Helper Functions
# ─────────────────────────────────────────

def get_all_plants():
    docs = list(plants_col.find({}, {"_id": 0}).sort("id", 1))
    return docs

def upsert_plants_from_df(df: pd.DataFrame):
    plants_col.delete_many({})
    records = df.to_dict(orient="records")
    if records:
        plants_col.insert_many(records)

def save_diagnosis(plant_id, plant_name, image, class_name, diagnosis_heb, notes):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    doc = {
        "plant_id":   plant_id,
        "plant_name": plant_name,
        "class_name": class_name,
        "diagnosis":  diagnosis_heb,
        "notes":      notes,
        "image_b64":  img_b64,
        "timestamp":  datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "created_at": datetime.now(timezone.utc),
    }
    diagnoses_col.insert_one(doc)

def load_diagnoses(plant_id):
    return list(diagnoses_col.find({"plant_id": plant_id}, {"_id": 0}).sort("created_at", 1))

def delete_diagnosis(plant_id, timestamp):
    diagnoses_col.delete_one({"plant_id": plant_id, "timestamp": timestamp})

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

@st.cache_resource
def load_wheat_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("טוען מודל בינה מלאכותית..."):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        lbl = checkpoint.get("classes", list(DISEASE_INFO.keys()))
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(lbl))
        m.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        m.eval()
        return m, lbl
    except Exception as e:
        st.error(f"שגיאה בטעינת המודל: {e}")
        return None, None

model, labels = load_wheat_model()

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_model(image):
    if model is None:
        return None, None
    with torch.no_grad():
        out  = model(transform(image).unsqueeze(0))
        prob = torch.nn.functional.softmax(out[0], dim=0)
        conf, pred = torch.max(prob, 0)
    if conf.item() < CONFIDENCE_THRESHOLD:
        return None, conf.item()
    return labels[pred.item()], conf.item()

# ─────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────

if st.session_state.page == "home":
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>מערכת מתקדמת לזיהוי מחלות צמחים</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>מבצעים: נבו הלר ומתן אדר | מנחה: אסי ברק</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'><span class='db-badge'>🟢 מערכת פעילה ומוכנה לשימוש</span></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("""<div class="home-box">
          <span style='font-size:3.5rem'>📸</span>
          <h3 style='color:#1565c0;margin-top:15px'>אבחון חזותי מהיר</h3>
          <p style='color:#666;font-size:1rem;line-height:1.6'>
            בדיקה מיידית של עלה — מעלים תמונה ומקבלים אבחון והנחיות טיפול.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("הפעל אבחון מהיר 🚀", use_container_width=True, key="btn_single", type="primary"):
            st.session_state.page = "single_diagnosis"; st.rerun()

    with col2:
        st.markdown("""<div class="home-box">
          <span style='font-size:3.5rem'>📊</span>
          <h3 style='color:#2e7d32;margin-top:15px'>ניהול ומעקב ניסוי</h3>
          <p style='color:#666;font-size:1rem;line-height:1.6'>
            ניהול צמחי הניסוי, תיעוד אבחונים — נתוני הצמחים, האבחונים והתמונות נשמרים ומסונכרנים בין כל המשתמשים.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("פתח מערכת ניסוי 🔬", use_container_width=True, key="btn_exp", type="primary"):
            st.session_state.page = "experiment_management"; st.rerun()

    with col3:
        st.markdown("""<div class="home-box">
          <span style='font-size:3.5rem'>⚙️</span>
          <h3 style='color:#6a1b9a;margin-top:15px'>ניהול נתונים</h3>
          <p style='color:#666;font-size:1rem;line-height:1.6'>
            עדכון וניהול נתוני הניסוי, ייצוא דוחות ותחזוקת המערכת.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ניהול נתונים ⚙️", use_container_width=True, key="btn_admin", type="primary"):
            st.session_state.page = "data_management"; st.rerun()

# ─────────────────────────────────────────
# QUICK DIAGNOSIS
# ─────────────────────────────────────────

elif st.session_state.page == "single_diagnosis":
    if st.button("🔙 חזרה לדף הבית", key="back1"):
        st.session_state.page = "home"; st.rerun()
    st.markdown("<h2 style='color:#1565c0'>📸 אבחון חזותי מהיר</h2>", unsafe_allow_html=True)
    st.write("העלאת תמונת עלה לזיהוי מחלה ידני — ללא שיוך לצמח ספציפי בניסוי")
    st.divider()

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("### 📥 הזנת תמונה")
            method = st.radio("בחר דרך:", ("צילום במצלמה 📸", "העלאת קובץ 📁"), key="sm")
            img_file = (st.camera_input("צלם עלה", key="sc")
                        if "מצלמה" in method
                        else st.file_uploader("בחר קובץ", type=["jpg","png","jpeg"], key="su"))
    with c2:
        if img_file:
            image = Image.open(img_file).convert("RGB")
            st.image(image, caption="התמונה שהוזנה", use_container_width=True)
            class_name, conf = run_model(image)
            if class_name is None:
                st.error("⚠️ לא זוהה עלה רלוונטי. נסה לצלם מקרוב ובתאורה טובה.")
            else:
                info = DISEASE_INFO.get(class_name, {"heb": class_name, "desc": "", "tip": ""})
                st.markdown(f"### 🎯 אבחון: <span style='color:#1565c0'><b>{info['heb']}</b></span>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="custom-card" style="border-right-color:#1565c0;background:#f1f8ff">
                  <h4 style="color:#1565c0;margin-top:0">🔬 תיאור המחלה:</h4>
                  <p style="line-height:1.6">{info['desc']}</p>
                  <hr style="border:0;border-top:1px solid #d0e2ff;margin:16px 0">
                  <h4 style="color:#2e7d32;margin-top:0">💡 המלצות טיפול:</h4>
                  <p style="line-height:1.6">{info['tip']}</p>
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# EXPERIMENT MANAGEMENT
# ─────────────────────────────────────────

elif st.session_state.page == "experiment_management":
    if st.button("🔙 חזרה לדף הבית", key="back2"):
        st.session_state.page = "home"; st.rerun()
    st.markdown("<h2 style='color:#2e7d32'>📊 מערכת ניהול ניסוי החיטה</h2>", unsafe_allow_html=True)
    st.divider()

    with st.spinner("טוען נתונים..."):
        plants = get_all_plants()

    if not plants:
        st.warning("⚠️ טרם נטענו נתוני ניסוי. יש לעבור לדף ניהול הנתונים ולהעלות קובץ נתונים.")
        st.stop()

    labels_list = [f"ID: {p['id']} | שם: {p['name']}" for p in plants]
    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("➡️ הקודם", use_container_width=True):
            if st.session_state.current_plant_idx > 0:
                st.session_state.current_plant_idx -= 1; st.rerun()
    with nav3:
        if st.button("הבא ⬅️", use_container_width=True):
            if st.session_state.current_plant_idx < len(labels_list) - 1:
                st.session_state.current_plant_idx += 1; st.rerun()
    with nav2:
        sel = st.selectbox("בחר צמח:", labels_list,
                           index=st.session_state.current_plant_idx, key="psel")
        st.session_state.current_plant_idx = labels_list.index(sel)

    plant      = plants[st.session_state.current_plant_idx]
    plant_id   = int(plant["id"])
    plant_name = str(plant["name"])

    with st.spinner("טוען היסטוריה..."):
        history = load_diagnoses(plant_id)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔢 מזהה", str(plant_id))
        c2.metric("🌱 שם", plant_name)
        c3.metric("🧪 טיפול", str(plant.get("#Treatment", "—")))
        c4.metric("📉 עקה", f"{float(plant.get('stressDegree', 0)):.3f}")

    treatment = plant.get("#Treatment", "")
    stress    = float(plant.get("stressDegree", 0))
    if treatment == "Drought" and stress > 0.15:
        color, icon, msg, bg = "#f44336", "⚠️", f"עקת יובש משמעותית (מדד: {stress:.3f})", "#fdf2f2"
    elif treatment == "Drought":
        color, icon, msg, bg = "#ff9800", "🔸", f"עקת יובש מתונה (מדד: {stress:.3f})", "#fffaf2"
    else:
        color, icon, msg, bg = "#2e7d32", "✅", "תקין — קבוצת ביקורת (Control)", "#f2fbf2"
    st.markdown(f"""
    <div class="custom-card" style="border-right-color:{color};background:{bg}">
      {icon} <b>סטטוס פנוטיפי:</b> {msg}
    </div>""", unsafe_allow_html=True)

    if history:
        latest = history[-1]
        c_name = latest.get("class_name", "")
        st.markdown(f"""
        <div class="custom-card" style="border-right-color:#2196f3;background:#e3f2fd">
          🔍 <b>אבחון אחרון ({latest['timestamp']}):</b><br><br>
          <b>🩺 אבחון:</b> {latest['diagnosis']}<br>
          <b>📝 הערות:</b> {latest['notes']}
        </div>""", unsafe_allow_html=True)
        if c_name in DISEASE_INFO and c_name != "HealthyLeaf":
            with st.expander(f"🔬 פירוט מורחב — {DISEASE_INFO[c_name]['heb']}"):
                st.markdown(f"**תיאור:** {DISEASE_INFO[c_name]['desc']}")
                st.markdown(f"**💡 המלצות:** {DISEASE_INFO[c_name]['tip']}")

    st.divider()
    st.subheader("📋 נתוני הצמח המלאים")
    st.dataframe(pd.DataFrame([plant]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📸 הוספת אבחון חדש")
    with st.container(border=True):
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            method = st.radio("דרך הזנה:", ("מצלמה 📸", "קובץ 📁"), key="em")
            img_file = (st.camera_input("צלם עלה", key="ec")
                        if "מצלמה" in method
                        else st.file_uploader("בחר קובץ", type=["jpg","png","jpeg"], key="eu"))
        with d2:
            notes = st.text_area("✍️ תיאור מצב הצמח:", placeholder="הקלד הערות מהחממה...", height=150)

        if img_file:
            image = Image.open(img_file).convert("RGB")
            class_name, _ = run_model(image)
            if class_name is None:
                auto_diag  = "לא זוהה עלה רלוונטי"
                class_name = "Unknown"
            else:
                auto_diag = DISEASE_INFO.get(class_name, {"heb": class_name})["heb"]
            st.markdown(f"**🔍 תוצאת ניתוח:** {auto_diag}")
            if st.button("💾 שמור אבחון", use_container_width=True, type="primary"):
                with st.spinner("שומר אבחון..."):
                    save_diagnosis(plant_id, plant_name, image, class_name, auto_diag,
                                   notes if notes else "לא הוכנס פירוט")
                st.success("✅ האבחון נשמר בהצלחה!")
                st.rerun()

    st.divider()
    st.subheader(f"🗄️ היסטוריית אבחונים — {plant_name}")
    if history:
        for rec in reversed(history):
            with st.container(border=True):
                h1, h2 = st.columns([1, 3], gap="medium")
                with h1:
                    try:
                        img_bytes = base64.b64decode(rec["image_b64"])
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.info("תמונה לא זמינה")
                with h2:
                    st.markdown(f"### 📅 `{rec['timestamp']}`")
                    st.markdown(f"**🔬 אבחון:** {rec['diagnosis']}")
                    st.markdown(f"**📝 הערות:** {rec['notes']}")
                    c_ref = rec.get("class_name", "")
                    if c_ref in DISEASE_INFO and c_ref != "HealthyLeaf":
                        st.info(f"💡 {DISEASE_INFO[c_ref]['tip']}")
                    if st.button("🗑️ מחק רשומה זו", key=f"del_{rec['timestamp']}"):
                        delete_diagnosis(plant_id, rec["timestamp"])
                        st.success("הרשומה נמחקה.")
                        st.rerun()
    else:
        st.info("אין עדיין אבחונים לצמח זה.")

# ─────────────────────────────────────────
# DATA MANAGEMENT
# ─────────────────────────────────────────

elif st.session_state.page == "data_management":
    if st.button("🔙 חזרה לדף הבית", key="back3"):
        st.session_state.page = "home"; st.rerun()
    st.markdown("<h2 style='color:#6a1b9a'>⚙️ ניהול נתוני הניסוי</h2>", unsafe_allow_html=True)
    st.divider()

    st.subheader("📥 עדכון נתוני הניסוי")
    with st.container(border=True):
        st.write("העלאת קובץ נתונים מעודכן. לאחר האישור הנתונים יעודכנו בכל המערכת.")
        csv_file = st.file_uploader("בחר קובץ CSV", type=["csv"], key="csv_upload")
        if csv_file:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"סה\"כ: **{len(df)}** שורות, **{len(df.columns)}** עמודות")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ אשר ועדכן נתונים", type="primary", use_container_width=True):
                    with st.spinner("מעדכן נתונים..."):
                        upsert_plants_from_df(df)
                    st.success(f"✅ הנתונים עודכנו בהצלחה!")
                    st.balloons()
            with col_b:
                if st.button("❌ ביטול", use_container_width=True):
                    st.rerun()

    st.divider()
    st.subheader("📊 סטטוס הניסוי")
    with st.container(border=True):
        with st.spinner("טוען..."):
            plant_count    = plants_col.count_documents({})
            diagnosis_count = diagnoses_col.count_documents({})
            last_diag = diagnoses_col.find_one({}, {"timestamp": 1, "_id": 0},
                                               sort=[("created_at", -1)])
            last_ts = last_diag["timestamp"] if last_diag else "אין"
        s1, s2, s3 = st.columns(3)
        s1.metric("🌱 צמחים בניסוי", plant_count)
        s2.metric("🔬 סה\"כ אבחונים", diagnosis_count)
        s3.metric("🕐 אבחון אחרון", last_ts)

    st.divider()
    st.subheader("📤 ייצוא נתונים")
    with st.container(border=True):
        ec1, ec2 = st.columns(2)
        with ec1:
            if st.button("📥 ייצא צמחים כ-CSV", use_container_width=True):
                plants_exp = get_all_plants()
                if plants_exp:
                    csv_data = pd.DataFrame(plants_exp).to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ הורד plants.csv", csv_data,
                                       "plants.csv", "text/csv", use_container_width=True)
                else:
                    st.warning("אין נתונים.")
        with ec2:
            if st.button("📥 ייצא אבחונים כ-CSV", use_container_width=True):
                docs = list(diagnoses_col.find({}, {"_id": 0, "image_b64": 0})
                            .sort("created_at", 1))
                if docs:
                    csv_diag = pd.DataFrame(docs).to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ הורד diagnoses.csv", csv_diag,
                                       "diagnoses.csv", "text/csv", use_container_width=True)
                else:
                    st.warning("אין אבחונים.")

    st.divider()
    with st.expander("🔴 אזור מסוכן — מחיקת נתונים"):
        st.warning("פעולות אלו בלתי הפיכות!")
        d1, d2 = st.columns(2)
        with d1:
            if st.button("🗑️ מחק את כל האבחונים", use_container_width=True):
                diagnoses_col.delete_many({})
                st.success("כל האבחונים נמחקו.")
                st.rerun()
        with d2:
            if st.button("🗑️ מחק את כל נתוני הצמחים", use_container_width=True):
                plants_col.delete_many({})
                st.success("כל נתוני הצמחים נמחקו.")
                st.rerun()
