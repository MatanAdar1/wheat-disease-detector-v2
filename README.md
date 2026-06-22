# 🌾 Smart Precision Agriculture System — גרסה משודרגת

מערכת לזיהוי מחלות חיטה עם MongoDB Atlas כשכבת נתונים קבועה.

---

## מה השתנה לעומת הגרסה הקודמת?

| תכונה | גרסה ישנה | גרסה חדשה |
|--------|-----------|-----------|
| שמירת אבחונים | קבצים על שרת Streamlit (נמחק ב-restart) | MongoDB Atlas (קבוע לצמיתות) |
| עדכון CSV | Push ל-GitHub + רענון ידני | כפתור בממשק — מיידי |
| תמונות | קבצים מקומיים | Base64 ב-MongoDB |
| ייצוא נתונים | אין | CSV בלחיצת כפתור |
| סנכרון בין מחשבים | חלקי | מלא — DB משותף |

---

## הגדרה ראשונית

### 1. MongoDB Atlas

1. היכנס ל-[MongoDB Atlas](https://cloud.mongodb.com)
2. צור Cluster חינמי (M0)
3. צור משתמש ב-Database Access
4. הוסף `0.0.0.0/0` ב-Network Access (לפיתוח)
5. העתק את ה-Connection String

### 2. הגדרת Secrets

**לפיתוח מקומי** — צור את הקובץ `.streamlit/secrets.toml`:

```toml
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
```

**ב-Streamlit Cloud** — עבור ל-App Settings → Secrets והדבק:

```toml
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
```

### 3. התקנה מקומית

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## מבנה מסד הנתונים (MongoDB)

```
wheat_disease_db/
├── plants/          # נתוני צמחי הניסוי (מיובא מ-CSV)
│   ├── id
│   ├── name
│   ├── #Treatment
│   ├── stressDegree
│   └── ... (כל עמודות ה-CSV)
│
└── diagnoses/       # היסטוריית אבחונים
    ├── plant_id
    ├── plant_name
    ├── class_name
    ├── diagnosis      (עברית)
    ├── notes
    ├── image_b64      (תמונה מקודדת)
    ├── timestamp      (תצוגה: DD/MM/YYYY HH:MM:SS)
    └── created_at     (UTC — לסידור כרונולוגי)
```

---

## מבנה האפליקציה

```
app.py                        ← קוד ראשי
requirements.txt              ← תלויות Python
.streamlit/
└── secrets.toml              ← MONGODB_URI (לא ב-Git!)
.gitignore                    ← כולל .streamlit/secrets.toml
```

---

## .gitignore מומלץ

```
.streamlit/secrets.toml
*.pt
__pycache__/
*.pyc
saved_plant_history/
```

---

## דפי האפליקציה

| דף | תיאור |
|----|-------|
| 🏠 דף הבית | ניווט ראשי + סטטוס חיבור ל-MongoDB |
| 📸 אבחון מהיר | העלאת תמונה → אבחון AI מיידי |
| 📊 ניהול ניסוי | דפדוף בצמחים, תיעוד אבחונים, היסטוריה |
| ⚙️ ניהול נתונים | העלאת CSV, סטטיסטיקות DB, ייצוא, מחיקה |
