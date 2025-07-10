import gradio as gr
import pandas as pd
import joblib
import os
import logging
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Sentence Transformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load Models and Data ---
try:
    model = joblib.load("Optuna_1_xgb_model_credit_score.pkl")
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

try:
    mortgages = pd.read_csv("MORTGAGE3.csv")
    mortgages['embedding'] = mortgages['product.description'].apply(
        lambda x: embedding_model.encode(str(x), convert_to_tensor=True)
    )
    logging.info("Mortgage data loaded successfully")
except Exception as e:
    logging.error(f"Error loading mortgage data: {e}")
    raise

# City density mapping
city_to_density = {
    "Алматы": 0.025,
    "Нур-Султан": 0.020,
    "Шымкент": 0.018,
    "Караганда": 0.010,
    "Актобе": 0.008,
    "Другое": 0.005
}

# --- Categorical Encoding Mappings ---
CATEGORY_MAPPINGS = {
    "CODE_GENDER": {"M": 0, "F": 1},
    "FLAG_OWN_REALTY": {"Y": 1, "N": 0},
    "NAME_INCOME_TYPE": {
        "Working": 0,
        "State servant": 1,
        "Commercial associate": 2,
        "Student": 3
    },
    "NAME_FAMILY_STATUS": {
        "Married": 0,
        "Single / not married": 1,
        "Civil marriage": 2,
        "Separated": 3,
        "Widow": 4
    },
    "NAME_EDUCATION_TYPE": {
        "Secondary / secondary special": 0,
        "Higher education": 1,
        "Incomplete higher": 2,
        "Lower secondary": 3,
        "Academic degree": 4
    },
    "CREDIT_ACTIVE": {
        "Active": 0,
        "Closed": 1,
        "Sold": 2,
        "Bad debt": 3
    },
    "CREDIT_TYPE": {
        "Mortgage": 0
    },
    "ORG_TYPE_GROUPED": {
        "Частная компания": 0,
        "ИП / Самозанятый": 1,
        "Торговая компания": 2,
        "Производственная компания": 3,
        "Медицинское учреждение": 4,
        "Государственная организация": 5,
        "Транспортная компания": 6,
        "Госслужбы (полиция, армия и т.д.)": 7,
        "Школа": 8,
        "Детский сад": 9,
        "Строительство": 10,
        "Банк": 11,
        "Охрана": 12,
        "ЖКХ": 13,
        "Почта": 14,
        "Сельское хозяйство": 15,
        "Ресторан / Кафе": 16,
        "Сфера услуг": 17,
        "Университет / колледж": 18,
        "Энергетика": 19,
        "Телеком": 20,
        "Гостиница": 21,
        "Страхование": 22,
        "Маркетинг / Реклама": 23,
        "Искусство / Культура": 24,
        "Риелтор": 25,
        "Юрист": 26,
        "Уборка": 27,
        "Религия": 28,
        "Другое": 29
    },
    "OCCUPATION_TYPE_RU": {
        "Разнорабочие": 0,
        "Основной персонал": 1,
        "Торговый персонал": 2,
        "Менеджеры": 3,
        "Водители": 4,
        "Высококвалифицированные специалисты": 5,
        "Бухгалтеры": 6,
        "Медицинский персонал": 7,
        "Повар": 8,
        "Охрана": 9,
        "Уборка": 10,
        "Частные услуги": 11,
        "Низкоквалифицированные рабочие": 12,
        "Секретари": 13,
        "Официанты / бармены": 14,
        "Агенты по недвижимости": 15,
        "HR": 16,
        "IT-специалисты": 17
    }
}

def preprocess_inputs(input_dict):
    """Convert categorical inputs to numerical values"""
    processed = input_dict.copy()
    for col, mapping in CATEGORY_MAPPINGS.items():
        if col in processed:
            processed[col] = mapping.get(processed[col], -1)  # -1 for unknown categories
    return processed

def predict_credit_score(
    CODE_GENDER,
    FLAG_OWN_REALTY,
    NAME_INCOME_TYPE,
    NAME_FAMILY_STATUS,
    NAME_EDUCATION_TYPE,
    CREDIT_ACTIVE,
    CREDIT_TYPE,
    ORG_TYPE_GROUPED,
    OCCUPATION_TYPE_RU,
    CNT_CHILDREN,
    CNT_FAM_MEMBERS,
    CNT_CREDIT_PROLONG,
    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    AMT_ANNUITY,
    AGE_YEARS,
    EMPLOYMENT_YEARS,
    CITY,
    LOAN_DURATION,
    ANNUITY_INCOME_RATIO
):
    try:
        # --- Preprocess Inputs ---
        input_dict = {
            "CODE_GENDER": CODE_GENDER,
            "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
            "NAME_INCOME_TYPE": NAME_INCOME_TYPE,
            "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
            "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
            "CREDIT_ACTIVE": CREDIT_ACTIVE,
            "CREDIT_TYPE": CREDIT_TYPE,
            "ORG_TYPE_GROUPED": ORG_TYPE_GROUPED,
            "OCCUPATION_TYPE_RU": OCCUPATION_TYPE_RU,
            "CNT_CHILDREN": CNT_CHILDREN,
            "CNT_FAM_MEMBERS": CNT_FAM_MEMBERS,
            "CNT_CREDIT_PROLONG": CNT_CREDIT_PROLONG,
            "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
            "AMT_CREDIT": AMT_CREDIT,
            "AMT_ANNUITY": AMT_ANNUITY,
            "DAYS_BIRTH": -int(AGE_YEARS * 365),
            "DAYS_EMPLOYED": -int(EMPLOYMENT_YEARS * 365),
            "REGION_POPULATION_RELATIVE": city_to_density.get(CITY, 0.005),
            "LOAN_DURATION": LOAN_DURATION,
            "ANNUITY_INCOME_RATIO": ANNUITY_INCOME_RATIO
        }
        
        # Apply categorical encoding
        processed_input = preprocess_inputs(input_dict)
        
       # Create DataFrame
        client_df = pd.DataFrame([processed_input])
        
        # Ensure numeric dtypes (updated without errors parameter)
        client_df = client_df.apply(pd.to_numeric)
        
        # Reorder columns to match training data
        CORRECT_FEATURE_ORDER = [
            'CODE_GENDER', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
            'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
            'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS', 'CREDIT_ACTIVE', 'CREDIT_TYPE',
            'CNT_CREDIT_PROLONG', 'LOAN_DURATION', 'ANNUITY_INCOME_RATIO',
            'ORG_TYPE_GROUPED', 'OCCUPATION_TYPE_RU'
        ]
        client_df = client_df[CORRECT_FEATURE_ORDER]
        
        # Debug output
        logging.info(f"Processed DataFrame columns:\n{client_df.columns.tolist()}")
        
                # Прогноз модели
        approval_prob = model.predict_proba(client_df)[0][1]  # вероятность одобрения
        credit_score = int(300 + approval_prob * 700)  # шкала от 300 до 1000

        # --- Поиск подходящих ипотек ---
        client_profile = f"{NAME_INCOME_TYPE}, {NAME_FAMILY_STATUS}, {NAME_EDUCATION_TYPE}, {OCCUPATION_TYPE_RU}"
        client_embedding = embedding_model.encode(client_profile, convert_to_tensor=True)

        mortgages["similarity"] = mortgages["embedding"].apply(
            lambda x: util.cos_sim(x, client_embedding).item()
        )

        top_matches = (
            mortgages.sort_values("similarity", ascending=False)
            .groupby('product.name')  # исключим дубликаты
            .first()
            .head(3)
            .reset_index()
        )

        # Форматирование рекомендаций
        if top_matches.empty:
            recs = "⚠️ Нет подходящих ипотек"
        else:
            formatted_recs = []
            for _, row in top_matches.iterrows():
                name = row['product.name']
                desc = str(row['product.description'])[:100].strip(" .") + "..."
                sim_score = row['similarity'] * 100
                url = row.get("product.company.website", "#")
                formatted_recs.append(f"**[{name}]({url})**\nСовпадение: {sim_score:.1f}%\n_{desc}_")
            recs = "\n\n".join(formatted_recs)

        return (
            f"💳 **Кредитный скоринг:** {credit_score}\n"
            f"📈 **Вероятность одобрения:** {approval_prob * 100:.1f}%\n\n"
            f"🏠 **Рекомендуемые ипотечные продукты:**\n\n{recs}"
        )
    
    
    except Exception as e:
        logging.error(f"Prediction error: {e}", exc_info=True)
        return f"⚠️ Ошибка: {str(e)}"

# --- Gradio Interface ---
inputs = [
    gr.Radio(["M", "F"], label="Пол (CODE_GENDER)"),
    gr.Radio(["Y", "N"], label="Есть ли недвижимость? (FLAG_OWN_REALTY)"),
    gr.Radio(
        ["Working", "State servant", "Commercial associate", "Student"], 
        label="Тип занятости (NAME_INCOME_TYPE)"
    ),
    gr.Radio(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"], 
        label="Семейное положение (NAME_FAMILY_STATUS)"
    ),
    gr.Radio(
        ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], 
        label="Уровень образования (NAME_EDUCATION_TYPE)"
    ),
    gr.Radio(
        ["Active", "Closed", "Sold", "Bad debt"], 
        label="Тип кредитной активности (CREDIT_ACTIVE)"
    ),
    gr.Radio(["Mortgage"], label="Тип кредита (CREDIT_TYPE)"),
    gr.Dropdown(
        list(CATEGORY_MAPPINGS["ORG_TYPE_GROUPED"].keys()), 
        label="Тип организации (ORG_TYPE_GROUPED)"
    ),
    gr.Dropdown(
        list(CATEGORY_MAPPINGS["OCCUPATION_TYPE_RU"].keys()), 
        label="Профессия (OCCUPATION_TYPE_RU)"
    ),
    gr.Number(label="Количество детей (CNT_CHILDREN)", value=0),
    gr.Number(label="Размер семьи (CNT_FAM_MEMBERS)", value=1),
    gr.Number(label="Продлевали ли вы кредит? (CNT_CREDIT_PROLONG)", value=0),
    gr.Number(label="Общий доход в месяц (AMT_INCOME_TOTAL)", value=100000),
    gr.Number(label="Сумма кредита (AMT_CREDIT)", value=1000000),
    gr.Number(label="Аннуитет (AMT_ANNUITY)", value=50000),
    gr.Number(label="Возраст (лет)", value=30),
    gr.Number(label="Стаж работы (лет)", value=5),
    gr.Dropdown(list(city_to_density.keys()), label="Город", value="Алматы"),
    gr.Number(label="Срок кредита (в месяцах)", value=240),
    gr.Number(label="Соотношение аннуитета к доходу", value=0.5)
]

demo = gr.Interface(
    fn=predict_credit_score,
    inputs=inputs,
    outputs=gr.Textbox(label="Результат", lines=10),
    title="💳 Анкета клиента для скоринга",
    description="Пожалуйста, заполните анкету для оценки кредитоспособности.",
    flagging_options=None
)

if __name__ == "__main__":
    demo.launch(share=True)