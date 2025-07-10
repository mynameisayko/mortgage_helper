from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import logging
import random
import openai

app = Flask(__name__)

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load mortgage data
try:
    mortgages = pd.read_csv("MORTGAGE3.csv")
    logger.info("Mortgage data loaded successfully")
except Exception as e:
    logger.error(f"Error loading mortgage data: {e}")
    mortgages = None

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

openai.api_key = "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"

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_inputs(input_dict):
    """Convert categorical inputs to numerical values"""
    processed = input_dict.copy()
    for col, mapping in CATEGORY_MAPPINGS.items():
        if col in processed:
            processed[col] = mapping.get(processed[col], -1)  # -1 for unknown categories
    return processed

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        logger.info(f"Received form data: {form_data}")
        
        # Extract data from form
        CODE_GENDER = form_data.get('CODE_GENDER', 'M')
        FLAG_OWN_REALTY = form_data.get('FLAG_OWN_REALTY', 'N')
        NAME_INCOME_TYPE = form_data.get('NAME_INCOME_TYPE', 'Working')
        NAME_FAMILY_STATUS = form_data.get('NAME_FAMILY_STATUS', 'Married')
        NAME_EDUCATION_TYPE = form_data.get('NAME_EDUCATION_TYPE', 'Secondary / secondary special')
        CREDIT_ACTIVE = form_data.get('CREDIT_ACTIVE', 'Active')
        CREDIT_TYPE = form_data.get('CREDIT_TYPE', 'Mortgage')
        ORG_TYPE_GROUPED = form_data.get('ORG_TYPE_GROUPED', 'Частная компания')
        OCCUPATION_TYPE_RU = form_data.get('OCCUPATION_TYPE_RU', 'Основной персонал')
        
        # Convert numeric fields
        CNT_CHILDREN = int(form_data.get('CNT_CHILDREN', 0))
        CNT_FAM_MEMBERS = int(form_data.get('CNT_FAM_MEMBERS', 1))
        CNT_CREDIT_PROLONG = int(form_data.get('CNT_CREDIT_PROLONG', 0))
        AMT_INCOME_TOTAL = float(form_data.get('AMT_INCOME_TOTAL', 100000))
        AMT_CREDIT = float(form_data.get('AMT_CREDIT', 1000000))
        AMT_ANNUITY = float(form_data.get('AMT_ANNUITY', 50000))
        
        # Convert age and employment
        AGE_YEARS = float(form_data.get('DAYS_BIRTH_years', 30))
        EMPLOYMENT_YEARS = float(form_data.get('DAYS_EMPLOYED_years', 5))
        
        # Other fields
        CITY = 'Алматы'  # Default city
        if 'REGION_POPULATION_RELATIVE' in form_data:
            # Find closest city based on density
            density = float(form_data.get('REGION_POPULATION_RELATIVE', 0.5))
            min_diff = float('inf')
            for city, city_density in city_to_density.items():
                diff = abs(density - city_density)
                if diff < min_diff:
                    min_diff = diff
                    CITY = city
        
        LOAN_DURATION = int(form_data.get('LOAN_DURATION', 240))
        ANNUITY_INCOME_RATIO = float(form_data.get('ANNUITY_INCOME_RATIO', 0.5))
        
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
        
        # Apply advanced credit scoring algorithm
        
        # Start with a base score
        base_score = 50.0
        
        # Factor 1: Property ownership
        if FLAG_OWN_REALTY == 'Y':
            base_score += 10.0
        else:
            base_score -= 5.0
            
        # Factor 2: Employment type
        employment_scores = {
            'Working': 5.0,
            'State servant': 10.0,
            'Commercial associate': 7.0,
            'Student': -10.0
        }
        base_score += employment_scores.get(NAME_INCOME_TYPE, 0.0)
        
        # Factor 3: Education
        education_scores = {
            'Lower secondary': -5.0,
            'Secondary / secondary special': 0.0,
            'Incomplete higher': 5.0,
            'Higher education': 10.0,
            'Academic degree': 15.0
        }
        base_score += education_scores.get(NAME_EDUCATION_TYPE, 0.0)
        
        # Factor 4: Credit history
        if CREDIT_ACTIVE == 'Active':
            base_score -= 5.0
        elif CREDIT_ACTIVE == 'Closed':
            base_score += 10.0
        elif CREDIT_ACTIVE == 'Bad debt':
            base_score -= 30.0
        
        # Factor 5: Age
        if AGE_YEARS < 25:
            base_score -= 10.0
        elif 25 <= AGE_YEARS < 35:
            base_score += 5.0
        elif 35 <= AGE_YEARS < 50:
            base_score += 10.0
        elif 50 <= AGE_YEARS < 65:
            base_score += 0.0
        else:
            base_score -= 10.0
            
        # Factor 6: Children
        if CNT_CHILDREN == 0:
            base_score += 5.0
        elif CNT_CHILDREN <= 2:
            base_score += 0.0
        else:
            base_score -= 5.0 * (CNT_CHILDREN - 2)
            
        # Factor 7: Credit to income ratio
        if AMT_INCOME_TOTAL > 0 and AMT_CREDIT > 0:
            ratio = AMT_CREDIT / AMT_INCOME_TOTAL
            if ratio < 3:
                base_score += 15.0
            elif ratio < 5:
                base_score += 5.0
            elif ratio < 10:
                base_score -= 10.0
            else:
                base_score -= 25.0
                
        # Factor 8: Employment length
        if EMPLOYMENT_YEARS < 1:
            base_score -= 10.0
        elif EMPLOYMENT_YEARS < 3:
            base_score += 0.0
        elif EMPLOYMENT_YEARS < 5:
            base_score += 5.0
        else:
            base_score += 10.0
            
        # Factor 9: Annuity to income ratio
        if ANNUITY_INCOME_RATIO > 0:
            if ANNUITY_INCOME_RATIO < 0.2:
                base_score += 15.0
            elif ANNUITY_INCOME_RATIO < 0.3:
                base_score += 10.0
            elif ANNUITY_INCOME_RATIO < 0.4:
                base_score += 0.0
            elif ANNUITY_INCOME_RATIO < 0.5:
                base_score -= 10.0
            else:
                base_score -= 20.0
       
        # Ensure score is between 0 and 100
        approval_chance = max(0, min(100, base_score))
        
        # Calculate scaled probability for credit score
        approval_prob = approval_chance / 100
        credit_score = int(300 + approval_prob * 700)
        
        # Fetch ChatGPT recommendation based on user inputs
        try:
            # Prepare a system message to instruct the model
            system_msg = "Вы — опытный ипотечный консультант. Давайте конкретные рекомендации по ипотечным стратегиям и продуктам на основе входных-личных данных клиента. Рекомендация должна быть максимально точной и личной. Не задавайте дополнительных вопросов и не используй форматирование в виде ** и т.д..- ты просто даешь общую рекомендацию"
            # Compose user message with all relevant client data
            user_msg = f"""Данные клиента:
- Пол: {CODE_GENDER}
- Есть недвижимость: {FLAG_OWN_REALTY}
- Тип занятости: {NAME_INCOME_TYPE}
- Семейное положение: {NAME_FAMILY_STATUS}
- Образование: {NAME_EDUCATION_TYPE}
- Кредитная история: {CREDIT_ACTIVE}
- Профессия: {OCCUPATION_TYPE_RU}
- Количество детей: {CNT_CHILDREN}
- Размер семьи: {CNT_FAM_MEMBERS}
- Ежемесячный доход: {AMT_INCOME_TOTAL}
- Сумма кредита: {AMT_CREDIT}
- Аннуитетный платёж: {AMT_ANNUITY}
- Срок кредита: {LOAN_DURATION} месяцев
- Соотношение аннуитета к доходу: {ANNUITY_INCOME_RATIO}
- Плотность региона: {city_to_density.get(CITY)}
Предложите подходящие ипотечные продукты и стратегии для клиента без дополнительных вопросов."""
            gpt_resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
            )
            gpt_recommendation = gpt_resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error fetching GPT recommendation: {e}")
            gpt_recommendation = ""
        
        # --- Find suitable mortgage products using simplified approach ---
        recs = "⚠️ Нет подходящих ипотек"
        
        if mortgages is not None:
            # Simple filtering based on client profile
            filtered_mortgages = mortgages
            
            # Filter by income type if in state service
            if NAME_INCOME_TYPE == 'State servant':
                # Try to find products for government employees
                gov_keywords = ['государств', 'госслуж', 'бюджет']
                filtered_mortgages = mortgages[mortgages['product.description'].fillna('').str.lower().str.contains('|'.join(gov_keywords), regex=True)]
            
            # If student, look for young family programs
            if AGE_YEARS < 25 or NAME_INCOME_TYPE == 'Student':
                young_keywords = ['молод', 'студент', 'жас отбасы']
                young_mortgages = mortgages[mortgages['product.description'].fillna('').str.lower().str.contains('|'.join(young_keywords), regex=True)]
                
                if not young_mortgages.empty:
                    filtered_mortgages = young_mortgages
            
            # Get random 3 mortgages if we have too many
            if len(filtered_mortgages) > 3:
                filtered_mortgages = filtered_mortgages.sample(3)
            
            # Format recommendations
            if not filtered_mortgages.empty:
                formatted_recs = []
                for _, row in filtered_mortgages.iterrows():
                    name = row['product.name']
                    desc = str(row.get('product.description', ''))[:100].strip(" .") + "..." if isinstance(row.get('product.description'), str) else "Нет описания"
                    sim_score = random.randint(70, 95)  # Simulate similarity score
                    url = row.get("product.company.website", "#")
                    formatted_recs.append(f"{name} - {desc} (Совпадение: {sim_score:.1f}%)")
                recs = "\n".join(formatted_recs)
        
        logger.info(f"Calculated approval chance: {approval_chance}%, credit score: {credit_score}")
        
        return jsonify({
            'approval_chance': approval_chance,
            'credit_score': credit_score,
            'recommendations': recs,
            'gpt_recommendation': gpt_recommendation,
            'message': f'Шанс одобрения кредита: {approval_chance}%'
        })
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Произошла ошибка при обработке запроса',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 