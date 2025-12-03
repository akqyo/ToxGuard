from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from captum.attr import IntegratedGradients 
import numpy as np 

# --- Константы ---
MODEL_DIR = "./rubert_model/" 
ALL_LABELS = ['NORMAL', 'INSULT', 'THREAT', 'OBSCENITY'] 

# --- Инициализация модели (Загрузка при старте) ---
try:
    # Загружаем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval() # Переводим модель в режим инференса
    print(f"Модель RuBERT успешно загружена из {MODEL_DIR}")
except Exception as e:
    print(f"ОШИБКА ЗАГРУЗКИ МОДЕЛИ: {e}")
    print("Проверь, что папка 'rubert_model/' лежит рядом с app_api.py")
    
# --- Схемы данных ---
class TextIn(BaseModel):
    text: str

class ModerationResult(BaseModel):
    text: str
    is_normal: bool
    is_toxic: bool
    threat: bool
    sexual_content: bool
    confidence: dict

# Функция для создания токенов для Captum
def construct_input_and_token_type(input_text):
    return tokenizer(
        input_text, 
        return_tensors="pt", 
        padding="max_length", 
        max_length=128, 
        truncation=True
    )

CLASS_INDEX = {label: i for i, label in enumerate(ALL_LABELS)}

def explain_text(text: str, target_class_index: int):
    """
    Рассчитывает значимость, используя Embedding-слой в качестве входного тензора
    для обхода ошибки типа.
    """

    input_data = construct_input_and_token_type(text)
    input_ids = input_data['input_ids'].to(model.device).long()
    attention_mask = input_data['attention_mask'].to(model.device).long()
    
    embeddings = model.get_input_embeddings()(input_ids)
    
    def forward_with_only_embeddings(embeddings, attention_mask=None, input_ids=None):
        return model(inputs_embeds=embeddings, attention_mask=attention_mask).logits

    ig = IntegratedGradients(forward_with_only_embeddings)

    attributions, delta = ig.attribute(
        inputs=embeddings, # Вход: Float Tensor
        target=target_class_index, 
        additional_forward_args=(attention_mask, input_ids), # Дополнительные входы
        return_convergence_delta=True
    )

    attributions = attributions.sum(dim=-1).squeeze(0) 
    
    attributions = attributions / torch.linalg.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    
    word_scores = []
    for token, score in zip(tokens, attributions):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        clean_token = token.replace('##', '') 
        word_scores.append({'word': clean_token, 'score': float(score)})
        
    return word_scores


# --- Основная логика классификации ---
def classify_text(text: str) -> ModerationResult:
    # Токенизация текста
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Получение вероятностей (логитов) и применение сигмоиды для многометочной классификации
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze().tolist() 
    
    # Форматирование результатов
    confidences = {}
    for i, label in enumerate(ALL_LABELS):
        # Округляем вероятность
        confidences[label] = round(probabilities[i], 4)

    # Принятие решения (порог 0.5)
    threshold = 0.5
    threat = confidences.get('THREAT', 0) > threshold
    sexual_content = confidences.get('OBSCENITY', 0) > threshold
    insult = confidences.get('INSULT', 0) > threshold
    normal = confidences.get('NORMAL', 0) > threshold
    
    # Определение общей токсичности
    is_toxic = threat or sexual_content or insult
    # Считаем нормальным, если предсказана метка NORMAL (и/или нет токсичных)
    is_normal = normal and not is_toxic 
    
    return ModerationResult(
        text=text,
        is_normal=is_normal,
        is_toxic=is_toxic,
        threat=threat,
        sexual_content=sexual_content,
        confidence=confidences
    )

# --- Эндпоинт ---
app = FastAPI(title="ToxGuard Moderator API", version="1.0")

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/moderate", response_model=ModerationResult)
def moderate_text_endpoint(input: TextIn):
    return classify_text(input.text)

class ExplainRequest(BaseModel):
    text: str
    target_class: str  # Имя класса, который нужно объяснить (INSULT, THREAT...)

# Эндпоинт для объяснения
@app.post("/api/explain")
def explain_decision(request: ExplainRequest):
    try:
        target_index = CLASS_INDEX.get(request.target_class.upper())
        if target_index is None:
            return {"error": "Invalid target class", "available_classes": ALL_LABELS}
            
        word_scores = explain_text(request.text, target_index)
        
        return {
            "text": request.text,
            "target_class": request.target_class,
            "explanation": word_scores
        }
    except Exception as e:
        return {"error": f"Ошибка при объяснении: {str(e)}"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}