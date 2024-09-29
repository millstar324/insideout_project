from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import random
from fastapi.middleware.cors import CORSMiddleware
import re
app = FastAPI()

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 요청을 허용 (*)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
   model_name = "1chae/emotion_conversation",
   max_seq_length = max_seq_length,
   dtype = dtype,
   load_in_4bit = load_in_4bit,)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Emotion list
emotion_list = {
    "기쁨": ("Optimistic and always positive, Joy is focused on finding happiness in every situation.", "./yellow_m.png"),
    "슬픔": ("Thoughtful and empathetic, Sadness helps to express and process difficult emotions.", "./blue_m.png"),
    "화남": ("Fiery and quick-tempered, Anger stands up for fairness but can be impulsive.", "./red_m.png"),
    "불안": ("Cautious and nervous, Fear is constantly worried about potential dangers and risks.", "./purple_m.png"),
    "까칠": ("Sharp and opinionated, Disgust avoids anything unpleasant and stands up for personal values.", "./green_m.png")
}

def extract_model_output(text):
    for data in text:
        # <eos> 앞에 있는 감정만 추출하는 정규식
        match = re.search(r'### 감정:\s*(.+?)<eos>', data)

        if match:
            emotions = match.group(1).strip()
            print(emotions)
        return emotions

def extract_model_output2(text):
    for data in text:
    # <eos> 앞에 있는 응답만 추출하는 정규식
        match = re.search(r'응답:\s*(.+?)<eos>', data)

        if match:
            response = match.group(1).strip()
            print(response)
        return response


def generate_character_response_prompt(message, character, personality):
    prompt = f"""<bos><start_of_turn>user
    다음의 문장에 대해 {character}이 캐릭터에 어울리는 응답을 하시오.
    문장: 남친이랑 헤어졌어.. 너무 보고싶다
    <end_of_turn>
    <start_of_turn>model--
    """
    return prompt



def question_gemma(sentence, model, tokenizer, temperature=0.0, max_new_tokens=50):
    prompt2 = """다음의 문장에서 느껴지는 감정을 '기쁨','슬픔','화남','짜증','불안' 이 다섯가지 감정 안에서 두개의 감정을 고르거나(한개의 감정보다는 되도록 2개의 감정을 골라줘) 해당하는 감정이 없으면 '좀더 자세히 말해주세요'라고 적어줘

    ### 문장:
    {}

    ### 감정:
    """
    # alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        prompt2.format(
            # "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,", # instruction
            sentence, # instruction
            # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 164, use_cache = True)
    print(tokenizer.batch_decode(outputs))


    emotion_prediction = extract_model_output(tokenizer.batch_decode(outputs))
    print(f"정리:{emotion_prediction}")
    random_emotion = random.choice(list(emotion_list.items()))

    prompt = """다음의 문장에 대해 ['{}']이 캐릭터에 어울리는 응답을 하시오.
    문장:
    {}
    응답:
    """
    # alpaca_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        prompt.format(
            # "Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,", # instruction
            random_emotion[0], # instruction
            sentence, # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    print(tokenizer.batch_decode(outputs))


    character_response = extract_model_output2(tokenizer.batch_decode(outputs))
    print(f"정리:{character_response}")

    return emotion_prediction, character_response, random_emotion[1][1], random_emotion[0]

class UserInput(BaseModel):
    sentence: str

@app.post("/predict")
async def predict_emotion_and_response(user_input: UserInput):
    emotion_prediction, character_response, _, rand_character = question_gemma(user_input.sentence, model, tokenizer)
    return JSONResponse({
        "emotion_prediction": emotion_prediction,
        "character_response": character_response,
        "character": rand_character
    })
