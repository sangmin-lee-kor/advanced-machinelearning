import evaluation
import json
import os

os.environ['TRANSFORMERS_CACHE'] = '.cache/huggingface/transformers/'


res_list = []
model_list = ["BaseModel", "LoRA", "PreFixTuning"]
# data_path = "data/test/movies.json"
for i in model_list :
    model_nm = "final_model/" + i
    lora_checkpoint = "final_model/" + i
    prefix_config = None
    if i == "PreFixTuning" :
        prefix_config = "final_model/" + i
        lora_checkpoint = None
    res_list.append(evaluation.main(model_name=model_nm, lora_checkpoint=lora_checkpoint, prefix_config= prefix_config))
    lora_checkpoint = None

with open("evaluation_result.json", "w") as file:
    json.dump(res_list, file, indent=4)