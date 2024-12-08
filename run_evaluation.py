import evaluation
import json
import os

os.environ['TRANSFORMERS_CACHE'] = '.cache/huggingface/transformers/'


res_list = []
model_list = ["BaseModel", "LoRA", "PreFixTuning", "hadamardAdapter"]
for i in model_list :
    model_nm = "final_model/" + i
    lora_checkpoint = "checkpoint/" + i
    prefix_config = None
    if i == "PreFixTuning" :
        prefix_config = "checkpoint/" + i
        lora_checkpoint = None
    if i != "hadamardAdapter" :
        res_list.append(evaluation.main2(model_name=model_nm, lora_checkpoint=lora_checkpoint, prefix_config= prefix_config))
    else :
        res_list.append(evaluation.main(model_name=model_nm, lora_checkpoint=lora_checkpoint, prefix_config= prefix_config))
    lora_checkpoint = None

with open("evaluation_result.json", "w") as file:
    json.dump(res_list, file, indent=4)