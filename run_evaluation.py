import evaluation
import json


res_list = []
model_list = ["BaseModel", "LoRA", "PreFixTuning"]
for i in model_list :
    model_nm = "final_model/" + i
    lora_checkpoint = "checkpoint/" + i
    res_list.append(evaluation.main(model_name=model_nm, lora_checkpoint=lora_checkpoint))
    lora_checkpoint = None

with open("evaluation_result.json", "w") as file:
    json.dump(res_list, file, indent=4)