import evaluation
import json


res_list = []
model_list = ["final_model/BaseModel", "final_model/LoRA"]
for i in model_list :
    res_list.append(evaluation.main(model_name=i))

with open("evaluation_result.json", "w") as file:
    json.dump(res_list, file, indent=4)