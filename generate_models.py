import subprocess
import os
os.environ['TRANSFORMERS_CACHE'] = '.cache/huggingface/transformers/'

def run_model(command):
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            text=True, 
            capture_output=True, 
            check=True
        )
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running command: {command}\n{e.stderr}")

def main():
    # 모델 실행 명령어 리스트

    commands = [
        "python llma_sequence_model.py --base_model='openlm-research/open_llama_7b' --data_path='data/movies.json' --output_dir='final_model/BaseModel'",
        "python llma_sequence_model.py --base_model='meta-llama/Llama-3.2-3B-Instruct' --data_path='data/movies.json' --output_dir='final_model/LoRA'",
        "python prefix_finetune.py --base_model='yahma/llama-7b-hf' --data_path='data/movies.json' --output_dir='final_model/PreFixTuning'"    
    ]
    
    for i, cmd in enumerate(commands, start=1):
        print(f"Running Model {i}...")
        run_model(cmd)
        print(f"Model {i} finished.\n")

if __name__ == "__main__":
    main()
