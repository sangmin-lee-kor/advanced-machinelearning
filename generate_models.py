import subprocess

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
        "python llma_sequence_model.py --data_path train_data_copy.json ",  # 첫 번째 모델 실행
    ]
    
    for i, cmd in enumerate(commands, start=1):
        print(f"Running Model {i}...")
        run_model(cmd)
        print(f"Model {i} finished.\n")

if __name__ == "__main__":
    main()
