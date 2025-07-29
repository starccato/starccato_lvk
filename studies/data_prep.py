from tqdm import tqdm
import subprocess

def run_quiet(cmd):
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    for i in tqdm(range(101)):
        run_quiet(["get_analysis_data", str(i), "--trigger_type", "blip"])
        run_quiet(["get_analysis_data", str(i), "--trigger_type", "noise"])

if __name__ == "__main__":
    main()

