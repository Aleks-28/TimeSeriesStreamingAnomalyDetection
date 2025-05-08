import multiprocessing
import subprocess
import os

# Function to extract individual jobs from the bash script
def extract_jobs(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    # Remove comments and empty lines
    jobs = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    return jobs

# Function to execute a single job (command)
def run_job_silent(job):
    print(f"Running: {job}")  # You can remove this if you don't want to see even this.
    with open(os.devnull, 'w') as devnull:
        try:
            subprocess.run(job, shell=True, stdout=devnull, stderr=devnull, check=True)
            print(f"Finished: {job}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {job}: {e}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Path to your bash script
    bash_file = "./scripts/run_jobs.sh"

    # Extract individual jobs from the script
    jobs = extract_jobs(bash_file)

    # Use multiprocessing to run jobs concurrently
    num_processes = min(len(jobs), multiprocessing.cpu_count())
    with multiprocessing.Pool(num_processes) as pool:
        pool.map(run_job_silent, jobs)
