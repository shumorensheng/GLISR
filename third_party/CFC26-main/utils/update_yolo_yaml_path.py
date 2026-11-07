import yaml
import glob

experiment_path = ""

for f in glob.glob(f"{experiment_path}/*.yaml"):
    with open(f) as file:
        data = yaml.safe_load(file)

    data['path'] = experiment_path

    with open(f, 'w') as file:
        yaml.safe_dump(data, file)
