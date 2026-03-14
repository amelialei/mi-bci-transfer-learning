import json

# Check subject_01_intermediate.json
with open(r"C:\Users\Hridyanshu\PycharmProjects\JupyterProject\results\9x5_study\subject_01_intermediate.json", 'r') as f:
    data = json.load(f)
    print("Type:", type(data))
    print("Number of items:", len(data))
    print("\nFirst item structure:")
    print(json.dumps(data[0], indent=2))