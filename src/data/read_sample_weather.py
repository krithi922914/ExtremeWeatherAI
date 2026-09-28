from netCDF4 import Dataset

file_path = "data/samples/sample_weather.nc"

dataset = Dataset(file_path, "r")

print("Dataset opened successfully!")
print()
print("Variables:")

for variable_name in dataset.variables:
    variable = dataset.variables[variable_name]
    print(
        variable_name,
        "->",
        "shape:", variable.shape,
        "units:", getattr(variable, "units", "N/A")
    )

dataset.close()