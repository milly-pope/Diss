from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
import pandas as pd

def reformat(samples, file):
    samples_new = samples.apply(lambda col: col.astype('category').cat.codes)
    arities = samples.nunique().tolist()
    with open(file, "w") as f:
        f.write(" ".join(samples.columns) + "\n")
        f.write(" ".join(map(str, arities)) + "\n")
        samples_new.to_csv(f, sep=" ", index=False, header=False)

def generate_all_samples(bif_path, output_prefix, sample_sizes, num_files=5):
    reader = BIFReader(bif_path)
    model = reader.get_model()
    sampler = BayesianModelSampling(model)

    for size in sample_sizes:
        for i in range(1, num_files + 1):
            print(f"Generating sample {i} with size {size}...")
            data = sampler.forward_sample(size=size)
            filename = f"{output_prefix}{i}_n{size}"
            reformat(data, filename)
            print(f"Saved: {filename}")

# Run this with your desired setup
sample_sizes = [1000, 10000, 50000]
generate_all_samples('alarm.bif', output_prefix='alarm', sample_sizes=sample_sizes)
