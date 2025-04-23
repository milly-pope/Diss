from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling


def reformat(samples, file):
    samples_new = samples.apply(lambda col: col.astype('category').cat.codes)
    arities = samples.nunique().tolist()
    with open(file, "w") as f:
        f.write(" ".join(samples.columns) + "\n")
        f.write(" ".join(map(str, arities)) + "\n")
        samples_new.to_csv(f, sep=" ", index=False, header=False)

def gen_samples(bif_path, output_prefix, sample_sizes, num_files=5):
    reader = BIFReader(bif_path)
    model = reader.get_model()
    sampler = BayesianModelSampling(model)

    for size in sample_sizes:
        for i in range(1, num_files + 1):
            data = sampler.forward_sample(size=size)
            filename = f"{output_prefix}{i}_n{size}"
            reformat(data, filename)

sample_sizes = [5000]
gen_samples('alarm.bif', output_prefix='alarm', sample_sizes=sample_sizes)
