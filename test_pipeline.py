import pandas as pd
import numpy as np
from data_processor import TabularDataProcessor
from gan_model import TabularGAN
from evaluator import ModelEvaluator

print("Testing pipeline on sample data...")
df = pd.read_csv("sample_data/credit_default_imbalanced.csv")
processor = TabularDataProcessor(target_col="Default_Status")
info = processor.analyze_distribution(df)
print("Imbalance Analysis:", info["status"], "Ratio:", info["imbalance_ratio"])

# Preprocess features
X_proc = processor.fit_transform(df)
min_class = info["min_class"]
minority_mask = (df["Default_Status"] == min_class).values
minority_samples = X_proc[minority_mask]
print("Minority shape:", minority_samples.shape)

# Train small GAN for 5 epochs to verify
gan = TabularGAN(input_dim=minority_samples.shape[1], latent_dim=16)
history = gan.train(minority_samples, epochs=5, batch_size=16)
print("GAN training history epochs:", len(history["epoch"]))

# Generate samples
synth = gan.generate(20)
print("Generated shape:", synth.shape)
df_synth = processor.inverse_transform(synth)
print("Reconstructed sample:\n", df_synth.head(2))

# Test quality evaluator
real_min = df[df["Default_Status"] == min_class].drop(columns=["Default_Status"])
quality = ModelEvaluator.evaluate_quality(real_min, df_synth, processor.numerical_cols)
print("Quality metrics:", quality["total_synthetic"], "Exact overlap:", quality["exact_overlap_with_real"])
print("ALL PIPELINE TESTS PASSED!")
