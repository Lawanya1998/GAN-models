from data import NoBoomDataset

# 👇 adjust the path to your dataset folder if needed
root_path = "data/batch_dist_ternary_acetone_1_butanol_methanol"

print("🔄 Loading training data...")
train_ds = NoBoomDataset(
    dataset="batch_dist_ternary_acetone_1_butanol_methanol",
    version="1.0",
    root=root_path,
    train=True,
    include_misc_faults=True,
    include_controller_faults=True
)

print("✅ Training data loaded successfully!")
print(f"Total CSV files loaded: {len(train_ds)}")
print(f"Feature count per sample: {train_ds.num_features}")
print(f"Example feature names: {train_ds.features[:5]}")

print("\n🔄 Loading testing data...")
test_ds = NoBoomDataset(
    dataset="batch_dist_ternary_acetone_1_butanol_methanol",
    version="1.0",
    root=root_path,
    train=False,
    include_misc_faults=True,
    include_controller_faults=True
)

print("✅ Testing data loaded successfully!")
print(f"Total CSV files loaded: {len(test_ds)}")
print(f"Feature count per sample: {test_ds.num_features}")

# 🧩 Check one example to make sure data is numeric tensors
x, y = train_ds[0]
print(f"\nExample data tensor shape: {x[0].shape}")
print(f"Example target tensor shape: {y[0].shape}")
print("First few target values:", y[0][:10].tolist())
