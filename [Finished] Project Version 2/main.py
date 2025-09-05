from data_loader import load_and_prepare_data
from model_builder import build_model
from train import train_model
from evaluate import evaluate_model, plot_results

# Load data
x_train, x_test, y_train, y_test = load_and_prepare_data()

# Build model
model = build_model()

# Train model
history = train_model(model, x_train, y_train, x_test, y_test)

# Evaluate model
evaluate_model(model, x_test, y_test)

# Plot metrics
plot_results(history)

# Save the model
model.save("skin_cancer_model.h5")
