import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
import numpy as np
import pickle
import os
import traceback

file_path = 'C:/Users/harin/Downloads/gdp-X-main/gdp-X-main/backend/model/gdp.csv'
model_dir = 'C:/Users/harin/Downloads/gdp-X-main/gdp-X-main/backend/model'
model_path = os.path.join(model_dir, 'gdp_model.pkl')

try:
    print("📥 Loading dataset...")
    data = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")

    print("🧹 Cleaning dataset...")
    data = data.dropna()
    data = data.astype(float)

    print("🔍 Preparing features and target...")
    X = data.drop(columns=['GDP (US$ Trillion)', 'Year'])
    y = np.log(data['GDP (US$ Trillion)'])

    print(f"📊 Features shape: {X.shape}, Target shape: {y.shape}")

    print("🤖 Training Ridge Regression model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_score = r2_score(y_train, y_train_pred)
    test_score = r2_score(y_test, y_test_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f"✅ Train R² Score: {train_score:.4f}")
    print(f"✅ Test R² Score: {test_score:.4f}")
    print(f"✅ Cross-Validation Mean R²: {cv_scores.mean():.4f}")

    print("💾 Saving model...")
    model_package = {
        'model': model,
        'features': list(X.columns)
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"🎉 Model saved successfully at: {model_path}")

    print("\n📌 Feature Coefficients:")
    for feat, coef in zip(X.columns, model.coef_):
        print(f"{feat}: {coef:.4f}")

except Exception as e:
    print("❌ Error occurred:")
    traceback.print_exc()
