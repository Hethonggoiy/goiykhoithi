import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ==== 1. Đọc dữ liệu huấn luyện ====
print("📥 Đang nạp dữ liệu...")
df = pd.read_csv("data/du_lieu_chuan.csv")

# ==== 2. Tách đặc trưng & nhãn ====
mon = ['Toán','Văn','Anh','Lý','Hóa','Sinh','Sử','Địa','GDKTPL','Tin']
sothich_cols = [col for col in df.columns if "ST_" in col]
X = df[mon + sothich_cols]
y = df['Khối thi']

# ==== 3. Tiền xử lý: LabelEncoder + Scaling ====
le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ==== 4. Chia train/test ====
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

n_classes = len(np.unique(y_enc))

# ==== 5. Huấn luyện mô hình XGBoost ====
print("🚀 Huấn luyện mô hình XGBoost...")
xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# ==== 6. Huấn luyện mô hình MLP ====
print("🚀 Huấn luyện mô hình MLP...")
model_mlp = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
model_mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_mlp.fit(X_train, to_categorical(y_train), epochs=50, batch_size=32, verbose=1, validation_split=0.1)

# ==== 7. Kết hợp mô hình: Hybrid Predict ====
def predict_hybrid(x_input_scaled, w1=0.4, w2=0.6):
    pred_mlp = model_mlp.predict(x_input_scaled)
    pred_xgb = xgb_model.predict_proba(x_input_scaled)
    return (w1 * pred_mlp + w2 * pred_xgb)

# ==== 8. Đánh giá mô hình ====
print("📊 Đánh giá mô hình Hybrid:")
preds = predict_hybrid(X_test)
pred_labels = np.argmax(preds, axis=1)
print(classification_report(y_test, pred_labels, target_names=le.classes_))
print(f"🎯 Accuracy: {accuracy_score(y_test, pred_labels):.4f}")

# ==== 9. Lưu mô hình ====
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/xgb_model.pkl")
model_mlp.save("models/mlp_model.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")
print("✅ Đã lưu mô hình vào thư mục models/")