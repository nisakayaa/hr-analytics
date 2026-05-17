# HR Analytics — Employee Attrition Prediction

Bir şirketin çalışanlarının hangi faktörler yüzünden işten ayrıldığını tahmin etmeye çalıştığım sınıflandırma projesi. Logistic Regression ve Random Forest modellerini eğitip karşılaştırdım.


## Sorduğum sorular

- Çalışanlar neden ayrılıyor? Hangi faktör en belirleyici?
- Logistic Regression mı, Random Forest mı daha iyi?
- Class imbalance var mı? (Ayrılanlar azınlıkta)
- Model sonuçlarını yöneticiye nasıl anlatırım?

## Bulgular

- Random Forest, Logistic Regression'a göre **F1 skorda ~%8 daha iyi**
- En önemli 3 feature: **Overtime, MonthlyIncome, Age**
- Mesai yapanların ayrılma olasılığı, yapmayanlara göre **2.5 kat fazla**
- Genç çalışanlar (25-30 yaş) en yüksek ayrılma riskinde
- Class imbalance var (%84 kaldı, %16 ayrıldı) → accuracy yanıltıcı, F1/recall'a baktım

## Yöntem

1. **EDA:** dağılımları, korelasyonları, target ile ilişkileri incele
2. **Preprocessing:** kategorik feature'lar için OneHotEncoder, sayısal için StandardScaler
3. **Train/test split:** %80/%20, stratified
4. **Model 1:** Logistic Regression (baseline)
5. **Model 2:** Random Forest (hyperparameter tuning yok, default — bilinçli tercih)
6. **Karşılaştırma:** accuracy, precision, recall, F1, ROC-AUC
7. **Feature importance:** Random Forest'ın `feature_importances_` ile

## Kullandığım araçlar

- pandas, numpy
- scikit-learn (LogisticRegression, RandomForestClassifier, metrics)
- matplotlib, seaborn

## Çalıştırmak için

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/eda.py        # önce keşifsel analiz
python src/model.py      # sonra modelleme
```

## Not

Class imbalance'la ilk burada karşılaştım. İlk denememde modelin %84 accuracy çıktığını görünce sevinmiştim — sonra fark ettim ki **model herkese "ayrılmayacak" dese de %84 doğru tahmin** yapardı. Confusion matrix'e bakmak şart, accuracy tek başına yanıltıyor.

Bir sonraki versiyonda **SMOTE** ile oversampling denemek istiyorum. Bir de gerçek bir yöneticiye sunmak gerekse, modelin sonuçlarını SHAP değerleriyle bireysel olarak açıklayabilmek lazım — bu da öğrenme listemde.

## Author

Nisa Kaya — [github.com/nisakayaa](https://github.com/nisakayaa)
