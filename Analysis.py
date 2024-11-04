import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import messagebox
import warnings

warnings.filterwarnings('ignore')


df_clients = pd.read_csv('echantillons_clients(in).csv', encoding='latin1')
df_transactions = pd.read_csv('transactions_monetiques_annee_glissante_anonymise(in).csv', encoding='latin1')
online_operations_df = pd.read_csv('operations_en_ligne_annee_glissante_anonymise(in).csv', encoding='latin1')
agency_operations_df = pd.read_csv('operations_en_agence_annee_glissante_anonymise(in).csv', encoding='latin1')
al_akhawayn_df = pd.read_csv('al_akhawayn_dataset_v2(in).csv', encoding='latin1')


print("Clients DataFrame shape:", df_clients.shape)
print("Transactions DataFrame shape:", df_transactions.shape)
print("Online Operations DataFrame shape:", online_operations_df.shape)
print("Agency Operations DataFrame shape:", agency_operations_df.shape)
print("Al Akhawayn DataFrame shape:", al_akhawayn_df.shape)


sample_size = min(1000, len(df_clients))
df_clients = df_clients.sample(n=sample_size, random_state=42)

client_ids = df_clients['identifiant_client'].astype(str).unique()
df_transactions = df_transactions[df_transactions['identifiant_client'].astype(str).isin(client_ids)]
online_operations_df = online_operations_df[online_operations_df['identifiant_client'].astype(str).isin(client_ids)]
agency_operations_df = agency_operations_df[agency_operations_df['identifiant_client'].astype(str).isin(client_ids)]
al_akhawayn_df = al_akhawayn_df[al_akhawayn_df['identifier'].astype(str).isin(client_ids)]


df_clients['identifiant_client'] = df_clients['identifiant_client'].astype(str)
df_transactions['identifiant_client'] = df_transactions['identifiant_client'].astype(str)
online_operations_df['identifiant_client'] = online_operations_df['identifiant_client'].astype(str)
agency_operations_df['identifiant_client'] = agency_operations_df['identifiant_client'].astype(str)
al_akhawayn_df['identifier'] = al_akhawayn_df['identifier'].astype(str)


transactions_agg = df_transactions.groupby('identifiant_client').agg({
    'montant_local': 'sum',
    'libelle_nature_transaction': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
    'date_transaction': 'max'
}).reset_index()

online_operations_agg = online_operations_df.groupby('identifiant_client').agg({
    'montant_transaction': 'sum',
    'service': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
}).reset_index()

agency_operations_agg = agency_operations_df.groupby('identifiant_client').agg({
    'montant_operation': 'sum',
    'operation': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
}).reset_index()


merged_data = df_clients.copy()

merged_data = merged_data.merge(transactions_agg, on='identifiant_client', how='left')

merged_data = merged_data.merge(online_operations_agg, on='identifiant_client', how='left', suffixes=('', '_online'))

merged_data = merged_data.merge(agency_operations_agg, on='identifiant_client', how='left', suffixes=('', '_agency'))

merged_data = merged_data.merge(
    al_akhawayn_df[['identifier', 'age', 'situation_familiale']],
    left_on='identifiant_client',
    right_on='identifier',
    how='left',
    suffixes=('', '_al')
)

merged_data['age'] = merged_data['age'].fillna(merged_data['age_al'])
merged_data['situation_familiale'] = merged_data['situation_familiale'].fillna(merged_data['situation_familiale_al'])

merged_data.drop(columns=['age_al', 'situation_familiale_al', 'identifier'], inplace=True)


required_columns = [
    'age', 'anciennete_client', 'genre', 'situation_familiale',
    'montant_local', 'libelle_nature_transaction'
]
missing_columns = [col for col in required_columns if col not in merged_data.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

for col in ['genre', 'situation_familiale']:
    merged_data[col] = merged_data[col].fillna(merged_data[col].mode()[0])

numerical_features = [
    'age', 'anciennete_client', 'montant_local',
    'montant_transaction', 'montant_operation'
]
for col in numerical_features:
    merged_data[col] = merged_data[col].fillna(merged_data[col].mean())


merged_data['date_transaction'] = pd.to_datetime(merged_data['date_transaction'], errors='coerce')

merged_data['days_since_last_transaction'] = (
    pd.to_datetime('today') - merged_data['date_transaction']
).dt.days
merged_data['days_since_last_transaction'] = merged_data['days_since_last_transaction'].fillna(
    merged_data['days_since_last_transaction'].mean()
)

merged_data['avg_transaction_amount'] = merged_data['montant_local'] / merged_data['anciennete_client']
merged_data['avg_transaction_amount'] = merged_data['avg_transaction_amount'].fillna(
    merged_data['avg_transaction_amount'].mean()
)



merged_data['age_group'] = pd.cut(merged_data['age'], bins=[18, 30, 45, 60, 75, 100], labels=['18-30', '31-45', '46-60', '61-75', '76+'])

plt.figure(figsize=(10,6))
sns.countplot(data=merged_data, x='age_group', order=['18-30', '31-45', '46-60', '61-75', '76+'])
plt.title('Frequency of Digital Payments by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Clients')
plt.show()


plt.figure(figsize=(6,6))
sns.countplot(data=merged_data, x='genre')
plt.title('Frequency of Digital Payments by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Clients')
plt.show()


age_group_avg = merged_data.groupby('age_group')['avg_transaction_amount'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=age_group_avg, x='age_group', y='avg_transaction_amount', order=['18-30', '31-45', '46-60', '61-75', '76+'])
plt.title('Average Transaction Amount by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Average Transaction Amount')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data=merged_data, x='days_since_last_transaction', bins=30, kde=True)
plt.title('Transaction Frequency Pattern')
plt.xlabel('Days Since Last Transaction')
plt.ylabel('Number of Clients')
plt.show()


categorical_features = ['genre', 'situation_familiale', 'service', 'operation']
numerical_features = [
    'age', 'anciennete_client', 'montant_local', 'montant_transaction',
    'montant_operation', 'days_since_last_transaction', 'avg_transaction_amount'
]

for col in categorical_features:
    merged_data[col] = merged_data[col].fillna('Unknown')

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_categorical = pd.DataFrame(encoder.fit_transform(merged_data[categorical_features]))
encoded_categorical.columns = encoder.get_feature_names_out(categorical_features)
encoded_categorical.index = merged_data.index

X = pd.concat([merged_data[numerical_features], encoded_categorical], axis=1)


merged_data['libelle_nature_transaction'] = merged_data['libelle_nature_transaction'].astype(str)

merged_data['libelle_nature_transaction'] = merged_data['libelle_nature_transaction'].fillna('Unknown')

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(merged_data['libelle_nature_transaction'])


class_counts = pd.Series(y).value_counts()
rare_classes = class_counts[class_counts < 6].index

if len(rare_classes) > 0:
    print(f"Classes with fewer than 6 samples detected: {list(rare_classes)}. They will be excluded.")
    mask = ~pd.Series(y).isin(rare_classes)
    X = X[mask]
    y = y[mask]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = pd.Series(y_train).reset_index(drop=True)
y_test = pd.Series(y_test).reset_index(drop=True)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classifier Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))


importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importances:")
print(feature_importance_df.head(10))


def get_user_input():
    input_window = tk.Toplevel(root)
    input_window.title("Input Client Data")

    labels = ["Age", "Seniority (in years)", "Montant Local", "Days Since Last Transaction", "Average Transaction Amount"]
    entries = []

    for label in labels:
        tk.Label(input_window, text=label).pack(pady=5)
        entry = tk.Entry(input_window)
        entry.pack(pady=5)
        entries.append(entry)

    def submit_input():
        try:
            user_data = {
                'age': float(entries[0].get()),
                'anciennete_client': float(entries[1].get()),
                'montant_local': float(entries[2].get()),
                'days_since_last_transaction': float(entries[3].get()),
                'avg_transaction_amount': float(entries[4].get())
            }
            
            categorical_columns = X.columns[len(user_data):]
            categorical_data = pd.DataFrame(0, index=[0], columns=categorical_columns)
            
            input_df = pd.concat([pd.DataFrame([user_data]), categorical_data], axis=1)
            
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            
            prediction = rf_model.predict(input_df)[0]
            prediction_label = label_encoder.inverse_transform([prediction])[0]

            messagebox.showinfo("Prediction Result", f"Predicted Transaction Type: {prediction_label}")
            input_window.destroy()

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

    submit_button = tk.Button(input_window, text="Submit", command=submit_input)
    submit_button.pack(pady=10)


def show_random_prediction():
    random_index = X_test.sample(1, random_state=42).index[0]
    random_user = X_test.loc[[random_index]]
    prediction = rf_model.predict(random_user)[0]
    actual_label = y_test.loc[random_index]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    actual_label_decoded = label_encoder.inverse_transform([actual_label])[0]

    message = f"""Client Information:
- Age: {random_user['age'].values[0]}
- Seniority: {random_user['anciennete_client'].values[0]} years
- Montant Local: {random_user['montant_local'].values[0]}
- Days Since Last Transaction: {random_user['days_since_last_transaction'].values[0]}
- Average Transaction Amount: {random_user['avg_transaction_amount'].values[0]}

Predicted Transaction Type: {prediction_label}
Actual Transaction Type: {actual_label_decoded}


"""
    messagebox.showinfo("Client Prediction", message)

root = tk.Tk()
root.title("Predictive Analytics Tool")

frame = tk.Frame(root)
frame.pack(pady=20)

random_predict_button = tk.Button(frame, text="Get Random User Prediction", command=show_random_prediction)
random_predict_button.pack()





clustering_features = [
    'age', 'anciennete_client', 'montant_local',
    'montant_transaction', 'montant_operation',
    'days_since_last_transaction', 'avg_transaction_amount'
]

X_clustering = merged_data[clustering_features]

X_clustering = X_clustering.fillna(X_clustering.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

merged_data['cluster'] = clusters


cluster_profiles = merged_data.groupby('cluster')[clustering_features].mean()

print("Cluster Profiles:")
print(cluster_profiles)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pc_df['cluster'] = clusters

plt.figure(figsize=(10,6))
sns.scatterplot(data=pc_df, x='PC1', y='PC2', hue='cluster', palette='Set1')
plt.title('Clusters Visualization')
plt.show()


for cluster_num in range(optimal_clusters):
    print(f"\nCluster {cluster_num} Characteristics:")
    cluster_data = merged_data[merged_data['cluster'] == cluster_num]
    print(cluster_data.describe())



sample_indices = X_test.sample(10, random_state=42).index
samples = X_test.loc[sample_indices]
actual_labels = y_test.loc[sample_indices]
predicted_labels = rf_model.predict(samples)

actual_labels_decoded = label_encoder.inverse_transform(actual_labels)
predicted_labels_decoded = label_encoder.inverse_transform(predicted_labels)

results_df = pd.DataFrame({
    'Client ID': merged_data.loc[samples.index, 'identifiant_client'],
    'Age': merged_data.loc[samples.index, 'age'],
    'Actual Transaction Type': actual_labels_decoded,
    'Predicted Transaction Type': predicted_labels_decoded
})

print("\nPredictions vs Actual Labels for 10 Samples:")
print(results_df)


root.mainloop()
