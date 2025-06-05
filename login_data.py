import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker and set seeds for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
NUM_RECORDS = 200_000  # Adjust as needed
NUM_USERS = 1000

# Generate fake usernames
users = [fake.user_name() for _ in range(NUM_USERS)]
statuses = ['Success', 'Failed']

# Generate login records
data = []
start_date = datetime(2025, 1, 1)

for _ in range(NUM_RECORDS):
    username = random.choice(users)
    timestamp = start_date + timedelta(
        days=random.randint(0, 130),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    ip_address = fake.ipv4_public()
    location = f"{fake.city()}, {fake.country()}"
    login_status = random.choices(statuses, weights=[0.85, 0.15])[0]  # 85% success, 15% failed
    access_duration = round(random.uniform(1, 60), 2) if login_status == 'Success' else 0
    
    data.append([
        username, timestamp, ip_address, location, login_status, access_duration
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "Username", "Timestamp", "IP_Address", "Location", "Login_Status", "Access_Duration_Min"
])

# Save to CSV
df.to_csv("dummy_login_data_200k.csv", index=False)

print("Dummy login data generated and saved as 'login_data_200k.csv'")

df = pd.read_csv('dummy_login_data_200k.csv')
print(df.head())

#data cleaning
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
print(df.isnull().sum())  # Check missing values
df.dropna(subset=['Timestamp'], inplace=True)  # Drop rows with invalid timestamps

#failed login attempts per user
failed_logins=df[df['Login_Status']=='Failed'].groupby('Username').size().reset_index(name='Failed_Attempts')
print(failed_logins.sort_values(by='Failed_Attempts',ascending=False).head(10))

#users logging in from multiple locations
locations_per_user=df.groupby('Username')['Location'].nunique().reset_index(name='Location_Count')
unusual_users=locations_per_user[locations_per_user['Location_Count']>3]
print(unusual_users)

#logins at odd hours(12AM-5AM)
df['Hour'] = df['Timestamp'].dt.hour
odd_hour_logins = df[(df['Hour'] >= 0) & (df['Hour'] <= 5)]
print(odd_hour_logins['Username'].value_counts().head(17))

#Visualization 
#failed login attempts count distribution
plt.figure(figsize=(10,6))
sns.histplot(failed_logins['Failed_Attempts'],bins=30)
plt.title('Distribution of Failed Login Attempts per User')
plt.xlabel('Failed Attempts')
plt.ylabel('Number of Users')
plt.show()

#login frequency by hour
plt.figure(figsize=(10,6))
sns.countplot(x='Hour',data=df)
plt.title('Login Countby Hour')
plt.show()

# extract day and hour
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# Create a binary flag for off-hours (e.g., 12 AM–5 AM)
df['Is_Odd_Hour'] = df['Hour'].apply(lambda x: 1 if x < 6 else 0)

from sklearn.ensemble import IsolationForest

features = ['Hour', 'DayOfWeek', 'Is_Odd_Hour']
X = df[features]

# Initialize and train model
model = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly'] = model.fit_predict(X)

# Label: -1 = anomaly, 1 = normal
anomalies = df[df['Anomaly'] == -1]
print(anomalies.head())

plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='Hour',y='DayOfWeek',hue='Anomaly',palette={1:'blue', -1:'red'})
plt.title("Login Behavior with Anomalies Highlighted")
plt.show()

df.to_csv("final_login_data.csv",index=False)