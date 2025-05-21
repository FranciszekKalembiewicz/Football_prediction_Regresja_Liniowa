import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings

# Ignorowanie ostrzeżeń
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Przygotowanie danych i modelowanie
df = pd.read_csv("results.csv")
df = df.copy()

# Przekształcenie wyniku meczu na punkty
df['home_points'] = df['result'].map({'H': 3, 'D': 1, 'A': 0})
df['away_points'] = df['result'].map({'H': 0, 'D': 1, 'A': 3})

# Sumowanie punktów dla gospodarzy i gości
home_points = df.groupby(['season', 'home_team'])['home_points'].sum().reset_index()
home_points.rename(columns={'home_team': 'team', 'home_points': 'points'}, inplace=True)

away_points = df.groupby(['season', 'away_team'])['away_points'].sum().reset_index()
away_points.rename(columns={'away_team': 'team', 'away_points': 'points'}, inplace=True)

# Łączymy punkty
all_points = pd.concat([home_points, away_points])
season_team_points = all_points.groupby(['season', 'team'])['points'].sum().reset_index()

# Lista wszystkich drużyn
teams = sorted(set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist()))

# Lista wszystkich sezonów
seasons = sorted(df['season'].unique())

# Tworzymy wszystkie kombinacje drużyn i sezonów
full_index = pd.MultiIndex.from_product([seasons, teams], names=['season', 'team'])
full_table = pd.DataFrame(index=full_index).reset_index()

# Łączenie z punktami, drużyny niegrające mają punkty = 0
full_table = full_table.merge(season_team_points, on=['season', 'team'], how='left')
full_table['points'] = full_table['points'].fillna(0)

# Liczenie pozycji na podstawie punktów
def assign_positions(group):
    sorted_teams = group.sort_values(by='points', ascending=False).reset_index(drop=True)
    sorted_teams['position'] = sorted_teams['points'].rank(method='min', ascending=False).astype(int)
    return sorted_teams

# Grupowanie po sezonach i przypisywanie pozycji
position_table = full_table.groupby('season', group_keys=False).apply(assign_positions).reset_index(drop=True)

# Jeśli drużyna nie grała w danym sezonie zostaje jej przypisana pozycja 21
def adjust_position(row):
    season_data = df[df['season'] == row['season']]
    played_teams = set(season_data['home_team']).union(set(season_data['away_team']))
    return row['position'] if row['team'] in played_teams else 21

position_table['position'] = position_table.apply(adjust_position, axis=1)

# Teraz wprowadzamy przygotowane dane do modelu
# Pivot table: przygotowanie X i Y
pivot_points = position_table.pivot(index='team', columns='season', values='points')
pivot_positions = position_table.pivot(index='team', columns='season', values='position')

# Dzielimy dane do nauki
X_points = pivot_points.loc[:, '2006-2007':'2016-2017'].copy()
Y_points = pivot_points['2017-2018'].copy()

X_positions = pivot_positions.loc[:, '2006-2007':'2016-2017'].copy()
Y_positions = pivot_positions['2017-2018'].copy()

# Tworzymy wagi sezonów (im sezon bliższy rzeczywistości tym bardziej wyniki są adekwatne)
weights = {
    '2006-2007': 1,
    '2007-2008': 2,
    '2008-2009': 3,
    '2009-2010': 4,
    '2010-2011': 5,
    '2011-2012': 6,
    '2012-2013': 7,
    '2013-2014': 8,
    '2014-2015': 9,
    '2015-2016': 10,
    '2016-2017': 11
}

# Ważymy dane
for season in X_points.columns:
    X_points.loc[:, season] = X_points.loc[:, season] * weights[season]
    X_positions.loc[:, season] = X_positions.loc[:, season] * weights[season]

print("Punkty -> X:", X_points.shape, "Y:", Y_points.shape)
print("Pozycje -> X:", X_positions.shape, "Y:", Y_positions.shape)

# Wybór modelu
# Wybieramy czy random forest True czy False jesli False to modelem bedzie linear regression
USE_RANDOM_FOREST = True

if USE_RANDOM_FOREST:
    model_points = RandomForestRegressor(n_estimators=100, random_state=42)
    model_positions = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model_points = LinearRegression()
    model_positions = LinearRegression()

# Walidacja krzyżowa
cv_scores_points = cross_val_score(model_points, X_points, Y_points, cv=5, scoring='neg_mean_absolute_error')
cv_scores_positions = cross_val_score(model_positions, X_positions, Y_positions, cv=5, scoring='neg_mean_absolute_error')

print(f"\nŚredni MAE (punkty) w walidacji krzyżowej: {-cv_scores_points.mean():.2f}")
print(f"Średni MAE (pozycje) w walidacji krzyżowej: {-cv_scores_positions.mean():.2f}")

# Trenowanie modeli
model_points.fit(X_points, Y_points)
model_positions.fit(X_positions, Y_positions)

# Predykcja
predicted_points = model_points.predict(X_points)
predicted_positions = model_positions.predict(X_positions)

# Ocena
mae_points = mean_absolute_error(Y_points, predicted_points)
rmse_points = np.sqrt(mean_squared_error(Y_points, predicted_points))
mae_positions = mean_absolute_error(Y_positions, predicted_positions)
rmse_positions = np.sqrt(mean_squared_error(Y_positions, predicted_positions))

print(f"\nDokładność predykcji punktów: MAE = {mae_points:.2f}, RMSE = {rmse_points:.2f}")
print(f"Dokładność predykcji pozycji: MAE = {mae_positions:.2f}, RMSE = {rmse_positions:.2f}")

# Porównanie wyników - punkty
comparison_points = pd.DataFrame({
    'Drużyna': Y_points.index,
    'Rzeczywiste Punkty': Y_points.values,
    'Przewidziane Punkty': np.round(predicted_points, 2)
}).sort_values(by='Przewidziane Punkty', ascending=False)

# Porównanie wyników - pozycje
comparison_positions = pd.DataFrame({
    'Drużyna': Y_positions.index,
    'Rzeczywista Pozycja': Y_positions.values,
    'Przewidziana Pozycja': np.round(predicted_positions, 2)
}).sort_values(by='Przewidziana Pozycja')

# Dodanie rzeczywistej pozycji do DataFrame z punktami
final_comparison = comparison_points.merge(comparison_positions[['Drużyna', 'Rzeczywista Pozycja']],
                                          on='Drużyna', how='left')

# Przypisanie przewidywanej pozycji na podstawie przewidywanych punktów
final_comparison['Przewidziana Pozycja (z punktów)'] = final_comparison['Przewidziane Punkty'].rank(ascending=False, method='min').astype(int)

# Sortowanie według przewidywanej pozycji (z punktów)
final_comparison = final_comparison.sort_values('Przewidziana Pozycja (z punktów)')

# Wybór kolumn do zapisania
output_df = final_comparison[['Drużyna',
                             'Przewidziane Punkty',
                             'Rzeczywiste Punkty',
                             'Przewidziana Pozycja (z punktów)',
                             'Rzeczywista Pozycja']]

# Zmiana nazw kolumn na polskie
output_df.columns = ['Drużyna', 'Przewidziane Punkty', 'Rzeczywiste Punkty', 'Przewidziana Pozycja', 'Rzeczywista Pozycja']

# Zapis do CSV
output_df.to_csv("predykcja_sezon_2017_2018.csv", index=False, encoding='utf-8-sig')

print("\nZapisano predykcje do pliku 'predykcja_sezon_2017_2018.csv'.")

# Druga część zadania:
# Wizualizacja tabeli i wyników

# Wczytanie predykcji z pliku
pred = pd.read_csv("predykcja_sezon_2017_2018.csv")

# Filtrowanie drużyn zajmujących miejsca 1-20
filtered_pred = pred[pred['Rzeczywista Pozycja'] <= 20]

# Wyświetlenie tabeli w konsoli
print("\nPorównanie przewidywanych i rzeczywistych wyników (miejsca 1-20):\n")
print(filtered_pred.to_string(index=False))  # Przejrzysty format tabelaryczny

# Tabela porównawcza jako plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')
table_data = filtered_pred[['Drużyna', 'Przewidziane Punkty', 'Rzeczywiste Punkty', 'Przewidziana Pozycja', 'Rzeczywista Pozycja']]
columns = list(table_data.columns)
rows = table_data.values.tolist()

# Tworzenie tabeli w wykresie matplotlib
table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(columns))))

# Tytuł tabeli
plt.title("Porównanie przewidywanych i rzeczywistych wyników (miejsca 1-20)", fontsize=14, fontweight='bold')
plt.show()

# Histogram - porównanie punktów przewidzianych i rzeczywistych
fig, ax = plt.subplots(figsize=(15, 7))
filtered_sorted = filtered_pred.sort_values(by='Przewidziane Punkty', ascending=False)
bar_width = 0.4
indices = np.arange(len(filtered_sorted))

# Słupki
bars_pred = ax.bar(indices - bar_width / 2, filtered_sorted['Przewidziane Punkty'], bar_width, label='Przewidziane punkty', color='royalblue')
bars_real = ax.bar(indices + bar_width / 2, filtered_sorted['Rzeczywiste Punkty'], bar_width, label='Rzeczywiste punkty', color='orange')

# Etykiety osi
ax.set_xlabel("Drużyna")
ax.set_ylabel("Liczba punktów")
ax.set_title("Porównanie przewidzianych i rzeczywistych punktów drużyn sezonu 2017-2018", fontsize=14)
ax.set_xticks(indices)
ax.set_xticklabels(filtered_sorted['Drużyna'], rotation=45, ha='right')

# Legenda
ax.legend()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()

plt.show()