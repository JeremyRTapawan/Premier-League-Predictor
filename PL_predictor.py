## Enhanced Premier League Predictor with Interactive Team Selection and League Standings
## Improved version with team display, user interaction, and league winner prediction

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Load dataset
matches = pd.read_csv("matches.csv", index_col=0)

# Convert and prepare basic features
matches["date"] = pd.to_datetime(matches["date"])
matches["h/a"] = matches["venue"].astype("category").cat.codes
matches["opp"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.extract(r"(\d+)").astype(int)
matches["day"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype(int)

# Enhanced feature engineering
matches["rest_days"] = matches.groupby("team")["date"].diff().dt.days
matches["is_weekend"] = matches["day"].isin([5, 6]).astype(int)
matches["month"] = matches["date"].dt.month
matches["is_evening"] = (matches["hour"] >= 17).astype(int)

# Goal difference and points
matches["goal_diff"] = matches["gf"] - matches["ga"]
matches["points"] = matches["target"] * 3 + (matches["result"] == "D").astype(int)

print("Dataset loaded and basic features created")
print(f"Total matches: {len(matches)}")
print(f"Date range: {matches['date'].min()} to {matches['date'].max()}")


# Advanced rolling averages function
def advanced_rolling_averages(group, cols, window=5):
    """Calculate rolling averages with multiple windows for team form"""
    group = group.sort_values("date").copy()

    # Basic rolling stats - fill NaN with 0 for early matches
    for col in cols:
        if col in group.columns:
            group[f"{col}_rolling_{window}"] = group[col].rolling(window, closed='left').mean().fillna(0)
            group[f"{col}_rolling_3"] = group[col].rolling(3, closed='left').mean().fillna(0)

    # Form indicators - fill NaN with reasonable defaults
    group[f"points_rolling_{window}"] = group["points"].rolling(window, closed='left').mean().fillna(1.0)
    group[f"gd_rolling_{window}"] = group["goal_diff"].rolling(window, closed='left').mean().fillna(0.0)
    group[f"wins_rolling_{window}"] = group["target"].rolling(window, closed='left').mean().fillna(0.33)

    # Recent form streaks - fill with 0
    group["recent_wins_3"] = group["target"].rolling(3, closed='left').sum().fillna(0)
    group["recent_losses_3"] = (group["result"] == "L").astype(int).rolling(3, closed='left').sum().fillna(0)

    return group


# Calculate head-to-head records
def calculate_h2h_record(matches_df):
    """Calculate historical head-to-head win rate"""
    matches_df = matches_df.sort_values("date").copy()
    h2h_records = []

    for idx, row in matches_df.iterrows():
        team = row["team"]
        opponent = row["opponent"]
        date = row["date"]

        historical = matches_df[
            (matches_df["date"] < date) &
            (matches_df["team"] == team) &
            (matches_df["opponent"] == opponent)
            ]

        if len(historical) > 0:
            h2h_win_rate = historical["target"].mean()
        else:
            h2h_win_rate = 0.33

        h2h_records.append(h2h_win_rate)

    return h2h_records


print("Calculating head-to-head records...")
matches["h2h_record"] = calculate_h2h_record(matches)

# Apply rolling averages
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
print("Calculating rolling averages...")

matches_rolling = (
    matches.groupby("team", group_keys=False)
    .apply(lambda g: advanced_rolling_averages(g, cols, window=5))
)

matches_rolling.index = range(matches_rolling.shape[0])

# Check for any remaining NaN values and handle them
print(f"NaN values before cleaning: {matches_rolling.isnull().sum().sum()}")

for col in matches_rolling.columns:
    if matches_rolling[col].dtype in ['float64', 'int64'] and matches_rolling[col].isnull().any():
        if 'rolling' in col or col in ['h2h_record']:
            matches_rolling[col] = matches_rolling[col].fillna(matches_rolling[col].median())
        else:
            matches_rolling[col] = matches_rolling[col].fillna(0)

print(f"NaN values after cleaning: {matches_rolling.isnull().sum().sum()}")
print(f"Matches after rolling calculations: {len(matches_rolling)}")

# Define feature sets
basic_predictors = ["h/a", "opp", "hour", "day", "month", "is_weekend", "is_evening", "h2h_record"]

rolling_predictors = []
for col in cols:
    rolling_predictors.extend([f"{col}_rolling_5", f"{col}_rolling_3"])

form_predictors = ["points_rolling_5", "gd_rolling_5", "wins_rolling_5", "recent_wins_3", "recent_losses_3"]

all_predictors = basic_predictors + rolling_predictors + form_predictors

print(f"Total features: {len(all_predictors)}")


# Time-based train/test split
def create_train_test_split(data, cutoff_date='2022-01-01'):
    """Create proper temporal split"""
    train = data[data["date"] < cutoff_date].copy()
    test = data[data["date"] >= cutoff_date].copy()

    print(f"Training set: {len(train)} matches")
    print(f"Test set: {len(test)} matches")
    print(f"Training date range: {train['date'].min()} to {train['date'].max()}")
    print(f"Test date range: {test['date'].min()} to {test['date'].max()}")

    return train, test


train, test = create_train_test_split(matches_rolling)


# Model evaluation function
def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """Comprehensive model evaluation"""
    print(f"\n=== {model_name} Performance ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred):.3f}")

    if y_prob is not None:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None else None
    }


# Train multiple models
def train_models(X_train, y_train, X_test, y_test, show_details=False):
    """Train and evaluate multiple models"""

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No training or test data available!")
        return {}, {}

    if show_details:
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                                min_samples_split=10, random_state=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=6,
                                                        random_state=1),
        'Logistic Regression': LogisticRegression(random_state=1, max_iter=1000)
    }

    results = {}
    predictions = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training AI models...")
    for name, model in models.items():
        if show_details:
            print(f"\nTraining {name}...")

        try:
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            results[name] = evaluate_model(y_test, y_pred, y_prob, name) if show_details else {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            predictions[name] = {'pred': y_pred, 'prob': y_prob, 'model': model,
                                 'scaler': scaler if name == 'Logistic Regression' else None}

        except Exception as e:
            if show_details:
                print(f"Error training {name}: {str(e)}")
            continue

    return results, predictions


# Train models
print("Setting up AI prediction system...")

all_results, all_predictions = train_models(
    train[all_predictors], train["target"],
    test[all_predictors], test["target"],
    show_details=False  # Set to True if you want detailed training info
)

# Get best model
best_model_name = max(all_results.keys(), key=lambda x: all_results[x]['roc_auc'])
best_model = all_predictions[best_model_name]['model']
best_scaler = all_predictions[best_model_name]['scaler']

print(f"✅ AI system ready! Using {best_model_name} model")
print(f"📊 Model accuracy: {all_results[best_model_name]['roc_auc']:.1%}")


# League Standings Functions
def calculate_league_table(season_start_date='2021-08-01', season_end_date='2022-05-31'):
    """Calculate league table for a specific season"""
    season_matches = matches[
        (matches['date'] >= season_start_date) &
        (matches['date'] <= season_end_date)
        ].copy()

    # Initialize league table
    teams = season_matches['team'].unique()
    table = []

    for team in teams:
        team_matches = season_matches[season_matches['team'] == team]

        # Calculate stats
        played = len(team_matches)
        wins = len(team_matches[team_matches['result'] == 'W'])
        draws = len(team_matches[team_matches['result'] == 'D'])
        losses = len(team_matches[team_matches['result'] == 'L'])
        goals_for = team_matches['gf'].sum()
        goals_against = team_matches['ga'].sum()
        goal_diff = goals_for - goals_against
        points = wins * 3 + draws

        table.append({
            'Team': team,
            'Played': played,
            'Won': wins,
            'Drawn': draws,
            'Lost': losses,
            'GF': goals_for,
            'GA': goals_against,
            'GD': goal_diff,
            'Points': points,
            'PPG': points / played if played > 0 else 0  # Points per game
        })

    # Convert to DataFrame and sort by points, then goal difference
    table_df = pd.DataFrame(table)
    table_df = table_df.sort_values(['Points', 'GD'], ascending=[False, False]).reset_index(drop=True)
    table_df['Position'] = range(1, len(table_df) + 1)

    return table_df


def predict_league_winner(model, scaler=None, prediction_date='2022-03-01', remaining_matches=10):
    """Predict league winner based on current form and remaining fixtures"""

    # Get current league table
    table = calculate_league_table(season_end_date=prediction_date)

    # Get teams and their current form
    teams = table['Team'].tolist()
    raw_predictions = {}

    for team in teams:
        # Get team's recent form
        team_recent = matches_rolling[
            (matches_rolling["team"] == team) &
            (matches_rolling["date"] < prediction_date)
            ].tail(5)

        if len(team_recent) == 0:
            continue

        # Calculate team strength metrics
        recent_ppg = team_recent['points'].mean()
        recent_win_rate = team_recent['target'].mean()
        recent_gd = team_recent['goal_diff'].mean()

        # Get current position and points
        current_pos = table[table['Team'] == team]['Position'].iloc[0]
        current_points = table[table['Team'] == team]['Points'].iloc[0]
        current_ppg = table[table['Team'] == team]['PPG'].iloc[0]

        # Predict points from remaining matches (simplified prediction)
        predicted_points_per_match = (recent_ppg * 0.6 + current_ppg * 0.4)
        predicted_additional_points = predicted_points_per_match * remaining_matches
        predicted_total_points = current_points + predicted_additional_points

        # Title probability based on multiple factors - raw score
        position_factor = max(0, (21 - current_pos) / 20)  # Higher for top teams
        form_factor = recent_win_rate
        consistency_factor = min(1, current_ppg / 3)  # How consistent they've been
        points_factor = min(1, current_points / 90)  # Based on current points

        # Calculate raw title score (not probability yet)
        raw_title_score = (position_factor * 0.3 + form_factor * 0.25 +
                           consistency_factor * 0.25 + points_factor * 0.2)

        raw_predictions[team] = {
            'current_position': current_pos,
            'current_points': current_points,
            'predicted_total_points': round(predicted_total_points, 1),
            'recent_form_ppg': round(recent_ppg, 2),
            'raw_title_score': raw_title_score,
            'recent_win_rate': round(recent_win_rate, 3)
        }

    # Normalize probabilities to sum to 1 (100%)
    total_raw_score = sum(pred['raw_title_score'] for pred in raw_predictions.values())

    predictions = {}
    for team, pred in raw_predictions.items():
        normalized_prob = pred['raw_title_score'] / total_raw_score if total_raw_score > 0 else 0

        predictions[team] = {
            'current_position': pred['current_position'],
            'current_points': pred['current_points'],
            'predicted_total_points': pred['predicted_total_points'],
            'recent_form_ppg': pred['recent_form_ppg'],
            'title_probability': round(normalized_prob, 3),
            'recent_win_rate': pred['recent_win_rate']
        }

    return predictions, table


def display_league_table(table, top_n=20):
    """Display league table in a nice format"""
    print("\n" + "=" * 90)
    print("PREMIER LEAGUE TABLE")
    print("=" * 90)
    print(
        f"{'Pos':<4} {'Team':<20} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<4} {'GA':<4} {'GD':<5} {'Pts':<4} {'PPG':<5}")
    print("-" * 90)

    for _, row in table.head(top_n).iterrows():
        pos = row['Position']

        # Add position indicators
        if pos == 1:
            pos_str = "🏆1"
        elif pos <= 4:
            pos_str = f"🔴{pos}"  # Champions League
        elif pos <= 6:
            pos_str = f"🟠{pos}"  # Europa League
        elif pos >= 18:
            pos_str = f"🔻{pos}"  # Relegation
        else:
            pos_str = f"{pos:>3}"

        print(f"{pos_str:<4} {row['Team']:<20} {row['Played']:<3} {row['Won']:<3} "
              f"{row['Drawn']:<3} {row['Lost']:<3} {row['GF']:<4} {row['GA']:<4} "
              f"{row['GD']:>+5} {row['Points']:<4} {row['PPG']:<5.2f}")


def display_title_predictions(predictions):
    """Display title race predictions"""
    # Sort by title probability
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1]['title_probability'], reverse=True)

    print("\n" + "=" * 80)
    print("🏆 PREMIER LEAGUE TITLE RACE PREDICTIONS")
    print("=" * 80)
    print(f"{'Team':<18} {'Pos':<4} {'Pts':<4} {'Pred':<5} {'Form':<5} {'Win%':<5} {'Title%':<7}")
    print("-" * 80)

    for i, (team, pred) in enumerate(sorted_predictions[:10]):
        title_pct = pred['title_probability'] * 100

        # Add emoji for top contenders
        if i == 0:
            team_display = f"👑 {team[:15]}"
        elif i < 3:
            team_display = f"🥇 {team[:15]}"
        elif i < 6:
            team_display = f"⭐ {team[:15]}"
        else:
            team_display = f"   {team[:15]}"

        print(f"{team_display:<18} {pred['current_position']:<4} {pred['current_points']:<4} "
              f"{pred['predicted_total_points']:<5} {pred['recent_form_ppg']:<5} "
              f"{pred['recent_win_rate'] * 100:<4.0f}% {title_pct:<6.1f}%")

    # Show key insights
    top_team = sorted_predictions[0]
    print(f"\n🎯 Most likely champion: {top_team[0]} ({top_team[1]['title_probability'] * 100:.1f}% chance)")

    top_3 = sorted_predictions[:3]
    combined_prob = sum([pred[1]['title_probability'] for pred in top_3]) * 100
    print(f"📊 Top 3 teams account for {combined_prob:.1f}% of title chances")

    # Verify probabilities sum to 100%
    total_prob = sum([pred[1]['title_probability'] for pred in sorted_predictions]) * 100
    print(f"✅ All probabilities sum to {total_prob:.1f}%")


# Display all teams function
def display_all_teams():
    """Display all teams in the league with numbering"""
    teams = sorted(matches["team"].unique())

    print("\n" + "=" * 60)
    print("PREMIER LEAGUE TEAMS")
    print("=" * 60)

    # Display teams in 2 columns
    mid_point = len(teams) // 2
    for i in range(mid_point):
        left_team = f"{i + 1:2d}. {teams[i]}"
        if i + mid_point < len(teams):
            right_team = f"{i + mid_point + 1:2d}. {teams[i + mid_point]}"
            print(f"{left_team:<30} {right_team}")
        else:
            print(left_team)

    return teams


# Get team choice function
def get_team_choice(teams, prompt_text):
    """Get valid team choice from user"""
    while True:
        try:
            print(f"\n{prompt_text}")
            choice = input("Enter team number or name: ").strip()

            # Check if it's a number
            if choice.isdigit():
                team_idx = int(choice) - 1
                if 0 <= team_idx < len(teams):
                    return teams[team_idx]
                else:
                    print(f"Please enter a number between 1 and {len(teams)}")
                    continue

            # Check if it's a team name (partial match)
            matching_teams = [team for team in teams if choice.lower() in team.lower()]
            if len(matching_teams) == 1:
                return matching_teams[0]
            elif len(matching_teams) > 1:
                print(f"Multiple teams match '{choice}': {', '.join(matching_teams)}")
                print("Please be more specific.")
                continue
            else:
                print(f"Team '{choice}' not found. Please try again.")
                continue

        except KeyboardInterrupt:
            print("\nGoodbye!")
            return None
        except Exception as e:
            print(f"Invalid input: {e}")
            continue


# Enhanced match prediction function
def predict_match_outcome(team1, team2, date, model, scaler=None):
    """Predict outcome for a specific match"""
    try:
        # Get recent stats for both teams before the match date
        team1_recent = matches_rolling[
            (matches_rolling["team"] == team1) &
            (matches_rolling["date"] < date)
            ].tail(1)

        team2_recent = matches_rolling[
            (matches_rolling["team"] == team2) &
            (matches_rolling["date"] < date)
            ].tail(1)

        if len(team1_recent) == 0 or len(team2_recent) == 0:
            return {"error": "Insufficient historical data"}

        # Create feature vectors
        features_home = team1_recent[all_predictors].iloc[0].copy()
        features_away = team2_recent[all_predictors].iloc[0].copy()

        # Adjust for venue
        features_home['h/a'] = 1  # Home
        features_away['h/a'] = 0  # Away

        # Get opponent encoding
        team_mapping = dict(zip(matches["opponent"].astype("category").cat.categories,
                                matches["opponent"].astype("category").cat.codes.unique()))
        features_home['opp'] = team_mapping.get(team2, 0)
        features_away['opp'] = team_mapping.get(team1, 0)

        # Make predictions
        if scaler:
            features_home_scaled = scaler.transform([features_home])
            features_away_scaled = scaler.transform([features_away])
            home_win_prob = model.predict_proba(features_home_scaled)[0][1]
            away_win_prob = model.predict_proba(features_away_scaled)[0][1]
        else:
            home_win_prob = model.predict_proba([features_home])[0][1]
            away_win_prob = model.predict_proba([features_away])[0][1]

        # Normalize probabilities
        total_win_prob = home_win_prob + away_win_prob
        if total_win_prob > 1:
            home_win_prob = home_win_prob / total_win_prob * 0.8
            away_win_prob = away_win_prob / total_win_prob * 0.8

        draw_prob = max(0, 1 - home_win_prob - away_win_prob)

        return {
            'home_team': team1,
            'away_team': team2,
            'home_win_prob': round(home_win_prob, 3),
            'draw_prob': round(draw_prob, 3),
            'away_win_prob': round(away_win_prob, 3),
            'predicted_winner': team1 if home_win_prob > away_win_prob else team2
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


# Display prediction results
def display_prediction(prediction):
    """Display prediction results in a nice format"""
    if "error" in prediction:
        print(f"\nError: {prediction['error']}")
        return

    print("\n" + "=" * 60)
    print("MATCH PREDICTION")
    print("=" * 60)
    print(f"Match: {prediction['home_team']} (H) vs {prediction['away_team']} (A)")
    print("-" * 60)
    print(f"🏠 {prediction['home_team']} Win:    {prediction['home_win_prob']:.1%}")
    print(f"🤝 Draw:               {prediction['draw_prob']:.1%}")
    print(f"✈️  {prediction['away_team']} Win:    {prediction['away_win_prob']:.1%}")
    print("-" * 60)
    print(f"🏆 Predicted Winner: {prediction['predicted_winner']}")
    print("=" * 60)


# Get team statistics
def get_team_stats(team_name, num_matches=10):
    """Get recent statistics for a team"""
    team_data = matches_rolling[matches_rolling["team"] == team_name].tail(num_matches)

    if len(team_data) == 0:
        return None

    recent_data = team_data.tail(5)

    stats = {
        'team': team_name,
        'recent_matches': len(recent_data),
        'wins': recent_data['target'].sum(),
        'draws': (recent_data['result'] == 'D').sum(),
        'losses': (recent_data['result'] == 'L').sum(),
        'goals_for': recent_data['gf'].sum(),
        'goals_against': recent_data['ga'].sum(),
        'avg_goals_for': recent_data['gf'].mean(),
        'avg_goals_against': recent_data['ga'].mean(),
        'win_rate': recent_data['target'].mean(),
        'points': recent_data['points'].sum()
    }

    return stats


# Display team statistics
def display_team_stats(stats):
    """Display team statistics in a nice format"""
    if stats is None:
        print("No data available for this team.")
        return

    print(f"\n📊 {stats['team']} - Recent Form (Last {stats['recent_matches']} matches)")
    print("-" * 50)
    print(f"Record: {stats['wins']}W - {stats['draws']}D - {stats['losses']}L")
    print(f"Goals: {stats['goals_for']} scored, {stats['goals_against']} conceded")
    print(f"Average: {stats['avg_goals_for']:.1f} goals for, {stats['avg_goals_against']:.1f} goals against")
    print(f"Win Rate: {stats['win_rate']:.1%}")
    print(f"Points: {stats['points']}/15 possible")


# Main interactive function
def run_interactive_predictor():
    """Main interactive function for predictions"""
    teams = display_all_teams()

    print(f"\n🤖 AI Prediction System Ready!")
    print(f"📊 Using {best_model_name} model (Accuracy: {all_results[best_model_name]['roc_auc']:.1%})")

    while True:
        print("\n" + "=" * 60)
        print("PREMIER LEAGUE PREDICTION SYSTEM")
        print("=" * 60)
        print("1. Predict match outcome")
        print("2. View team statistics")
        print("3. Show league table & title race")
        print("4. Show all teams")
        print("5. Exit")

        try:
            choice = input("\nSelect an option (1-5): ").strip()

            if choice == '1':
                # Match prediction
                home_team = get_team_choice(teams, "Select HOME team:")
                if home_team is None:
                    break

                away_team = get_team_choice(teams, "Select AWAY team:")
                if away_team is None:
                    break

                if home_team == away_team:
                    print("Teams cannot play against themselves!")
                    continue

                # Use a recent date for prediction
                prediction_date = pd.to_datetime('2022-02-01')
                prediction = predict_match_outcome(home_team, away_team, prediction_date, best_model, best_scaler)
                display_prediction(prediction)

            elif choice == '2':
                # Team statistics
                team = get_team_choice(teams, "Select team to view statistics:")
                if team is None:
                    break

                stats = get_team_stats(team)
                display_team_stats(stats)

            elif choice == '3':
                # League table and title predictions
                print("\n🔍 Analyzing league standings and title race...")

                # Calculate current table
                table = calculate_league_table(season_end_date='2022-03-01')
                display_league_table(table)

                # Predict title winner
                predictions, _ = predict_league_winner(best_model, best_scaler)
                display_title_predictions(predictions)

                input("\nPress Enter to continue...")

            elif choice == '4':
                # Show all teams
                display_all_teams()

            elif choice == '5':
                print("Thank you for using the Premier League Predictor!")
                break

            else:
                print("Invalid choice. Please select 1-5.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


# Run the interactive predictor
if __name__ == "__main__":
    run_interactive_predictor()