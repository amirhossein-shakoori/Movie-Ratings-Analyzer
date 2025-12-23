"""
Movie Ratings Analysis Project
==============================

This project analyzes movie ratings dataset using NumPy and Pandas.
Focus is on statistical analysis, filtering, and grouping of movie ratings data.

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

# ============================================================================
# 📦 IMPORT LIBRARIES
# ============================================================================
import numpy as np  # For numerical computations and statistics
import pandas as pd  # For data analysis and manipulation
from datetime import datetime  # For date operations
import re  # For string pattern matching
import ast  # For parsing Python structures from strings
from collections import Counter  # For counting elements
from display import display_simple # For display data
# ============================================================================
# 📁 LOAD AND EXPLORE DATASET
# ============================================================================

# Load dataset from CSV file
df = pd.read_csv('movies_metadata.csv', low_memory=False)

# Display basic dataset information
print(f"Shape: {df.shape}")  # DataFrame dimensions (rows and columns)
print(f"Columns: {df.columns.tolist()}")  # List of column names

print("=== DATASET INFO ===")
print(f"Total movies: {len(df)}")  # Total number of movies
print(f"Total columns: {len(df.columns)}")  # Total number of columns

print("\nFirst 5 rows:")  # Display first 5 rows of data
display_simple(df.head())

print("\nData types:")  # Display data types of columns
print(df.dtypes.value_counts())  # Count different data types




# ============================================================================
# 🧮 TASK 1.1: BASIC STATISTICS WITH NUMPY
# ============================================================================

# Convert vote_average column to NumPy array for statistical calculations
ratings = df['vote_average'].dropna().astype(float).values  # Remove NaN values

print("=== RATING STATISTICS USING NUMPY ===")
print(f"Number of valid ratings: {len(ratings)}")  # Count of valid ratings

# Get user input for statistics threshold
try:
    rating_threshold = float(input("\nEnter minimum rating to analyze (default 0): ") or 0)
except ValueError:
    rating_threshold = 0
    print("Invalid input. Using default value 0.")

# Filter ratings above threshold
filtered_ratings = ratings[ratings >= rating_threshold]

print(f"\nAnalyzing {len(filtered_ratings)} movies with rating >= {rating_threshold}")

# Calculate descriptive statistics using NumPy functions
stats = {
    "Mean": np.mean(filtered_ratings),  # Average
    "Median": np.median(filtered_ratings),  # Median
    "Standard Deviation": np.std(filtered_ratings),  # Standard deviation
    "Variance": np.var(filtered_ratings),  # Variance
    "Minimum": np.min(filtered_ratings),  # Minimum value
    "Maximum": np.max(filtered_ratings),  # Maximum value
    "25th Percentile": np.percentile(filtered_ratings, 25),  # First quartile
    "50th Percentile (Median)": np.percentile(filtered_ratings, 50),  # Second quartile (median)
    "75th Percentile": np.percentile(filtered_ratings, 75),  # Third quartile
    "Range": np.ptp(filtered_ratings)  # Range (max - min)
}

# Display calculated statistics with proper formatting
for stat_name, stat_value in stats.items():
    print(f"{stat_name}: {stat_value:.4f}")  # Display with 4 decimal places

# ============================================================================
# 🎯 TASK 1.2: FILTERING WITH NUMPY
# ============================================================================

# Get user input for rating threshold
try:
    RATING_THRESHOLD = float(input("\nEnter rating threshold (default 7.5): ") or 7.5)
except ValueError:
    RATING_THRESHOLD = 7.5
    print("Invalid input. Using default value 7.5.")

# Create boolean mask for movies with rating above threshold
high_rating_mask = ratings > RATING_THRESHOLD
# Find indices of high-rated movies
high_rated_indices = np.where(high_rating_mask)[0]

print(f"\n=== MOVIES WITH RATING > {RATING_THRESHOLD} ===")
print(f"Number of movies: {len(high_rated_indices)}")  # Count of high-rated movies
print(f"Percentage: {(len(high_rated_indices)/len(ratings)*100):.2f}%")  # Percentage

# Get user input for number of top movies to display
try:
    num_top_movies = int(input("\nHow many top movies to display? (default 10): ") or 10)
except ValueError:
    num_top_movies = 10
    print("Invalid input. Using default value 10.")

# Extract information for high-rated movies
high_rated_movies = []
for idx in high_rated_indices[:num_top_movies]:  # Only first N movies
    # Find movie index in original dataframe
    movie_idx = df[df['vote_average'] == ratings[idx]].index[0]
    # Extract movie title and rating
    title = df.loc[movie_idx, 'title']
    rating = ratings[idx]
    high_rated_movies.append((title, rating))

print(f"\nTop {num_top_movies} high-rated movies:")
for i, (title, rating) in enumerate(high_rated_movies, 1):
    print(f"{i}. {title}: {rating:.1f}")

# ============================================================================
# 📅 EXTRACT YEAR FROM RELEASE DATE
# ============================================================================

def extract_year(date_string):
    """
    Extract year from date string in various formats
    
    Args:
        date_string (str): String containing date
        
    Returns:
        int or None: Extracted year or None if error
    """
    if pd.isna(date_string):  # Check for NaN
        return None
    
    date_string = str(date_string)  # Convert to string
    
    # List of expected date formats
    date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y']
    
    # Try to parse date with each format
    for fmt in date_formats:
        try:
            date_obj = datetime.strptime(date_string, fmt)
            return date_obj.year  # Return year
        except ValueError:
            continue  # Try next format on error
    
    # If standard formats fail, extract year with regex
    year_match = re.search(r'\b(\d{4})\b', date_string)
    if year_match:
        return int(year_match.group(1))  # Convert to integer
    
    return None  # Return None if unsuccessful

# Apply year extraction function to release_date column
df['release_year'] = df['release_date'].apply(extract_year)

print(f"\nYears extracted successfully!")
print(f"Unique years: {df['release_year'].nunique()}")  # Count of unique years
print(f"Year range: {df['release_year'].min()} - {df['release_year'].max()}")  # Time range

# ============================================================================
# 🎬 FILTER MOVIES FROM SPECIFIC YEAR
# ============================================================================

# Get user input for target year
try:
    TARGET_YEAR = int(input("\nEnter year to filter movies (default 2015): ") or 2015)
except ValueError:
    TARGET_YEAR = 2015
    print("Invalid input. Using default value 2015.")

# Create mask for movies from target year
year_mask = df['release_year'] == TARGET_YEAR
# Filter dataframe based on mask
movies_target_year = df[year_mask]

print(f"\n=== MOVIES FROM YEAR {TARGET_YEAR} ===")
print(f"Number of movies: {len(movies_target_year)}")  # Count of movies from target year
print(f"Average rating: {movies_target_year['vote_average'].mean():.2f}")  # Average rating

# Sort movies from target year by rating
top_target_year_movies = movies_target_year.sort_values('vote_average', ascending=False)[['title', 'vote_average', 'vote_count']].head(10)
print(f"\nTop 10 movies from {TARGET_YEAR}:")
display_simple(top_target_year_movies)

# ============================================================================
# 📊 TASK 2.1: PANDAS DATA HANDLING
# ============================================================================

print("\n=== PANDAS DATA HANDLING ===")

print("\n1. Data Types:")  # Display data types
print(df.dtypes)  # Data type of each column

print("\n2. Missing Values Analysis:")  # Analyze missing values
# Count NaN values in each column
missing_data = df.isnull().sum()
# Calculate percentage of missing values
missing_percentage = (missing_data / len(df)) * 100

# Create dataframe for displaying missing values
missing_df = pd.DataFrame({
    'Missing_Count': missing_data,  # Count of missing values
    'Missing_Percentage': missing_percentage  # Percentage of missing values
})

print("Top 10 columns with missing values:")  # 10 columns with most missing values
display_simple(missing_df.sort_values('Missing_Percentage', ascending=False).head(10))

# ============================================================================
# 🧹 HANDLING MISSING DATA
# ============================================================================

print("\n=== HANDLING MISSING DATA ===")

# Store original dataframe shape for comparison
original_shape = df.shape

print("\n1. Before handling missing values:")  # Before handling missing values
print(f"Total missing values: {df.isnull().sum().sum()}")  # Total missing values

# Handle missing values in numerical columns
numeric_cols = ['runtime', 'vote_average', 'vote_count', 'popularity']
for col in numeric_cols:
    if col in df.columns:
        # Convert to numeric and handle errors
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Calculate median
        median_val = df[col].median()
        # Replace NaN values with median
        df[col].fillna(median_val, inplace=True)
        print(f"Filled {col} with median: {median_val:.2f}")  # Report

# Handle missing values in categorical columns
categorical_cols = ['original_language', 'status', 'homepage']
for col in categorical_cols:
    if col in df.columns:
        # Replace NaN with 'Unknown'
        df[col].fillna('Unknown', inplace=True)
        print(f"Filled {col} with 'Unknown'")  # Report

# Handle missing values in movie title
df['title'].fillna(df['original_title'], inplace=True)  # Use original title
df['title'].fillna('Unknown Title', inplace=True)  # Otherwise 'Unknown Title'

print("\n2. After handling missing values:")  # After handling missing values
print(f"Total missing values: {df.isnull().sum().sum()}")  # Remaining missing values
print(f"DataFrame shape: {df.shape} (Original: {original_shape})")  # Compare dimensions

# ============================================================================
# 🔍 SORTING AND FILTERING MOVIES
# ============================================================================

print("\n=== SORTING AND FILTERING MOVIES ===")

# Get user input for number of top movies to display
try:
    num_top_display = int(input("\nEnter number of top movies to display (default 20): ") or 20)
except ValueError:
    num_top_display = 20
    print("Invalid input. Using default value 20.")

print(f"\n1. Top {num_top_display} Highest Rated Movies:")
# Sort descending by rating and display top N
top_movies = df.sort_values('vote_average', ascending=False)[['title', 'vote_average', 'vote_count', 'release_year']].head(num_top_display)
display_simple(top_movies)

# Get user input for minimum votes threshold
try:
    MIN_VOTES = int(input(f"\nEnter minimum votes for filtering (default 1000): ") or 1000)
except ValueError:
    MIN_VOTES = 1000
    print("Invalid input. Using default value 1000.")

print(f"\n2. Top {num_top_display} Highest Rated Movies (with at least {MIN_VOTES} votes):")
# Filter movies with minimum MIN_VOTES votes
filtered_top_movies = df[df['vote_count'] >= MIN_VOTES]
# Sort and display top N
filtered_top_movies = filtered_top_movies.sort_values('vote_average', ascending=False)[['title', 'vote_average', 'vote_count', 'release_year']].head(num_top_display)
display_simple(filtered_top_movies)

# Get user input for runtime range
try:
    MIN_RUNTIME = int(input("\nEnter minimum runtime in minutes (default 60): ") or 60)
    MAX_RUNTIME = int(input("Enter maximum runtime in minutes (default 180): ") or 180)
except ValueError:
    MIN_RUNTIME = 60
    MAX_RUNTIME = 180
    print("Invalid input. Using default values 60-180.")

print(f"\n3. Movies with runtime between {MIN_RUNTIME} and {MAX_RUNTIME} minutes:")
# Filter movies with runtime in specified range
runtime_filtered = df[(df['runtime'] >= MIN_RUNTIME) & (df['runtime'] <= MAX_RUNTIME)]
print(f"Number of movies: {len(runtime_filtered)}")  # Count of filtered movies
print(f"Average rating: {runtime_filtered['vote_average'].mean():.2f}")  # Average rating

# ============================================================================
# 🎭 PARSING GENRES COLUMN
# ============================================================================

print("\n=== PARSING GENRES COLUMN ===")

print("\n=== DATASET INFO ===")  # Dataset information
print(f"Total movies: {len(df)}")  # Total movies
print(f"Columns with genres data type: {df['genres'].dtype}")  # Genres column data type

# Display sample of genres data
print("\n=== SAMPLE GENRES DATA ===")
for i in range(5):
    print(f"Row {i}: {df['genres'].iloc[i]}")  # First 5 samples

def parse_genres_fast(genres_entry):
    """
    Fast parsing of genres string to list of genre names
    
    Args:
        genres_entry: Input can be string, list, or NaN
    
    Returns:
        list: List of genre names
    """
    if pd.isna(genres_entry):  # Check for NaN
        return []
    
    # If already a list (from previous processing)
    if isinstance(genres_entry, list):
        # Extract genre names from list of dictionaries
        return [genre.get('name', '') for genre in genres_entry 
                if isinstance(genre, dict) and 'name' in genre]
    
    # If it's a string
    if isinstance(genres_entry, str):
        # Check if string resembles list of dictionaries
        if genres_entry.strip().startswith('[') and genres_entry.strip().endswith(']'):
            try:
                # Convert string to Python object
                parsed = ast.literal_eval(genres_entry)
                if isinstance(parsed, list):
                    # Extract genre names
                    return [genre.get('name', '') for genre in parsed 
                           if isinstance(genre, dict) and 'name' in genre]
            except (ValueError, SyntaxError):  # Handle parsing errors
                return []
    
    return []  # Return empty list if no match

print("Parsing genres...")  # Start genres parsing
# Apply genres parsing function to genres column
df['genres_list'] = df['genres'].apply(parse_genres_fast)

# Count number of genres per movie
df['num_genres'] = df['genres_list'].apply(len)

print(f"\n=== GENRES PARSING RESULTS ===")  # Parsing results
print(f"Movies with parsed genres: {df['genres_list'].notna().sum()}")  # Movies with parsed genres
print(f"Movies with 0 genres: {(df['num_genres'] == 0).sum()}")  # Movies with no genres
print(f"Movies with 1 genre: {(df['num_genres'] == 1).sum()}")  # Movies with 1 genre
print(f"Movies with 2+ genres: {(df['num_genres'] >= 2).sum()}")  # Movies with 2+ genres

print(f"\nSample parsed genres:")  # Sample of parsed genres
for i in range(3):
    print(f"Row {i}: {df['genres_list'].iloc[i]}")  # First 3 samples

# ============================================================================
# 📈 EXTRACTING ALL UNIQUE GENRES
# ============================================================================

print("\n=== EXTRACTING ALL UNIQUE GENRES ===")

# Create flattened list of all genres
all_genres = []
# Create Counter for counting genres
genre_counter = Counter()

# Iterate over genres list of each movie
for genres in df['genres_list']:
    all_genres.extend(genres)  # Add genres to master list
    genre_counter.update(genres)  # Update count

# Create list of unique genres
unique_genres = list(set(all_genres))

print(f"Total genre occurrences: {len(all_genres)}")  # Total genre occurrences
print(f"Unique genres found: {len(unique_genres)}")  # Count of unique genres

print(f"\nTop 20 most common genres:")  # 20 most common genres
for genre, count in genre_counter.most_common(20):
    print(f"  {genre}: {count:,} movies ({count/len(df)*100:.1f}%)")  # Display with formatting

# ============================================================================
# 🎬 FILTER MOVIES BY GENRE
# ============================================================================

print("\n=== FILTER MOVIES BY GENRE ===")

# Create copy of dataframe for processing
df_clean = df.copy()

# Create dictionary for fast genre to movies lookup
genre_to_movies = {}
for idx, genres in enumerate(df_clean['genres_list']):
    for genre in genres:
        if genre not in genre_to_movies:
            genre_to_movies[genre] = []  # Create new list for genre
        genre_to_movies[genre].append(idx)  # Add movie index

print(f"Created lookup for {len(genre_to_movies)} genres")  # Report number of genres

def get_movies_by_genre_fast(genre_name):
    """
    Fast filtering of movies by genre using pre-built dictionary
    
    Args:
        genre_name (str): Name of genre to filter by
    
    Returns:
        pd.DataFrame: DataFrame of movies with specified genre
    """
    if genre_name not in genre_to_movies:
        return pd.DataFrame()  # Return empty DataFrame if genre not found
    indices = genre_to_movies[genre_name]  # Get movie indices
    return df_clean.iloc[indices]  # Return corresponding rows

# Get user input for number of top genres to analyze
try:
    TOP_N_GENRES = int(input("\nEnter number of top genres to analyze (default 10): ") or 10)
except ValueError:
    TOP_N_GENRES = 10
    print("Invalid input. Using default value 10.")

# Analyze top N genres
top_genres = sorted(genre_to_movies.items(), key=lambda x: len(x[1]), reverse=True)[:TOP_N_GENRES]

print(f"\nTop {TOP_N_GENRES} Genres Analysis:")  # Top genres analysis
print("-" * 50)  # Separator line

for genre, indices in top_genres:
    # Extract movies of current genre
    genre_movies = df_clean.iloc[indices]
    # Calculate average rating and average votes
    avg_rating = genre_movies['vote_average'].mean()
    avg_votes = genre_movies['vote_count'].mean()
    
    print(f"\n{genre}:")  # Display genre name
    print(f"  Movies: {len(indices):,}")  # Number of movies
    print(f"  Avg Rating: {avg_rating:.2f}")  # Average rating
    print(f"  Avg Votes: {avg_votes:.0f}")  # Average number of votes
    
    # Get user input for number of top movies to display per genre
    try:
        top_movies_count = int(input(f"How many top {genre} movies to display? (default 3): ") or 3)
    except ValueError:
        top_movies_count = 3
        print("Invalid input. Using default value 3.")
    
    # Top movies in this genre
    top_movies = genre_movies.nlargest(top_movies_count, 'vote_average')[['title', 'vote_average']]
    for _, movie in top_movies.iterrows():
        print(f"  - {movie['title'][:40]}...: {movie['vote_average']:.1f}")  # Display truncated title

# ============================================================================
# 🔍 SORTING AND FILTERING (ADVANCED)
# ============================================================================

print("\n=== SORTING AND FILTERING ===")

# Get user input for minimum votes threshold for ranking
try:
    MIN_VOTES_FOR_RANKING = int(input("\nEnter minimum votes for ranking (default 100): ") or 100)
except ValueError:
    MIN_VOTES_FOR_RANKING = 100
    print("Invalid input. Using default value 100.")

print(f"\n1. Top Movies (with at least {MIN_VOTES_FOR_RANKING} votes):")
# Filter and sort movies with minimum votes
top_movies = df_clean[df_clean['vote_count'] >= MIN_VOTES_FOR_RANKING]
top_movies_sorted = top_movies.nlargest(10, 'vote_average')[['title', 'vote_average', 'vote_count']]
display_simple(top_movies_sorted)

# Get user input for high quality criteria
try:
    high_rating_threshold = float(input("\nEnter rating threshold for high quality movies (default 7.5): ") or 7.5)
    high_votes_threshold = int(input("Enter votes threshold for high quality movies (default 1000): ") or 1000)
except ValueError:
    high_rating_threshold = 7.5
    high_votes_threshold = 1000
    print("Invalid input. Using default values 7.5 and 1000.")

print(f"\n2. High Quality Movies (Rating > {high_rating_threshold}, Votes > {high_votes_threshold}):")
# Filter high quality movies
high_quality = df_clean[
    (df_clean['vote_average'] > high_rating_threshold) & 
    (df_clean['vote_count'] > high_votes_threshold)
]
print(f"  Found: {len(high_quality):,} movies")  # Count of found movies
if len(high_quality) > 0:
    display_simple(high_quality[['title', 'vote_average', 'vote_count']].head())

# Get user input for genre combination
genre1 = input("\nEnter first genre for combination (default 'Action'): ") or 'Action'
genre2 = input("Enter second genre for combination (default 'Adventure'): ") or 'Adventure'

print(f"\n3. {genre1} & {genre2} Movies:")
# Filter movies with both genres
genre_combination = df_clean[
    df_clean['genres_list'].apply(lambda x: genre1 in x and genre2 in x)
]
print(f"  Found: {len(genre_combination):,} movies")  # Count of found movies
if len(genre_combination) > 0:
    display_simple(genre_combination[['title', 'vote_average', 'genres_list']].head())

# ============================================================================
# 📊 TASK 2.2: GROUPING AND AGGREGATION
# ============================================================================

print("\n=== GROUPING AND AGGREGATION ===")

print("\n1. Average Rating by Genre (All genres):")  # Average rating by genre

# Get user input for minimum movies per genre
try:
    min_movies_per_genre = int(input("\nEnter minimum movies per genre for analysis (default 10): ") or 10)
except ValueError:
    min_movies_per_genre = 10
    print("Invalid input. Using default value 10.")

# Create list of genre statistics
genre_stats = []
for genre, indices in genre_to_movies.items():
    # Only consider genres with minimum movies
    if len(indices) >= min_movies_per_genre:
        genre_movies = df_clean.iloc[indices]
        # Calculate statistics
        avg_rating = genre_movies['vote_average'].mean()
        median_rating = genre_movies['vote_average'].median()
        genre_stats.append({
            'Genre': genre,  # Genre name
            'Movie Count': len(indices),  # Number of movies
            'Avg Rating': avg_rating,  # Average rating
            'Median Rating': median_rating,  # Median rating
            'Total Votes': genre_movies['vote_count'].sum()  # Total votes
        })

# Convert to dataframe
genre_stats_df = pd.DataFrame(genre_stats)
# Sort descending by average rating
genre_stats_df = genre_stats_df.sort_values('Avg Rating', ascending=False)

print(f"\nAnalyzed {len(genre_stats_df)} genres with at least {min_movies_per_genre} movies")
print(f"Top 15 genres by average rating:")

# Get user input for number of genres to display
try:
    num_genres_display = int(input(f"How many genres to display? (default 15): ") or 15)
except ValueError:
    num_genres_display = 15
    print("Invalid input. Using default value 15.")

display_simple(genre_stats_df.head(num_genres_display))

# ============================================================================
# 🏁 END OF ANALYSIS
# ============================================================================
print("\n" + "="*50)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*50)