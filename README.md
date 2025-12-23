# 🎬 MovieHub Analytics

**MovieHub Analytics** is a comprehensive Python-based project for analyzing movie ratings and metadata. This project demonstrates data manipulation, statistical analysis, and visualization techniques using real-world movie data from Kaggle.

## 📊 Project Overview

This project performs in-depth analysis of the `movies_metadata.csv` dataset containing information about 45,466 movies. The analysis covers everything from basic statistical computations to advanced genre-based filtering and temporal trend analysis.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/pandas-1.5%2B-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-green)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Status](https://img.shields.io/badge/status-active-success)

## 🚀 Features

### 📈 **Statistical Analysis**
- Comprehensive rating statistics (mean, median, standard deviation, percentiles)
- Distribution analysis and outlier detection
- Year-by-year performance metrics

### 🎭 **Genre Intelligence**
- Advanced parsing of complex genre data structures
- Genre popularity and performance rankings
- Cross-genre combination analysis

### 🔍 **Smart Filtering**
- Customizable rating thresholds
- Temporal filtering by release year
- Runtime-based movie categorization
- Multi-criteria quality filtering

### 📊 **Data Quality**
- Robust handling of missing values
- Type conversion and data normalization
- Memory-efficient processing



## ⚡ Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MovieHub-Analytics.git
cd MovieHub-Analytics
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**
   - Download `movies_metadata.csv` from [Kaggle](https://www.kaggle.com/rounakbanik/the-movies-dataset)
   - Place it in the `data/` directory

4. **Run the analysis**
```bash
python movie_analysis.py
```

## 📚 Usage Examples

### Basic Analysis
```python
# Run full analysis with interactive prompts
python movie_analysis.py

# Custom analysis with specific parameters
python movie_analysis.py --min_rating 7.0 --year 2015 --min_votes 1000
```

### Key Functionalities

1. **Rating Statistics**
```python
# Get overall rating distribution
Mean: 5.62
Median: 6.00
Standard Deviation: 1.92
```

2. **Genre Analysis**
```python
# Top performing genres
Documentary: 6.61 average rating
Animation: 6.24 average rating
Drama: 6.22 average rating
```

3. **Temporal Trends**
```python
# Movies by year
2015: 742 movies, average rating: 5.87
2014: 702 movies, average rating: 5.85
2013: 658 movies, average rating: 5.82
```

## 🔧 Configuration

Customize your analysis through interactive prompts or modify `config.py`:

```python
# Default analysis parameters
DEFAULT_SETTINGS = {
    'rating_threshold': 7.5,
    'min_votes': 1000,
    'target_year': 2015,
    'min_runtime': 60,
    'max_runtime': 180,
    'top_n': 10
}
```

## 📊 Analysis Modules

### 1. Data Exploration
- Dataset shape and structure
- Data types and missing values
- Basic descriptive statistics

### 2. Statistical Analysis
- Rating distribution analysis
- Outlier detection
- Correlation analysis between features

### 3. Genre Processing
- JSON-like genre parsing
- Genre frequency analysis
- Cross-genre relationship mapping

### 4. Temporal Analysis
- Yearly movie count trends
- Rating trends over time
- Seasonal performance patterns

### 5. Filtering & Sorting
- Custom rating filters
- Vote count thresholds
- Runtime-based categorization
- Multi-criteria sorting

## 📈 Sample Outputs

### Statistical Summary
```
=== RATING STATISTICS ===
Movies analyzed: 45,466
Mean rating: 5.62/10
Median rating: 6.00/10
Standard deviation: 1.92
Top 1% rating: 8.10/10
```

### Genre Performance
```
=== TOP 10 GENRES BY RATING ===
1. Documentary (6.61) - 1,045 movies
2. Animation (6.24) - 698 movies
3. Drama (6.22) - 14,756 movies
4. Music (6.15) - 412 movies
5. War (6.10) - 328 movies
```

### Yearly Trends
```
=== MOVIE PRODUCTION TRENDS ===
Peak year: 2015 (742 movies)
Best rated year: 1994 (6.52 average)
Most consistent: 2000-2010 (5.8-6.1 range)
```

## 🛠️ Dependencies

```txt
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0      # For visualization
seaborn>=0.12.0        # For enhanced plots
tabulate>=0.9.0        # For beautiful table formatting
python-dateutil>=2.8.2 # For date parsing
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Areas
- Add new analysis modules
- Improve data visualization
- Optimize performance for large datasets
- Add unit tests
- Enhance documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset) from Kaggle
- **Inspiration**: Various data science courses and tutorials
- **Tools**: Pandas, NumPy, and the open-source Python community



## 🎯 Project Goals

- [x] Complete basic statistical analysis
- [x] Implement genre parsing and analysis
- [x] Add interactive user input
- [x] Create comprehensive documentation
- [ ] Add data visualization dashboard
- [ ] Implement machine learning predictions
- [ ] Create REST API for data access
- [ ] Add support for streaming data

---

**Made with ❤️ for movie enthusiasts and data scientists**

*"Every movie tells a story, and every rating reveals a preference"*
