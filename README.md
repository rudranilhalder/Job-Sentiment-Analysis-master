# Job Sentiment Analysis Web Application

This Flask web application is designed for analyzing job descriptions and providing insights into their sentiment. Users can sign up, log in, and submit job descriptions for analysis. The application leverages machine learning models to determine the sentiment of the job descriptions and provides a summary of key information extracted from the text.

## Features

- **User Authentication**: Securely manage user accounts with sign-up, login, and logout functionality.
- **Job Description Analysis**: Analyze job descriptions to determine their sentiment (positive, neutral, negative).
- **Summary Extraction**: Extract key information from job descriptions, including job title, key responsibilities, qualifications, skills required, job location, and salary.
- **Sentiment Score Calculation**: Combine sentiment analysis scores from multiple models to provide a more accurate sentiment score.
- **Data Storage**: Store analyzed job descriptions and their results in a MongoDB database for future reference.

## Technologies Used

- **Flask**: Python web framework for building the application.
- **MongoDB**: NoSQL database for storing user data and job analysis results.
- **Transformers**: Hugging Face's library for state-of-the-art natural language processing models.
- **Scipy**: For applying softmax function to model outputs.
- **Langchain**: For generating conversational responses based on job descriptions.
- **Sumy**: For summarizing job descriptions using Latent Semantic Analysis (LSA).

## Setup Instructions

1. Clone this repository: `git clone <repository-url>`
2. Set up MongoDB and configure the connection URI in `app.py`.
3. Set your personal OpenAI API key in the environment variable `OPENAI_API_KEY`.
4. Run the Flask application: `python app.py`
5. Access the application in your web browser at `http://localhost:5001`.

## Usage

1. Navigate to the `/` route to access the welcome page.
2. Sign up for an account if you're a new user.
3. Log in using your email and password.
4. Submit a job description for analysis by navigating to the `/analyze` route.
5. Review the analysis results, including sentiment score and extracted information, on the `/result` route.
6. Log out of your account by navigating to the `/logout` route.

## Contributing

Contributions to improve the application's functionality, performance, or documentation are welcome. Please follow the standard fork-and-pull request workflow.
