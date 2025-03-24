House Price Prediction Web App
This is a web application built using Flask that predicts house prices based on features like house age, distance to the nearest MRT station, and the number of nearby convenience stores. The model is trained using Linear Regression on a dataset.
 Table of Contents
Project Introduction

Technologies Used

Project Description

Getting Started

Pre-requisites

Installation

License

Project Introduction
The aim of this project is to build a house price prediction model that estimates the price of a real estate property based on factors like:

House age

Distance to the nearest MRT station

Number of convenience stores nearby

The goal is to provide a reliable tool for buyers, real estate agents, and developers to make informed property decisions.

We have developed a web application that allows users to input property details and receive price predictions instantly.

🛠️ Technologies Used
Python → Programming language

Pandas → Data processing and cleaning

NumPy → Numerical operations

Scikit-learn → Model building and evaluation

Flask → Backend server

HTML, CSS, JavaScript → Frontend web interface

Joblib → Model serialization

Project Description
This project applies Data Science and Machine Learning fundamentals, including:

Data Cleaning: Preprocessing the real estate dataset.

Feature Engineering: Selecting and transforming relevant features.

Model Building: Using Linear Regression to predict house prices.

Model Evaluation: Assessing accuracy using Mean Squared Error (MSE) and R² score.

The project includes a Flask web server that handles prediction requests and displays the results on the web interface.

💻 Frontend Features:
User-friendly form to input property details (age, distance, and number of stores).

Real-time prediction displayed on the same page.

Clean and responsive design with HTML, CSS, and JavaScript.
