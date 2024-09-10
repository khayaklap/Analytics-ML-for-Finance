## Introduction 

Hello, my name is Khai-Lap. I am a 4th year undergraduate student majoring in finance and analytics at McGill University. Having taken boring investment banking focused classes with a heavy emphasis on Excel, I realized that my passion lies in a more logical & computational approach to finance, where insights are derived by understanding the underlying data rather than making baseless assumptions. I have not looked back since, and hope to share what I have learned so far. I am certainly not a high-level coder, nor do I think I will be one in the near future, as you will see.

## Objective

This repository demonstrates applications of different subsets of machine learning in trading, portfolio optimization, and risk management, using Python. Beyond tuning hyperparameters and making predictions, we will dig into the maths behind the convergence process of each algorithm, as well as its strengths & weaknesses. This will allow us to apply relevant feature engineering techniques to input clean data into our models and make reliable deductions. Methodologies covered are supervised learning, unsupervised learning, deep learning, and reinforcement learning.

## Necessary Libraries

Libraries include Pandas for data manipulation, NumPy for linear algebra computations, SciPy for statistical tests, CVXPY for optimization, Statsmodels for time series analysis, Scikit-Learn for preprocessing & machine learning, PyTorch for deep learning, SpaCy & NLTK for natural language processing, and Matplotlib & Seaborn for visualization. For larger datasets exceeding 1 million rows, PySpark is preferred over Pandas. Although advanced knowledge of each library is not required, it is recommended that you read the documentation and know key methods & attributes.

## Data Sources

Datasets are retrieved from multiple finance data providers with varying degrees of reliability, such as the Bloomberg Terminal, Yahoo Finance & Compustat. Some are selected for their speed and ease of extraction, others for their completeness. As the saying goes, garbage in, garbage out. Models' output are only as reliable as their training data. For your convenience, datasets in this repository have been cleaned and loaded as xlsx or csv files in each project's Data folder. Please note that each project's code has been written specifically to run on its corresponding data.

## Disclaimer

Projects in this repository do not constitute investment advice in any shape or form. There is no guarantee that results achieved will be replicable in the future, especially during periods of market downturns. Machine learning models are not only prone to performance deterioration over time, but also achieve prediction accuracy at the expense of interpretability (Black Box). They are particularly vulnerable to overfitting and data biases. Therefore, you should do your own due diligence, make your own interpretations of findings, and bear responsibility for your own risk & losses.

## Contact Info

You can reach me via email at khai.vuong@mail.mcgill.ca. Alternatively, you can send a LinkedIn connect request to [Khai Lap Vuong](https://www.linkedin.com/in/khai-lap-vuong/). All discussions on the covered topics above or suggestions on how to improve the code are welcome!
