import pandas as pd
import numpy as np
import dataReader
import GaussianProcess as gp
import kernels 

#Get previous returns and volumes
def getFeatures(prices_df):
	#Look at lags of 1-7, -1 for future return (to predict)
	for i in [-1, 1, 2, 3, 4, 5, 6, 7]: 
		j = str(i)
		prices_df['Close_' + j] = prices_df['Close'].shift(i)
		prices_df['Volume_' + j] = prices_df['Volume'].shift(i)
		prices_df['Return_' + j] = prices_df['Close_' + j]/prices_df['Close'] - 1
		prices_df['Volume_Pct_' + j] = prices_df['Volume_' + j]/prices_df['Volume'] - 1

	#Obtain relevant rows and columns
	prices_df = prices_df.dropna(axis = 0)
	col_names = prices_df.columns[(prices_df.columns.str.contains('Return_')) | (prices_df.columns.str.contains('Volume_Pct_'))]
	prices_df = prices_df.loc[:, col_names]
	return prices_df

#Evaluate RMSE
def rmse(pred_returns, actual_returns):
	return np.sqrt(((pred_returns - actual_returns) ** 2).mean())

#Evaluate whether test points are contained within confidence intervals
def checkConfidenceIntervals(pred_returns, actual_returns, pred_var):
	lower_bound = pred_returns - 2 * pred_var
	upper_bound = pred_returns + 2 * pred_var
	count = 0
	for i in range(len(actual_returns)):
		if lower_bound[i] < actual_returns[i] and upper_bound[i] > actual_returns[i]:
			count += 1
	return count/actual_returns.shape[0]


#Evaluate a given ticker
def evaluate_ticker(ticker, kernel, training_size = 0.8):
	if not isinstance(kernel, kernels.Kernel):
		raise ValueError("GP kernel must be a valid kernel.")
	d = dataReader.DataReader()
	data = d.get_ticker(ticker)
	clean = getFeatures(data)

	#Split into training and test
	total_obs = clean.shape[0]
	train_test_split = int(total_obs * training_size)
	train_data = clean.iloc[:train_test_split, ]
	test_data = clean.iloc[train_test_split:, ]

	X_train = train_data.iloc[:, 1:].values
	Y_train = train_data.iloc[:, 0].values
	X_test = test_data.iloc[:, 1:].values
	Y_test = test_data.iloc[:, 0].values

	#Evaluate
	gp_obj = gp.GaussianProcess(kernel)
	Y_pred, sigma_pred = gp_obj.derive_conditional(X_train, Y_train, X_test)

	#Get confidence intervals and rmse 
	pred_var = np.sqrt(np.diag(sigma_pred))
	err = rmse(Y_pred, Y_test)
	conf = checkConfidenceIntervals(Y_pred, Y_test, pred_var)
	return err, conf

#Check some rmse and confidence interval results for various tickers using two basic kernels
if __name__ == '__main__':
	SquareExpKernel = kernels.SquareExpKernel(sigma = 1, length_scale = 1)
	OrnsteinKernel = kernels.OrnsteinKernel(sigma = 1, length_scale = 1)
	ticker_list = ['AAPL', 'MSFT', 'FB', 'GOOG']
	for ticker in ticker_list:
		print("Ticker: " + ticker)
		SquareExpKernelResults = evaluate_ticker(ticker, SquareExpKernel)
		OrnsteinKernelResults = evaluate_ticker(ticker, OrnsteinKernel)
		print("RMSE with SquareExpKernel: %0.4f" %(SquareExpKernelResults[0]))
		print("SquareExpKernel confidence interval contains %0.4f of actual predictions" %(SquareExpKernelResults[1]))
		print("RMSE with OrnsteinKernel: %0.4f" %(OrnsteinKernelResults[0]))
		print("OrnsteinKernel confidence interval contains %0.4f of actual predictions" %(OrnsteinKernelResults[1]))


