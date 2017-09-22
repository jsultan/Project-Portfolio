#Test on your own voice
test = pd.read_csv('my_voice.csv')

test_x = test.iloc[:, :-1].values
test_y = test.iloc[:,-1].values
test_y = gender_encoder.fit_transform(test_y)
test_x = scaler.fit_transform(test_x)

test_pred = model.predict(test_x)
test_pred = np.round(test_pred[:,1])
print(metrics.accuracy_score(test_pred,test_y))