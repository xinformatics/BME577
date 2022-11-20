num_background = 5
index = 0
background_ts, test_ts = train_x[:num_background], test_x[index:index + 5]






ts_phi_1 = np.zeros((len(test_ts),test_ts.shape[1], test_ts.shape[2]))
for i in range(len(test_ts)):
    window_len = 15
    gtw = StationaryTimeWindow(model, window_len, B_ts=background_ts, test_ts=test_ts[i:i+1], model_type='lstm')
    ts_phi_1[i,:,:] = gtw.shap_values()[0]


    