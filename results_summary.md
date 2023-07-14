# results summary

Vanilla GCN with c_out=20:
* Average time per epoch: 7.21 seconds
* Minumum MAE: 4.243
* Minimum RMSE: 5.9289
* Minimum MAPE: 822.3481

MixHop with c_out=20 and adjacency_powers = [0, 1, 2]:
* Minumum MAE: 4.2155
* Minimum RMSE: 5.8993
* Minimum MAPE: 1388.2932

Vanilla GCN with c_out=64:
* Average time per epoch: 9.67 seconds
* Minumum MAE: 4.1171
* Minimum RMSE: 5.7544
* Minimum MAPE: 1346.6374

MixHop with c_out=64 and adjacency_powers = [0, 1, 2]:
* Average time per epoch: 31.74 seconds
* Minumum MAE: 4.1326
* Minimum RMSE: 5.764
* Minimum MAPE: 1257.4511

MixHop with c_out=64 and adjacency_powers = [0, 1, 2, 3]:
* Average time per epoch: 47.03 seconds
* Minumum MAE: 4.16
* Minimum RMSE: 5.8445
* Minimum MAPE: 1173.146
