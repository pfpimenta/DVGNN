# Tdrive dataset
python3 prediction-code/transportation/main.py --data_name Tdrive --max_epoch 60 --batch_size 4 --use_mixhop --static_graph --mixhop_neighborhood 2
python3 prediction-code/transportation/main.py --data_name Tdrive --max_epoch 60 --batch_size 4 --use_mixhop --static_graph --mixhop_neighborhood 3
python3 prediction-code/transportation/main.py --data_name Tdrive --max_epoch 60 --batch_size 4 --static_graph
python3 prediction-code/transportation/main.py --data_name Tdrive --max_epoch 60 --batch_size 4 --use_mixhop --dynamic_graph --mixhop_neighborhood 2
python3 prediction-code/transportation/main.py --data_name Tdrive --max_epoch 60 --batch_size 4 --use_mixhop --dynamic_graph --mixhop_neighborhood 3
python3 prediction-code/transportation/main.py --data_name Tdrive --max_epoch 60 --batch_size 4 --dynamic_graph

# Los_loop
python3 prediction-code/transportation/main.py --data_name Los_loop --max_epoch 60 --batch_size 4 --use_mixhop --static_graph --mixhop_neighborhood 2
python3 prediction-code/transportation/main.py --data_name Los_loop --max_epoch 60 --batch_size 4 --use_mixhop --static_graph --mixhop_neighborhood 3
python3 prediction-code/transportation/main.py --data_name Los_loop --max_epoch 60 --batch_size 4 --static_graph
python3 prediction-code/transportation/main.py --data_name Los_loop --max_epoch 60 --batch_size 4 --use_mixhop --dynamic_graph --mixhop_neighborhood 2
python3 prediction-code/transportation/main.py --data_name Los_loop --max_epoch 60 --batch_size 4 --use_mixhop --dynamic_graph --mixhop_neighborhood 3
python3 prediction-code/transportation/main.py --data_name Los_loop --max_epoch 60 --batch_size 4 --dynamic_graph

# PEMS08
python3 prediction-code/transportation/main.py --data_name PEMS08 --max_epoch 60 --batch_size 4 --use_mixhop --static_graph --mixhop_neighborhood 2
python3 prediction-code/transportation/main.py --data_name PEMS08 --max_epoch 60 --batch_size 4 --use_mixhop --static_graph --mixhop_neighborhood 3
python3 prediction-code/transportation/main.py --data_name PEMS08 --max_epoch 60 --batch_size 4 --static_graph
python3 prediction-code/transportation/main.py --data_name PEMS08 --max_epoch 60 --batch_size 4 --use_mixhop --dynamic_graph --mixhop_neighborhood 2
python3 prediction-code/transportation/main.py --data_name PEMS08 --max_epoch 60 --batch_size 4 --use_mixhop --dynamic_graph --mixhop_neighborhood 3
python3 prediction-code/transportation/main.py --data_name PEMS08 --max_epoch 60 --batch_size 4 --dynamic_graph
