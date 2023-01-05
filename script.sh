# LSTM based method
python train_net.py --max_epoch 5 --cfg 230106_lstm_base
python inference.py --save_numpy --draw_graph --draw_gt --cfg 230106_lstm_base
python inverse_test.py --cfg 230106_lstm_base

python train_net.py --max_epoch 5 --cfg 230106_lstm_base




# training new
python train_net.py --max_epoch 5 --cfg base_m
python train_net.py --max_epoch 5 --cfg base_m_full

python test_net.py --cfg ./cfgs_backup2/base_m 
python test_net.py --cfg base_m 

python inference.py --save_numpy --draw_graph --draw_gt --cfg multi_modal_task
python inference.py --save_numpy --draw_graph --draw_gt --cfg multi_modal_task_full

python inverse_test.py --cfg base_m
python inverse_test.py --cfg base_m_full

# inverse test
python train_net.py --gpu 5 --max_epoch 300 \
 --cfg base_m --k_fold 

python test_net.py --cfg base_m --k_fold  

python inverse_test.py --cfg base_m --k_fold 0
python inverse_test.py --cfg multi_modal --k_fold 0

# CODE TEST
python train_net.py --gpu 5 --max_epoch 300 \
 --cfg base_m

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg base_c
 
python train_net.py --gpu 5 --max_epoch 300 \
 --cfg base_d

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg multi_modal

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg multi_task

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg multi_modal_task


python test_net.py --cfg base_m
python test_net.py --cfg base_c
python test_net.py --cfg base_d

python test_net.py --cfg multi_modal
python test_net.py --cfg multi_task
python test_net.py --cfg multi_modal_task


python inference.py --save_numpy --draw_graph --draw_gt --gpu 5 --cfg base_m
python inference.py --save_numpy --draw_graph --draw_gt --gpu 5 --cfg base_c
python inference.py --save_numpy --draw_graph --draw_gt --gpu 5 --cfg base_d

python inference.py --save_numpy --draw_graph --draw_gt --gpu 5 --cfg multi_modal
python inference.py --save_numpy --draw_graph --draw_gt --gpu 5 --cfg multi_task
python inference.py --save_numpy --draw_graph --draw_gt --gpu 5 --cfg multi_modal_task


# old reproduce
python train_net.py --gpu 5 --max_epoch 300 \
 --cfg base_m_old

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg base_c_old

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg base_d_old

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg multi_modal_old

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg multi_task_old

python train_net.py --gpu 5 --max_epoch 300 \
 --cfg multi_modal_task_old

python test_net.py --cfg base_m_old
python test_net.py --cfg base_c_old
python test_net.py --cfg base_d_old

python test_net.py --cfg multi_modal_old
python test_net.py --cfg multi_task_old
python test_net.py --cfg multi_modal_task_old
