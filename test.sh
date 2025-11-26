seed=2025
data_dir="./MMGN_data/gst_data.npz"
num_epochs=500
x_points=32
y_points=32
xtarget_points=192
ytarget_points=288
# t_points=16
dropout_rate=0.25
model_dir="./saved-models-fmgino"
result_save_dir="./saved-predictions-fmgino"

for script in "test_FMGINO.py"; #"test_GINO.py" "test_MMGN.py" "test_VBLLGINO.py"; 
do
    for t_points in 16 32 64 128;
    do 
        python $script \
            --seed $seed \
            --data-dir $data_dir \
            --num-epochs $num_epochs \
            --x-points $x_points \
            --y-points $y_points \
            --dropout-rate $dropout_rate \
            --xtarget-points $xtarget_points \
            --ytarget-points $ytarget_points \
            --t-points $t_points \
            --model-dir $model_dir \
            --result-save-dir $result_save_dir #&
    done
    wait
done

echo "DONE"