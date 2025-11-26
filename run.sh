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


for script in "train_FMGINO.py"; #"train_FMVBLLGINO.py" "train_GINO.py" "train_VBLLGINO.py" "train_MMGN.py";
do
    for t_points in 128 64 32 16;
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
            --model-dir $model_dir #&
    done
    wait
done

echo "DONE"