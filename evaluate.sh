# parser.add_argument("--seed", type=int, required=True)
# parser.add_argument("--data-dir", type=str, required=True)
# parser.add_argument("--num-epochs", type=int, default=500)
# parser.add_argument("--x-points", type=int, required=True)
# parser.add_argument("--y-points", type=int, required=True)
# parser.add_argument("--dropout-rate", type=float, default=0.0)
# parser.add_argument("--xtarget-points", type=int, required=True)
# parser.add_argument("--ytarget-points", type=int, required=True)
# parser.add_argument("--t-points", type=int, required=True)
# parser.add_argument("--model-dir", type=str, required=True)

cd $HOME
cd ./MMGN

source ../anaconda3/bin/activate
conda activate hyperdm

seed=730007773
# data_dir="./MMGN_data/gst_data.npz"
num_epochs=500
x_points=32
y_points=32
xtarget_points=192
ytarget_points=288
# t_points=16
# dropout_rate=0.25
# model_dir="./saved-models-lightweight-gino"
result_save_dir="./saved-predictions" #-lightweight-gino"
eval_save_dir="./saved-evaluations-wcrps"

for script in "evaluate.py";
do
    for t_points in 16 32 64 128;
    do 
        python $script \
            --seed $seed \
            --x-points $x_points \
            --y-points $y_points \
            --xtarget-points $xtarget_points \
            --ytarget-points $ytarget_points \
            --t-points $t_points \
            --result-save-dir $result_save_dir \
            --eval-save-dir $eval_save_dir #&
    done
    # wait
done

wait

echo "DONE"