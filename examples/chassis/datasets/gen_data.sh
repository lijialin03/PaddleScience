# for seed in {0..9}
# do
#     python gen_data_inverse_ca.py -n "train" -seed $seed
# done

# python gen_data_inverse_ca.py -n "test" -seed 0

for seed in {10..19}
do
    python gen_data_inverse_ca.py -n "train_mult_type" -seed $seed -bs 10000
done