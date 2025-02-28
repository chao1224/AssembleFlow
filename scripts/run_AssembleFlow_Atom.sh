cd ..

seed_list=(0 42 123)

for seed in "${seed_list[@]}"; do
    output_folder=output/COD_5000/seed_"$seed"_atom
    mkdir -p "$output_folder"

    python main.py \
    --dataset=COD_5000 --model=AssembleFlow_Atom --emb_dim=128 --hidden_dim=128 --num_layers=5 --num_convs=5 --cutoff=5 --cluster_cutoff=50 \
    --num_timesteps=200 --model_3d=PaiNN --PaiNN_radius_cutoff=5 --PaiNN_gamma=3.25 --epochs=2000 --seed="$seed" --lr=1e-4 --lr_scheduler=CosineAnnealingLR \
    --num_workers=0 --batch_size=8 --test_batch_size=16 --subsample_ratio=0.1 --alpha_rotation=10 --alpha_translation=1 --print_every_epoch=10 \
    --inference_num_repeat=1 --inference_interval=10 --inference_step_size=1. \
    --verbose --output_model_dir="$output_folder"
done





for seed in "${seed_list[@]}"; do
    output_folder=output/COD_10000/seed_"$seed"_atom
    mkdir -p "$output_folder"

    python main.py \
    --dataset=COD_10000 --model=AssembleFlow_Atom --emb_dim=128 --hidden_dim=128 --num_layers=5 --num_convs=5 --cutoff=5 --cluster_cutoff=50 \
    --num_timesteps=200 --model_3d=PaiNN --PaiNN_radius_cutoff=5 --PaiNN_gamma=3.25 --epochs=2000 --seed="$seed" --lr=1e-4 --lr_scheduler=CosineAnnealingLR \
    --num_workers=0 --batch_size=8 --test_batch_size=16 --subsample_ratio=0.1 --alpha_rotation=10 --alpha_translation=1 --print_every_epoch=10 \
    --inference_num_repeat=1 --inference_interval=10 --inference_step_size=1. \
    --verbose --output_model_dir="$output_folder"

done





for seed in "${seed_list[@]}"; do
    output_folder=output/COD/seed_"$seed"_atom
    mkdir -p "$output_folder"

    python main.py \
    --dataset=COD --model=AssembleFlow_Atom --emb_dim=128 --hidden_dim=128 --num_layers=5 --num_convs=5 --cutoff=5 --cluster_cutoff=50 \
    --num_timesteps=200 --model_3d=PaiNN --PaiNN_radius_cutoff=5 --PaiNN_gamma=3.25 --epochs=1000 --seed="$seed" --lr=1e-4 --lr_scheduler=CosineAnnealingLR \
    --num_workers=0 --batch_size=8 --test_batch_size=16 --subsample_ratio=0.1 --alpha_rotation=10 --alpha_translation=1 --print_every_epoch=10 \
    --inference_num_repeat=1 --inference_interval=10 --inference_step_size=1. \
    --verbose --output_model_dir="$output_folder"

done
