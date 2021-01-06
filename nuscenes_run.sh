python nuscenes_generate_tracking_results.py val final \
&& \
python nuscenes_eval_tracking.py --output_dir ./results/nuscenes/val_final \
./results/nuscenes/val_final/nuscenes_val_final.json > \
./results/nuscenes/val_final/output.txt