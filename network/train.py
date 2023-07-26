# %%
import net
from network.pl_trainer import pl_train

from logging_utils.argparse_utils import add_bool_arg
import argparse
from spherical_coords_comparator import generate_validation_training_hists, generate_accuracy_hists

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ------------------ device-based parameters -----------------
    parser.add_argument("--device_name", type=str, default="cuda:0")

    # ------------------ imu-based parameters -----------------
    parser.add_argument("--inertial_window_length", type=float, default=3.5, help="desired length of inertial window in seconds")
    parser.add_argument("--nominal_imu_frequency", type=float, default=400., help="Hz")
    parser.add_argument("--sampling_frequency", type=float, default=20., help="Hz")
    parser.add_argument("--velocity_mag_threshold", type=float, default=0.0, help="m/s")

    # ------------------ model hyperparameters -----------------
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=3)
    parser.add_argument("--target_learning_rate", type=float, default=1e-4)
    parser.add_argument("--residual_block_depth", type=int, default=3)

    # ------------------ training parameters -----------------
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)

    # ------------------ modelling parameters -----------------
    add_bool_arg(parser, name="use_gt", default=False)
    add_bool_arg(parser, name="validate", default=True)
    add_bool_arg(parser, name="self_augment", default=True)
    add_bool_arg(parser, name="train_raw_velocity", default=True)
    parser.add_argument("--frame_target", type=str, default="current_k_gravity_aligned")
    parser.add_argument("--initial_orientation_error", type=float, default=0.0, help="rads")

    # ------------------ directory parameters -----------------
    parser.add_argument("--model_output", type=str, default="network_models/")
    parser.add_argument("--tensorboard_dir", type=str, default="runs/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")

    # ------------------ file parameters -----------------
    parser.add_argument("--root_dir", type=str, default="/home/angad/learned_quad_inertial/learned_quad_inertial_odometry/")
    parser.add_argument("--training_set_loc", type=str, default="network/splits/train.txt")
    parser.add_argument("--val_set_loc", type=str, default="network/splits/val_original_formatting.txt")
    parser.add_argument("--test_set_loc", type=str, default="network/splits/test.txt")
    parser.add_argument("--ground_truth_output_name", type=str, default="data.hdf5")
    parser.add_argument("--start_idx", type=int, default=50)

    # ------------------ model naming -----------------
    parser.add_argument("--model_name", type=str, default="final_velReg_augment_3_5")

    # ------------------ model visualization parameters -----------------
    parser.add_argument("--loss_vis_thresh", type=float, default=0.5)

    args = parser.parse_args()

    pl_train(args)

# %%
