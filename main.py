import os
import time
import random
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

from AssembleFlow.datasets import CrystallizationDatasetCOD
from AssembleFlow.models import AssembleFlow
from AssembleFlow.evaluate_AssembleFlow import evaluate_crystallization


def get_keys(data):
    if callable(getattr(data, 'keys', None)):
        return data.keys()
    else:
        return data.keys


def random_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    num_mols = len(dataset)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[: int(frac_train * num_mols)]
    valid_idx = all_idx[
        int(frac_train * num_mols) : int(frac_valid * num_mols)
        + int(frac_train * num_mols)
    ]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols) :]

    print("len of train: {}, val: {}, test: {}".format(len(train_idx), len(valid_idx), len(test_idx)))
    print("train_idx", train_idx[:5])
    print("valid_idx", valid_idx[:5])
    print("test_idx", test_idx[:5])

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset


def save_model(save_best):
    if args.output_model_dir is not None:
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model_best.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def load_model():
    output_model_path = os.path.join(args.output_model_dir, "model_best.pth")
    model_weight = torch.load(output_model_path)
    model.load_state_dict(model_weight["model"])
    return


def train(loader):
    model.train()
    loss_accum = 0

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    batch_total_count = len(loader)
    if args.subsample_ratio < 1:
        batch_total_count = int(args.subsample_ratio * batch_total_count)
    batch_count = 0
        
    for batch_id, batch in enumerate(L):
        try:
            if batch.x.shape[0] > 10000:
                # print("=====")
                continue
            batch.x = batch.x[:, 0]
            
            batch = batch.to(device)
            loss = model(batch)

            if torch.isnan(loss):
                print("=== invalid", batch_id)
                continue

            # # TODO: debugging only
            # print(batch_id, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += loss.cpu().detach().item()

            if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
                lr_scheduler.step(epoch - 1 + step / num_iters)

            batch_count += 1
            if batch_count >= batch_total_count:
                break

        except Exception as e:
            print(batch_id, e)

    loss_accum /= batch_count
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR", "AlphaFoldLRScheduler"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)

    return loss_accum


@torch.no_grad()
def eval(loader):
    model.eval()
    loss_accum = 0

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    batch_count = 0

    for batch_id, batch in enumerate(L):
        try:
            batch = batch.to(device)
            batch.x = batch.x[:, 0]

            loss = model(batch)

            if torch.isnan(loss):
                print("=== invalid")
                continue

            optimizer.zero_grad()
            loss_accum += loss.cpu().detach().item()
            batch_count += 1
        
        except Exception as e:
            print("eval", batch_id, e)

    loss_accum /= batch_count
    return loss_accum


def inference_and_evaluate(loader, dataset, model, args):
    def repeat_data_list(batch_data, num_repeat):
        raw_data_list = Batch.to_data_list(batch_data)
        data_list = []
        for data_idx, data in enumerate(raw_data_list):
            for repeat_idx in range(num_repeat):
                neo_key_mapping = {}
                for key in get_keys(data):
                    neo_key_mapping[key] = data[key]
                N = neo_key_mapping["x"].shape[0]
                repeat_idx_to_idx = [data_idx]
                neo_key_mapping["repeat_idx_to_idx"] = torch.LongTensor(repeat_idx_to_idx)

                data_duplicate = Data.from_dict(neo_key_mapping)
                data_list.append(data_duplicate)
        return Batch.from_data_list(data_list)

    model.eval()

    num_repeat = args.inference_num_repeat

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    packing_matching_atom_wise_list, packing_matching_mass_center_list, packing_collision_list, packing_separation_list = [], [], [], []
    
    # Save data for each batch
    for batch_idx, batch in enumerate(L):
        batch.x = batch.x[:, 0]
        batch = batch.to(device)
        num_graph = batch.batch.max().item() + 1
        print("batch_idx", batch_idx, "\t\tnum graph", num_graph)
        # print("batch", batch.batch)

        repeated_batch = repeat_data_list(batch, num_repeat).to(device)  # graph repeated with num_repeat times
        print("batch", repeated_batch, repeated_batch.repeat_idx_to_idx)

        interval_list, pos_list, _, _ = model.position_inference(
            data=repeated_batch, inference_interval=args.inference_interval, step_size=args.inference_step_size, verbose=args.verbose)

        for interval, pos in zip(interval_list, pos_list):
            key = "pred_positions_tid_{}".format(interval)
            repeated_batch[key] = pos
            repeated_batch._slice_dict[key] = repeated_batch._slice_dict['x'].clone()  # This addes the slices to position, which is the same as x
            repeated_batch._inc_dict[key] = repeated_batch._inc_dict['x'].clone()  # This addes the slices to position, which is the same as x

        repeated_graph_list = Batch.to_data_list(repeated_batch)
        assert len(repeated_graph_list) == num_repeat * num_graph

        for i in range(num_graph):
            for r in range(num_repeat):
                index = i * num_repeat + r
                graph = repeated_graph_list[index]
                assert graph.repeat_idx_to_idx[0].item() == i
        
        if args.output_model_dir is not None:
            output_folder = os.path.join(args.output_model_dir, "output_data_size_{}_repeat_{}".format(args.test_batch_size, args.inference_num_repeat))
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, "batch_{}.pt".format(batch_idx))
            print("saving to {}".format(output_path))
            data, slices = dataset.collate(repeated_graph_list)
            torch.save((data, slices), output_path)

            atom_wise_list, mass_center_list, collision_list, separation_list = evaluate_crystallization(
                output_path,
                num_timesteps=args.num_timesteps)
            assert len(atom_wise_list) == num_graph
            assert len(mass_center_list) == num_graph

            packing_matching_atom_wise_list.extend(atom_wise_list)
            packing_matching_mass_center_list.extend(mass_center_list)
            packing_collision_list.extend(collision_list)
            packing_separation_list.extend(separation_list)

        print("\n\n")

    print("packing matching (atom-wise): {}".format(np.mean(packing_matching_atom_wise_list)))
    print("packing matching (mass-center): {}".format(np.mean(packing_matching_mass_center_list)))

    matching_threshold = 100
    valid_packing_matching_atom_wise_list = [x for x in packing_matching_atom_wise_list if not np.isnan(x) and not np.isinf(x) and x <= matching_threshold]
    valid_packing_matching_mass_center_list = [x for x in packing_matching_mass_center_list if not np.isnan(x) and not np.isinf(x) and x <= matching_threshold]

    print("packing matching (atom-wise): {}".format(np.mean(valid_packing_matching_atom_wise_list)))
    print("packing matching (mass-center): {}".format(np.mean(valid_packing_matching_mass_center_list)))
    print("valid ratio (atom-wise): {}".format( len(valid_packing_matching_atom_wise_list) / len(packing_matching_atom_wise_list) ))
    print("valid ratio (mass-center): {}".format( len(valid_packing_matching_mass_center_list) / len(packing_matching_mass_center_list) ))
    print("collision: {}".format(np.mean(packing_collision_list)))
    print("separation: {}".format(np.mean(packing_separation_list)))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # about seed and basic info
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)

    # about optimization strategies
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--subsample_ratio", type=float, default=0.1)
    parser.add_argument("--print_every_epoch", type=int, default=1)
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--alpha_rotation", type=float, default=10)
    parser.add_argument("--alpha_translation", type=float, default=1)
    parser.add_argument("--data_root", type=str, default="./data/COD")

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=False)

    # for PaiNN
    parser.add_argument("--PaiNN_radius_cutoff", type=float, default=5.0)
    parser.add_argument("--PaiNN_n_interactions", type=int, default=3)
    parser.add_argument("--PaiNN_n_rbf", type=int, default=20)
    parser.add_argument("--PaiNN_readout", type=str, default="mean", choices=["mean", "add"])
    parser.add_argument("--PaiNN_gamma", type=float, default=3.25)

    # for AssembleFlow
    parser.add_argument("--model", type=str, default="AssembleFlow_Atom")
    parser.add_argument(
        "--model_3d",
        type=str,
        default="PaiNN",
    )
    parser.add_argument("--dataset", type=str, default="COD")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--cutoff", type=float, default=10.)
    parser.add_argument("--cluster_cutoff", type=float, default=50.)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--num_gaussians", type=int, default=20)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=3.25)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_convs", type=int, default=2)
    parser.add_argument("--num_head", type=int, default=8)
    parser.add_argument("--num_sub_layers", type=int, default=3) # only for 03

    parser.add_argument('--inference_num_repeat', type=int, default=1)
    parser.add_argument('--inference_interval', type=int, default=10)
    parser.add_argument('--inference_step_size', type=float, default=1.)

    parser.add_argument('--output_model_dir', type=str, default=None)
    parser.add_argument("--load_pretrained", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset == "COD":
        subset = -1
        dataset = CrystallizationDatasetCOD(args.data_root, subset=subset)
    elif args.dataset == "COD_5000":
        subset = 5000
        dataset = CrystallizationDatasetCOD(args.data_root, subset=subset)
    elif args.dataset == "COD_10000":
        subset = 10000
        dataset = CrystallizationDatasetCOD(args.data_root, subset=subset)

    # TODO: will modify the splitting later
    train_dataset, valid_dataset, test_dataset = random_split(dataset, 0.8, 0.1, 0.1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    ##### setup Flow-Matching model #####
    node_class = 119
    
    model = AssembleFlow(
        emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, cutoff=args.cutoff, cluster_cutoff=args.cluster_cutoff, node_class=node_class,
        num_timesteps=args.num_timesteps,
        args=args).to(device)
    print("model", model)

    # set up optimizer
    model_param_group = [
        {"params": model.parameters(), "lr": args.lr}
    ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "AlphaFoldLRScheduler":
        lr_scheduler = AlphaFoldLRScheduler(
            optimizer, max_lr=args.lr, 
            warmup_no_steps=100,
            start_decay_after_n_steps=500,
            decay_every_n_steps=500)
        print("Apply lr scheduler AlphaFoldLRScheduler")
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.epochs, eta_min=1e-4
        )
        print("Apply lr scheduler CosineAnnealingWarmRestarts")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        print("Apply lr scheduler ReduceLROnPlateau")
    elif args.lr_scheduler == "None":
        pass
    else:
        print("lr scheduler {} is not included.".format(args.lr_scheduler))
    
    if not args.load_pretrained:
        # NOTE: do training only when not loading from pretrained
        optimal_val_loss = 1e10

        #### learning #####
        for e in range(1, 1+args.epochs):
            start_time = time.time()
            print("\nepoch {}".format(e))
            loss_accum = train(train_dataloader)
            print("loss accum: {}".format(loss_accum))

            if e % args.eval_interval == 0:
                val_loss = eval(valid_dataloader)
                print("val loss: {}".format(val_loss))
                if val_loss < optimal_val_loss:
                    optimal_val_loss = val_loss
                    save_model(save_best=True)

            print("Took \t{} seconds\n".format(time.time() - start_time))

        print("\n")
        save_model(save_best=False)

    if args.output_model_dir is None:
        exit()

    print("\nLoading ...")
    load_model()

    ##### inference #####
    inference_and_evaluate(test_dataloader, test_dataset, model, args)
