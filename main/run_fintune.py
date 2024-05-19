ROOT_PATH = '/root/autodl-tmp/no-codingRNA-pretrain/main'
device = 'cuda:0'
from asyncio import current_task
from arguments import arg_parse
import sys
import pickle
sys.path.append(ROOT_PATH)
import argparse
import glob
import json
import logging
import os
import re
import shutil
import random
from multiprocessing import Pool
from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from src.transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DNATokenizer,
    get_linear_schedule_with_warmup,
)
from src.transformers import glue_compute_metrics as compute_metrics
from src.transformers import glue_convert_examples_to_features as convert_examples_to_features
from src.transformers import glue_output_modes as output_modes
from src.transformers import glue_processors as processors
sys.path.append(ROOT_PATH + '/graph_fea')
from embneed import initialize_optimizer
infoGraph = initialize_optimizer()
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "rna": (BertConfig, BertForSequenceClassification, DNATokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer)
}

TOKEN_ID_GROUP = ["bert"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, train_dataset, model, tokenizer, task=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    no_decay = ["bias", "LayerNorm.weight"]

    # Get parameters for both models
    params_model1 = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_model2 = [p for n, p in infoGraph.named_parameters() if not any(nd in n for nd in no_decay)]

    optimizer_grouped_parameters = [
        {"params": params_model1, "weight_decay": args.weight_decay},
        {"params": params_model2, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in infoGraph.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent * t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    best_auc = 0
    last_auc = 0
    stop_count = 0
    for _ in train_iterator:
        current_index = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)
            key = None
            if task == 'rnafamily':
                key = "family_train"
            emb = infoGraph.a(current_index, infoGraph, key)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "info_fea": emb, "task": task, "key": "train"}
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in TOKEN_ID_GROUP else None
            )  
            outputs, pooled_output = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            current_index += len(batch[0])
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                infoGraph.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        if args.task_name == "rnafamily":
                              eval_results, label, pooled_output = evaluate(args, model, tokenizer)
                              all_fea= [{key: value } for key, value in zip(label, pooled_output)]
                              if eval_results["auc"] > best_auc:
                                  np.save(ROOT_PATH + '/task_out/' + args.task_name + '/all_fea.npy', all_fea)
                        pre_results = predict(args, model, tokenizer)
                        with open(ROOT_PATH + '/task_out/' + args.task_name + '/result.txt', 'a') as file:
                            file.write("eval===========\n")
                            for key in sorted(eval_results.keys()):
                                file.write(f"{key}: {eval_results[key]}, ")
                                file.write("\n")
                            file.write("predict===========\n")
                            for key in sorted(pre_results.keys()):
                                file.write(f"{key}: {pre_results[key]}, ")
                        if args.early_stop != 0:
                            # record current auc to perform early stop
                            if eval_results["auc"] < last_auc:
                                stop_count += 1
                            else:
                                stop_count = 0

                            last_auc = eval_results["auc"]

                            if stop_count == args.early_stop:
                                logger.info("Early stop")
                                return global_step, tr_loss / global_step

                        for key, value in eval_results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    softmax = torch.nn.Softmax(dim=1)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, pre=False, evaluate=True)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        probs = None
        out_label_ids = None
        current_eval_index = 0
        i = 0
        all_best_fea = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            eval_key = None
            if args.task_name == 'rnafamily':
                    eval_key = "family_dev"
            emb = infoGraph.a(current_eval_index, infoGraph, eval_key)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "info_fea": emb, "task":args.task_name, "key": "eval"}
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )
                outputs, pooled_output = model(**inputs)
                pooled_output  = pooled_output.cpu().numpy()
                for fea in pooled_output:
                    all_best_fea.append(fea)
                current_eval_index += len(batch[0])
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        if args.do_ensemble_pred:
            result = compute_metrics(eval_task, preds, out_label_ids, probs[:, 1])
        else:
            result = compute_metrics(eval_task, preds, out_label_ids, probs)
        results.update(result)

    if args.do_ensemble_pred:
        return results, eval_task, preds, out_label_ids, probs
    elif args.task_name == "rnafamily":
        return results, out_label_ids, all_best_fea
    else:
        return results


def predict(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    pred_task_names = (args.task_name,)
    pred_outputs_dirs = (args.result_dir,)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    softmax = torch.nn.Softmax(dim=1)
    predictions = {}
    for pred_task, pred_output_dir in zip(pred_task_names, pred_outputs_dirs):
        pred_dataset = load_and_cache_examples(args, pred_task, tokenizer, pre=True, evaluate=False)
        args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, 1)
        # args.pred_batch_size = args.per_gpu_pred_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        pred_sampler = SequentialSampler(pred_dataset)
        pred_dataloader = DataLoader(pred_dataset, sampler=pred_sampler, batch_size=args.pred_batch_size)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(pred_dataset))
        logger.info("  Batch size = %d", args.pred_batch_size)
        preds = None
        out_label_ids = None
        current_pre_index = 0
        for batch in tqdm(pred_dataloader, desc="Predicting"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            pre_key = None
            if args.task_name == 'rnafamily':
                    pre_key = "family_pre"
            emb = infoGraph.a(current_pre_index, infoGraph, pre_key)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "info_fea": emb, "task":args.task_name, "key": "pre"}
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs, pooled_output = model(**inputs)
                _, logits = outputs[:2]
            current_pre_index += len(batch[0])
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            probs = softmax(torch.tensor(preds, dtype=torch.float32)).numpy()
            preds = np.argmax(preds, axis=1)
        if args.do_ensemble_pred:
            result = compute_metrics(pred_task, preds, out_label_ids, probs[:, 1])
        else:
            result = compute_metrics(pred_task, preds, out_label_ids, probs)
        logger.info("***** Pred results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return result


def load_and_cache_examples(args, task, tokenizer, pre=False, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    current_task = ""
    if evaluate:
        current_task = "dev"
    elif pre:
        current_task="pre"
    else:
        current_task="train"
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            current_task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if pre:
           examples = processor.get_pre_examples(args.data_dir)
        elif evaluate:  
           examples = processor.get_dev_examples(args.data_dir)
        else:
           examples = processor.get_train_examples(args.data_dir)
        print("finish loading examples")

        # params for convert_examples_to_features
        max_length = args.max_seq_length
        pad_on_left = bool(args.model_type in ["xlnet"])
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0

        if args.n_process == 1:
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=max_length,
                output_mode=output_mode,
                pad_on_left=pad_on_left,  # pad on the left for xlnet
                pad_token=pad_token,
                pad_token_segment_id=pad_token_segment_id, )

        else:
            n_proc = int(args.n_process)
            if evaluate:
                n_proc = max(int(n_proc / 4), 1)
            print("number of processes for converting feature: " + str(n_proc))
            p = Pool(n_proc)
            indexes = [0]
            len_slice = int(len(examples) / n_proc)
            for i in range(1, n_proc + 1):
                if i != n_proc:
                    indexes.append(len_slice * (i))
                else:
                    indexes.append(len(examples))

            results = []

            for i in range(n_proc):
                results.append(p.apply_async(convert_examples_to_features, args=(
                examples[indexes[i]:indexes[i + 1]], tokenizer, max_length, None, label_list, output_mode, pad_on_left,
                pad_token, pad_token_segment_id, True,)))
                print(str(i + 1) + ' processor started !')

            p.close()
            p.join()

            features = []
            for result in results:
                features.extend(result.get())

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    device = "cuda:0"
    args = arg_parse()
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    config.split = int(args.max_seq_length / 512)
    config.rnn = args.rnn
    config.num_rnn_layer = args.num_rnn_layer
    config.rnn_dropout = args.rnn_dropout
    config.rnn_hidden = args.rnn_hidden

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    logger.info('finish loading model')

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    torch.cuda.set_device(device)
    model.to(device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, pre=False, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, args.task_name)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Prediction
    predictions = {}
    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = args.output_dir
        logger.info("Predict using the following checkpoint: %s", checkpoint)
        prefix = ''
        model = model_class.from_pretrained(checkpoint)
        model.to(device)
        prediction = predict(args, model, tokenizer, prefix=prefix)

if __name__ == "__main__":
    main()
