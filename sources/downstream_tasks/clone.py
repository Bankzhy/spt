import argparse
import logging
import os
import sys

import numpy as np
from transformers import BartConfig, IntervalStrategy, SchedulerType, Seq2SeqTrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

from args import add_args
from data.data_utils import load_clone_mapping

curPath = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(curPath)
sys.path.append('../..')
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)

from models.bart import BartForClassificationAndGeneration
from utils.callbacks import LogStateCallBack

from data.dataset import init_dataset
from utils.general import human_format, count_params, layer_wise_parameters
from utils.trainer import CodeTrainer, CodeCLSTrainer
from data.vocab import load_vocab, init_vocab, Vocab
import enums
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
from typing import Union, Tuple

def run_clone(
        args,
        trained_model: Union[BartForClassificationAndGeneration, str] = None,
        trained_vocab: Union[Tuple[Vocab, Vocab, Vocab], str] = None,
        only_test=False):
    # --------------------------------------------------
    # datasets
    # --------------------------------------------------
    only_test = False
    logger.info('-' * 100)
    logger.info('Loading datasets')
    datasets = dict()
    splits = ['test'] if only_test else ['train', 'valid', 'test']
    for split in splits:
        clone_mapping = load_clone_mapping(dataset_root=args.dataset_save_dir)
        datasets[split] = init_dataset(args=args,
                                       mode=enums.TRAINING_MODE_FINE_TUNE,
                                       task=enums.TASK_CLONE_DETECTION,
                                       language=args.summarization_language,
                                       clone_mapping=clone_mapping,
                                       split=split)
        # datasets[split] = datasets[split].subset(0.1)
        logger.info(f'The size of {split} set: {len(datasets[split])}')
    if args.train_subset_ratio and 'train' in datasets:
        datasets['train'] = datasets['train'].subset(args.train_subset_ratio)
        logger.info(f'The train is trimmed to subset due to the argument: train_subset_ratio={args.train_subset_ratio}')
        logger.info('The size of trimmed train set: {}'.format(len(datasets['train'])))

    logger.info('Datasets loaded successfully')

    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_vocab:
        if isinstance(trained_vocab, tuple):
            logger.info('Vocabularies are passed through parameter')
            assert len(trained_vocab) == 3
            code_vocab, ast_vocab, nl_vocab = trained_vocab
        else:
            logger.info('Loading vocabularies from files')
            code_vocab = load_vocab(vocab_root=trained_vocab, name=args.code_vocab_name)
            ast_vocab = load_vocab(vocab_root=trained_vocab, name=args.ast_vocab_name)
            nl_vocab = load_vocab(vocab_root=trained_vocab, name=args.nl_vocab_name)
    else:
        logger.info('Building vocabularies')
        code_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                                name=args.code_vocab_name,
                                method=args.code_tokenize_method,
                                vocab_size=args.code_vocab_size,
                                datasets=[datasets['train'].codes],
                                ignore_case=True,
                                save_root=args.vocab_root)
        nl_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                              name=args.nl_vocab_name,
                              method=args.nl_tokenize_method,
                              vocab_size=args.nl_vocab_size,
                              datasets=[datasets['train'].nls],
                              ignore_case=True,
                              save_root=args.vocab_root,
                              index_offset=len(code_vocab))
        ast_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                               name=args.ast_vocab_name,
                               method='word',
                               datasets=[datasets['train'].asts],
                               save_root=args.vocab_root,
                               index_offset=len(code_vocab) + len(nl_vocab))
    logger.info(f'The size of code vocabulary: {len(code_vocab)}')
    logger.info(f'The size of nl vocabulary: {len(nl_vocab)}')
    logger.info(f'The size of ast vocabulary: {len(ast_vocab)}')
    logger.info('Vocabularies built successfully')

    # --------------------------------------------------
    # model
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_model:
        if isinstance(trained_model, BartForClassificationAndGeneration):
            logger.info('Model is passed through parameter')
            model = trained_model
        else:
            logger.info('Loading the model from file')
            config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
            model = BartForClassificationAndGeneration.from_pretrained(os.path.join(trained_model, 'pytorch_model.bin'),
                                                                       config=config)
    else:
        logger.info('Building the model')
        config = BartConfig(vocab_size=len(code_vocab) + len(nl_vocab) + len(ast_vocab),
                            max_position_embeddings=1024,
                            encoder_layers=args.n_layer,
                            encoder_ffn_dim=args.d_ff,
                            encoder_attention_heads=args.n_head,
                            decoder_layers=args.n_layer,
                            decoder_ffn_dim=args.d_ff,
                            decoder_attention_heads=args.n_head,
                            activation_function='gelu',
                            d_model=args.d_model,
                            dropout=args.dropout,
                            use_cache=True,
                            pad_token_id=Vocab.START_VOCAB.index(Vocab.PAD_TOKEN),
                            bos_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            is_encoder_decoder=True,
                            decoder_start_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            forced_eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            max_length=args.max_nl_len,
                            min_length=1,
                            num_beams=args.beam_width,
                            num_labels=2)
        model = BartForClassificationAndGeneration(config)
    model.set_model_mode(enums.MODEL_MODE_CLS)
    # log model statistics
    logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Initializing the running configurations')

    def decode_preds(preds):
        preds, labels = preds
        decoded_preds = nl_vocab.decode_batch(preds)
        decoded_labels = nl_vocab.decode_batch(labels)
        return decoded_labels, decoded_preds

    # compute metrics
    def compute_valid_metrics(eval_preds):
        logits = eval_preds.predictions[0]
        labels = eval_preds.label_ids
        predictions = np.argmax(logits, axis=-1)
        from sklearn.metrics import recall_score
        recall = recall_score(labels, predictions)
        from sklearn.metrics import precision_score
        precision = precision_score(labels, predictions)
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, predictions)
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
        }

        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
        return result

    def compute_test_metrics(eval_preds):
        # decoded_preds, decoded_labels = eval_preds
        logits = eval_preds.predictions[0]
        labels = eval_preds.label_ids
        predictions = np.argmax(logits, axis=-1)
        from sklearn.metrics import recall_score
        recall = recall_score(labels, predictions)
        from sklearn.metrics import precision_score
        precision = precision_score(labels, predictions)
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, predictions)
        result = {
            "eval_recall": float(recall),
            "eval_precision": float(precision),
            "eval_f1": float(f1),
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

        return result

    # 尽量不要用IntervalStrategy.EPOCH， 太过频繁影响训练效果，还会曾家训练时间
    training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.checkpoint_root, enums.TASK_CLONE_DETECTION),
                                             overwrite_output_dir=True,
                                             do_train=True,
                                             do_eval=True,
                                             do_predict=True,
                                             evaluation_strategy=IntervalStrategy.STEPS,
                                             eval_steps=2500,
                                             prediction_loss_only=False,
                                             per_device_train_batch_size=args.batch_size,
                                             per_device_eval_batch_size=args.eval_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             weight_decay=args.lr_decay_rate,
                                             max_grad_norm=args.grad_clipping_norm,
                                             num_train_epochs=args.n_epoch,
                                             lr_scheduler_type=SchedulerType.LINEAR,
                                             warmup_steps=args.warmup_steps,
                                             logging_dir=os.path.join(args.tensor_board_root, enums.TASK_CLONE_DETECTION),
                                             logging_strategy=IntervalStrategy.STEPS,
                                             logging_steps=2500,
                                             save_strategy=IntervalStrategy.STEPS,
                                             save_steps=2500,
                                             save_total_limit=5,
                                             seed=args.random_seed,
                                             fp16=args.fp16,
                                             dataloader_drop_last=False,
                                             run_name=args.model_name,
                                             load_best_model_at_end=True,
                                             # metric_for_best_model='bleu',
                                             greater_is_better=True,
                                             ignore_data_skip=False,
                                             label_smoothing_factor=args.label_smoothing,
                                             eval_accumulation_steps=100,
                                             report_to=['tensorboard'],
                                             dataloader_pin_memory=True,
                                             predict_with_generate=True)
    trainer = CodeCLSTrainer(main_args=args,
                          code_vocab=code_vocab,
                          ast_vocab=ast_vocab,
                          nl_vocab=nl_vocab,
                          task=enums.TASK_CLONE_DETECTION,
                          model=model,
                          args=training_args,
                          data_collator=None,
                          train_dataset=datasets['train'] if 'train' in datasets else None,
                          eval_dataset=datasets['valid'] if 'valid' in datasets else None,
                          tokenizer=nl_vocab,
                          model_init=None,
                          compute_metrics=compute_valid_metrics,
                          callbacks=[
                              EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience),
                              LogStateCallBack()])
    logger.info('Running configurations initialized successfully')

    # --------------------------------------------------
    # train
    # --------------------------------------------------
    if not only_test:
        logger.info('-' * 100)
        # logger.info('loading checkpoint')
        # last_checkpoint = get_last_checkpoint(os.path.join(args.checkpoint_root, enums.TASK_SUMMARIZATION),)
        logger.info('Start training')
        # train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        train_result = trainer.train()
        logger.info('Training finished')
        trainer.save_model(args.model_root)
        trainer.save_state()
        metrics = train_result.metrics
        trainer.log_metrics(split='train', metrics=metrics)
        trainer.save_metrics(split='train', metrics=metrics)

        # --------------------------------------------------
        # eval
        # --------------------------------------------------
        # logger.info('-' * 100)
        # logger.info('Start evaluating')
        # eval_metrics = trainer.evaluate(metric_key_prefix='valid',
        #                                 max_length=args.max_decode_step,
        #                                 num_beams=args.beam_width)
        # trainer.log_metrics(split='valid', metrics=eval_metrics)
        # trainer.save_metrics(split='valid', metrics=eval_metrics)

    # --------------------------------------------------
    # predict
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Start testing')
    trainer.compute_metrics = compute_test_metrics
    predict_results = trainer.predict(test_dataset=datasets['test'],
                                      metric_key_prefix='test',
                                      max_length=args.max_nl_len,
                                      num_beams=args.beam_width)
    predict_metrics = predict_results.metrics
    references = predict_metrics.pop('test_references')
    candidates = predict_metrics.pop('test_candidates')
    trainer.log_metrics(split='test', metrics=predict_metrics)
    trainer.save_metrics(split='test', metrics=predict_metrics)
    # save testing results
    with open(os.path.join(args.output_root, f'{enums.TASK_CLONE_DETECTION}_test_results.txt'),
              mode='w', encoding='utf-8') as result_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_CLONE_DETECTIONN}_test_refs.txt'),
                 mode='w', encoding='utf-8') as refs_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_CLONE_DETECTION}_test_cans.txt'),
                 mode='w', encoding='utf-8') as cans_f:
        sample_id = 0
        for reference, candidate in zip(references, candidates):
            result_f.write(f'sample {sample_id}:\n')
            sample_id += 1
            result_f.write(f'reference: {reference}\n')
            result_f.write(f'candidate: {candidate}\n')
            result_f.write('\n')
            refs_f.write(reference + '\n')
            cans_f.write(candidate + '\n')
        for name, score in predict_metrics.items():
            result_f.write(f'{name}: {score}\n')
    logger.info('Testing finished')
    for name, score in predict_metrics.items():
        logger.info(f'{name}: {score}')



if __name__ == '__main__':
    run_clone()