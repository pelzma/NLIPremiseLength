# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:52:50 2021

@author: Administrator
"""

import datasets
from datasets import concatenate_datasets, load_dataset
import statistics

anli_dev_r1_avg_len = 0.0
anli_dev_r2_avg_len = 0.0
anli_dev_r3_avg_len = 0.0
anli_id = ('anli',)
anli_dataset = datasets.load_dataset(*anli_id)
anli_r1_premise = anli_dataset['dev_r1']['premise']
anli_r2_premise = anli_dataset['dev_r2']['premise']
anli_r3_premise = anli_dataset['dev_r3']['premise']

for i in range(len(anli_r1_premise)):
    anli_dev_r1_avg_len += len(anli_r1_premise[i])
anli_dev_r1_avg_len = anli_dev_r1_avg_len/len(anli_r1_premise)

for i in range(len(anli_r2_premise)):
    anli_dev_r2_avg_len += len(anli_r2_premise[i])
anli_dev_r2_avg_len = anli_dev_r2_avg_len/len(anli_r2_premise)

for i in range(len(anli_r3_premise)):
    anli_dev_r3_avg_len += len(anli_r3_premise[i])
anli_dev_r3_avg_len = anli_dev_r3_avg_len/len(anli_r3_premise)



snli_val_avg_len = 0.0
snli_id = ('snli',)
snli_dataset = datasets.load_dataset(*snli_id)
snli_premise = snli_dataset['validation']['premise']
for i in range(len(snli_premise)):
    snli_val_avg_len += len(snli_premise[i])
snli_val_avg_len = snli_val_avg_len/len(snli_premise)

pelz_avg_len = 0.0
pelz = load_dataset('csv', data_files='pelz_nli_examples_1.csv')
pelz_premise = pelz['train']['premise']
for i in range(len(pelz_premise)):
    pelz_avg_len += len(pelz_premise[i])
pelz_avg_len = pelz_avg_len/len(pelz_premise)

print("anli_dev_r1 length:")
print(anli_dev_r1_avg_len)
print("anli_dev_r2 length:")
print(anli_dev_r2_avg_len)
print("anli_dev_r3 length:")
print(anli_dev_r3_avg_len)
print("anli_val length:")
print(snli_val_avg_len)
print("pelz length:")
print(pelz_avg_len)

import jsonlines

pelz_exs_2 = []
with jsonlines.open('eval_predictions_2d.jsonl') as f:
    for line in f.iter():
        pelz_exs_2.append(line)
pelz_exs_2_wrong= []
for i in range(len(pelz_exs_2)):
    if pelz_exs_2[i]['label'] != pelz_exs_2[i]['predicted_label']:
        pelz_exs_2_wrong.append(pelz_exs_2[i])
pelz_exs_2_wrong_avg_len = 0
for i in range(len(pelz_exs_2_wrong)):
    pelz_exs_2_wrong_avg_len += len(pelz_exs_2_wrong[i]['premise'])
pelz_exs_2_wrong_avg_len = pelz_exs_2_wrong_avg_len/len(pelz_exs_2_wrong)
print("pelz with anli model wrong length:")
print(pelz_exs_2_wrong_avg_len)

pelz_exs_1 = []
with jsonlines.open('eval_predictions_1d.jsonl') as f:
    for line in f.iter():
        pelz_exs_1.append(line)
pelz_exs_1_wrong= []
for i in range(len(pelz_exs_1)):
    if pelz_exs_1[i]['label'] != pelz_exs_1[i]['predicted_label']:
        pelz_exs_1_wrong.append(pelz_exs_1[i])
pelz_exs_1_wrong_avg_len = 0
for i in range(len(pelz_exs_1_wrong)):
    pelz_exs_1_wrong_avg_len += len(pelz_exs_1_wrong[i]['premise'])
pelz_exs_1_wrong_avg_len = pelz_exs_1_wrong_avg_len/len(pelz_exs_1_wrong)
print("pelz without anli model wrong length:")
print(pelz_exs_1_wrong_avg_len)

snli_exs_2 = []
with jsonlines.open('eval_predictions_2c.jsonl') as f:
    for line in f.iter():
        snli_exs_2.append(line)
snli_exs_2_wrong= []
for i in range(len(snli_exs_2)):
    if snli_exs_2[i]['label'] != snli_exs_2[i]['predicted_label']:
        snli_exs_2_wrong.append(snli_exs_2[i])
snli_exs_2_wrong_avg_len = 0
for i in range(len(snli_exs_2_wrong)):
    snli_exs_2_wrong_avg_len += len(snli_exs_2_wrong[i]['premise'])
snli_exs_2_wrong_avg_len = snli_exs_2_wrong_avg_len/len(snli_exs_2_wrong)
print("snli with anli model wrong length:")
print(snli_exs_2_wrong_avg_len)

snli_exs_1 = []
with jsonlines.open('eval_predictions_1c.jsonl') as f:
    for line in f.iter():
        snli_exs_1.append(line)
snli_exs_1_wrong= []
for i in range(len(snli_exs_1)):
    if snli_exs_1[i]['label'] != snli_exs_1[i]['predicted_label']:
        snli_exs_1_wrong.append(snli_exs_1[i])
snli_exs_1_wrong_avg_len = 0
for i in range(len(snli_exs_1_wrong)):
    snli_exs_1_wrong_avg_len += len(snli_exs_1_wrong[i]['premise'])
snli_exs_1_wrong_avg_len = snli_exs_1_wrong_avg_len/len(snli_exs_1_wrong)
print("snli without anli model wrong length:")
print(snli_exs_1_wrong_avg_len)

snli_exs_1_long= []
snli_exs_1_long_correct = 0
snli_exs_1_long_entailment = 0
snli_exs_1_long_pred_entailment = 0
snli_exs_1_long_neutral = 0
snli_exs_1_long_pred_neutral = 0
for i in range(len(snli_exs_1)):
    if len(snli_exs_1[i]['premise']) < len(snli_exs_1[i]['hypothesis']):
        snli_exs_1_long.append(snli_exs_1[i])
        if snli_exs_1[i]['label'] == snli_exs_1[i]['predicted_label']:
            snli_exs_1_long_correct += 1
        if snli_exs_1[i]['label'] == 0:
            snli_exs_1_long_entailment += 1
        if snli_exs_1[i]['predicted_label'] == 0:
            snli_exs_1_long_pred_entailment += 1
        if snli_exs_1[i]['label'] == 1:
            snli_exs_1_long_neutral += 1
        if snli_exs_1[i]['predicted_label'] == 1:
            snli_exs_1_long_pred_neutral += 1
snli_exs_1_long_acc = snli_exs_1_long_correct/len(snli_exs_1_long)
print("snli 1 accuracy when premise shorter than hypothesis:")
print(snli_exs_1_long_acc)

snli_exs_2 = []
with jsonlines.open('eval_predictions_2c.jsonl') as f:
    for line in f.iter():
        snli_exs_2.append(line)
snli_exs_2_wrong= []
for i in range(len(snli_exs_2)):
    if snli_exs_2[i]['label'] != snli_exs_2[i]['predicted_label']:
        snli_exs_2_wrong.append(snli_exs_2[i])
snli_exs_2_wrong_avg_len = 0
for i in range(len(snli_exs_2_wrong)):
    snli_exs_2_wrong_avg_len += len(snli_exs_2_wrong[i]['premise'])
snli_exs_2_wrong_avg_len = snli_exs_2_wrong_avg_len/len(snli_exs_2_wrong)

snli_exs_2_long= []
snli_exs_2_long_correct = 0
snli_exs_2_long_entailment = 0
snli_exs_2_long_pred_entailment = 0
snli_exs_2_long_neutral = 0
snli_exs_2_long_pred_neutral = 0
for i in range(len(snli_exs_1)):
    if len(snli_exs_2[i]['premise']) < len(snli_exs_2[i]['hypothesis']):
        snli_exs_2_long.append(snli_exs_2[i])
        if snli_exs_2[i]['label'] == snli_exs_2[i]['predicted_label']:
            snli_exs_2_long_correct += 1
        if snli_exs_2[i]['label'] == 0:
            snli_exs_2_long_entailment += 1
        if snli_exs_2[i]['predicted_label'] == 0:
            snli_exs_2_long_pred_entailment += 1
        if snli_exs_2[i]['label'] == 1:
            snli_exs_2_long_neutral += 1
        if snli_exs_2[i]['predicted_label'] == 1:
            snli_exs_2_long_pred_neutral += 1
snli_exs_2_long_acc = snli_exs_2_long_correct/len(snli_exs_1_long)
print("snli 2 accuracy when premise shorter than hypothesis:")
print(snli_exs_2_long_acc)

snli_exs_1_prem_len =[]
for i in range(len(snli_exs_1)):
    snli_exs_1_prem_len.append(len(snli_exs_1[i]['premise']))
sd = statistics.stdev(snli_exs_1_prem_len)
mean = statistics.mean(snli_exs_1_prem_len)
len_threshold = mean + 2*sd
min_len_threshold = mean
    

snli_exs_1_long= []
snli_exs_1_long_correct = 0
snli_exs_1_long_entailment = 0
snli_exs_1_long_pred_entailment = 0
snli_exs_1_long_neutral = 0
snli_exs_1_long_pred_neutral = 0
for i in range(len(snli_exs_1)):
    if len(snli_exs_1[i]['premise']) > len_threshold:
        snli_exs_1_long.append(snli_exs_1[i])
        if snli_exs_1[i]['label'] == snli_exs_1[i]['predicted_label']:
            snli_exs_1_long_correct += 1
        if snli_exs_1[i]['label'] == 0:
            snli_exs_1_long_entailment += 1
        if snli_exs_1[i]['predicted_label'] == 0:
            snli_exs_1_long_pred_entailment += 1
        if snli_exs_1[i]['label'] == 1:
            snli_exs_1_long_neutral += 1
        if snli_exs_1[i]['predicted_label'] == 1:
            snli_exs_1_long_pred_neutral += 1
snli_exs_1_long_acc = snli_exs_1_long_correct/len(snli_exs_1_long)
print("snli 1 long premises accuracy:")
print(snli_exs_1_long_acc)

snli_exs_2_long= []
snli_exs_2_long_correct = 0
snli_exs_2_long_entailment = 0
snli_exs_2_long_pred_entailment = 0
snli_exs_2_long_neutral = 0
snli_exs_2_long_pred_neutral = 0
for i in range(len(snli_exs_2)):
    if len(snli_exs_2[i]['premise']) > len_threshold:
        snli_exs_2_long.append(snli_exs_2[i])
        if snli_exs_2[i]['label'] == snli_exs_2[i]['predicted_label']:
            snli_exs_2_long_correct += 1
        if snli_exs_2[i]['label'] == 0:
            snli_exs_2_long_entailment += 1
        if snli_exs_2[i]['predicted_label'] == 0:
            snli_exs_2_long_pred_entailment += 1
        if snli_exs_2[i]['label'] == 1:
            snli_exs_2_long_neutral += 1
        if snli_exs_2[i]['predicted_label'] == 1:
            snli_exs_2_long_pred_neutral += 1
snli_exs_2_long_acc = snli_exs_2_long_correct/len(snli_exs_2_long)
print("snli 2 long premises accuracy:")
print(snli_exs_2_long_acc)

snli_exs_1_short= []
snli_exs_1_short_correct = 0
for i in range(len(snli_exs_1)):
    if len(snli_exs_1[i]['premise']) < min_len_threshold:
        snli_exs_1_short.append(snli_exs_1[i])
        if snli_exs_1[i]['label'] == snli_exs_1[i]['predicted_label']:
            snli_exs_1_short_correct += 1
snli_exs_1_short_acc = snli_exs_1_short_correct/len(snli_exs_1_short)
print("snli 1 short premises accuracy:")
print(snli_exs_1_short_acc)

snli_exs_2_short= []
snli_exs_2_short_correct = 0
for i in range(len(snli_exs_2)):
    if len(snli_exs_2[i]['premise']) < min_len_threshold:
        snli_exs_2_short.append(snli_exs_2[i])
        if snli_exs_2[i]['label'] == snli_exs_2[i]['predicted_label']:
            snli_exs_2_short_correct += 1
snli_exs_2_short_acc = snli_exs_2_short_correct/len(snli_exs_1_short)
print("snli 2 short premises accuracy:")
print(snli_exs_2_short_acc)