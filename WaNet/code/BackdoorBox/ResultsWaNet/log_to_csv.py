import csv
import re

with open('log.txt') as f:
    log = f.read()

# 提取mean loss
pattern = r"Epoch:(\d+)/\d+.*?==========Test result on benign test dataset==========.*?mean loss: (\d+\.\d+).*?==========Test result on poisoned test dataset==========.*?mean loss: (\d+\.\d+)"
results = re.findall(pattern, log, re.DOTALL)

with open('loss.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'benign test dataset', 'poisoned test dataset'])
    writer.writerows(results)

# 提取Top-1 accuracy
pattern = r"Epoch:(\d+)/\d+.*?==========Test result on benign test dataset==========.*?Top-1 accuracy: (\d+\.\d+).*?==========Test result on poisoned test dataset==========.*?Top-1 accuracy: (\d+\.\d+)"    
results = re.findall(pattern, log, re.DOTALL)

with open('accuracy.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'benign test dataset', 'poisoned test dataset']) 
    writer.writerows(results)