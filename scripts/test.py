# -*- coding: utf-8 -*-

import math
import torch

def test_model(model, testloader, device):

    correct = 0
    total = 0

    for idx, (labels, inputs) in enumerate(testloader):
        # iter_batch = math.ceil(testloader/testloader.batch_size)
        # print(f'[phase: test] batch: {idx+1}/{iter_batch}', end='\r')

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            # Convert to probabilities
            probabilities = torch.exp(outputs)
            _, predicted = torch.max(outputs, 1)

            # total += labels.size(0)
            total = idx+1
            correct += torch.sum(predicted == labels.data)
            # print(f'{torch.max(torch.exp(outputs), dim=1)} - {torch.exp(outputs)}')

    print(f'[phase: test] acc: {total} {correct} {100*(correct.item()/total):.3f}')