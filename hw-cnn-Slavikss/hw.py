from abc import ABC
from typing import List, Tuple, Union
import torch
import torchvision
import numpy as np
import random
from PIL import Image
import yaml


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Task(ABC):
    def solve():
        """
        Function to implement your solution, write here
        """

    def evaluate():
        """
        Function to evaluate your solution
        """


class Task1(Task):
    """
    In this task you will be asked to calculate
    number of parameters for resnet18 model.

    1. Import resnet18 model from torchvision library
    *Hint* install torchvision, if needed: `pip install torchvision`
    2. There are 3 types of modules, which contain parameters:
        2.1. `Conv2d`
        2.2. `BatchNorm2d`
        2.3. `Linear`
    Your answer should be as follows:
        3 item tuple (or list): (a, b, c), where:
            a - total number of parameters in all Conv2d layers
            b - total number of parameters in all BatchNorm2d layers
            c - total number of parameters in all Linear layers
    *Hint* `sum([x.numel() for x in model.parameters()])` - it gives
    you the total number of all parameters. (So, you can check, that
    your 3 item sum is the same)
    """

    def __init__(self) -> None:
        self.task_name = "task1"

    def solve(self) -> Union[Tuple[int, int, int], List[int]]:
        # model
        model = torchvision.models.resnet18()

        # number of parameters
        a = sum(sum(x.numel() for x in layer.parameters() 
            if isinstance(layer, torch.nn.Linear)) 
            for _, layer in model.named_modules())

        b = sum(sum(x.numel() for x in layer.parameters() 
            if isinstance(layer, torch.nn.Conv2d)) 
            for _, layer in model.named_modules())

        c = sum(sum(x.numel() for x in layer.parameters() 
            if isinstance(layer, torch.nn.BatchNorm2d)) \
            for _, layer in model.named_modules())
        
        return a,b,c

    def evaluate(self):
        solution = self.solve()
        assert isinstance(solution, tuple) or isinstance(
            solution, list
        ), "Should be a tuple or a list"
        assert len(solution) == 3, "Solution must contain 3 elements"
        return {self.task_name: {"answer": solution}}


class Task2(Task):
    """
    In this task you will get yourself a little bit more
    familiar with image transformations

    1. Download the following image:
    https://acdn.tinkoff.ru/static/documents/703d269e-ec01-4187-9544-5f01fb27bbb6.png
    2. Use `torchvision.transforms` to make next transformations (it's mandatory)
        - get center crop of the image (size should be 224x224)
        - convert obtained image into grayscale
        - convert this grayscale image into a tensor 
    3. Your answer is the mean value of this tensor 
    (Be carefull, the answer should be <= 1. Can you see why?)
    """

    def __init__(self) -> None:
        self.task_name = "task2"

    def solve(self) -> float:

        # get image
        url = "https://acdn.tinkoff.ru/static/documents/703d269e-ec01-4187-9544-5f01fb27bbb6.png"
        r = requests.get(url, allow_redirects=True)
        open('image.png', 'wb').write(r.content)

        #open image
        img = Image.open('image.png')

        # transforms
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((224,224)),
            torchvision.transforms.Grayscale(1),
            torchvision.transforms.ToTensor()
        ])

        # apply transforms and get mean of tensor
        return torch.mean(transforms(img)).item()

    def evaluate(self):
        solution = self.solve()
        assert isinstance(solution, float), "Answer must be float number"
        return {self.task_name: {"answer": solution}}


class Task3(Task):
    """
    In this task you will finetune pretrained model 
    on the "bees and ants" dataset

    *Disclaimer* 
    Due to randomness, results may be
    different on different runs. In order to achieve
    the result as similar as possible with the correct result,
    It is strongly recommended to ensure you have the following environment:

    torch==1.12.1+cu113
    torchvision==0.13.1+cu113

    (The correct result was obtained in the colab, with cuda-enabled runtime)

    Please, strictly follow the instructions:

    0. Download the following dataset:
    https://download.pytorch.org/tutorial/hymenoptera_data.zip
    (you can download it using `wget` and then unpack it using `unzip`)

    1. Set seeds for reproducibility:
    (source: https://pytorch.org/docs/stable/notes/randomness.html)
        - seed is 0
        - set seed for torch
        - set seed for python
        - set seed for numpy
    
    2. Prepare resnet18 model
    (source: https://pytorch.org/vision/stable/models.html)
        - pretrained weights are `ResNet18_Weights.IMAGENET1K_V1`
    *Note* you will need to use proper image preprocessing (You can find it
    inside `ResNet18_Weights.IMAGENET1K_V1` object)

    3. Freeze layers and create new fully_connected layer
        - you should freeze all layers, except the last BasicBlock
        (so, every module, whose name starts 
        with `layer4.1` must be NOT freezed)
        - create new fully_connected layer with 2 output features
        (bias must be set to `True`)
    
    4. Prepare 2 datasets. Use `torchvision.datasets.ImageFolder` 
    to create them
        - train dataset (it is located in `hymenoptera_data/train`)
        - val dataset (it is located in `hymenoptera_data/val`)
    
    5. Create DataLoaders with the following params:
        - batch_size: 64
        - num_workers: 1
        - shuffle: True for train dataset, False for val dataset
        - left the rest parameters by default
    
    6. For optimization use SGD with the following params:
        - lr: 0.001
        - momentum: 0.9
    
    7. Criterion is `CrossEntropyLoss`

    8. Train your model for 5 epochs.

    9. Your final result is the accuracy metric on the validation dataset
    after training is done. 
    Please be careful when implementing accuracy (you should have 
    `total_correct / total_number` exactly)

    The checker will compare your accuracy with the "correct" one.
    Relative tolerance is set to 1e-5
    """

    def __init__(self) -> None:
        self.task_name = "task3"

    def solve(self) -> float:
        # !wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
        # !unzip /content/hymenoptera_data.zip 


        # Init new model
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bb = torchvision.models.resnet18()
                self.fc = torch.nn.Linear(1000,2)

                params_not_to_freeze = [
                'layer4.1.conv1.weight',
                'layer4.1.bn1.weight',
                'layer4.1.bn1.bias',
                'layer4.1.conv2.weight',
                'layer4.1.bn2.weight',
                'layer4.1.bn2.bias',
                'fc.weight',
                'fc.bias']

                for name, param in self.bb.named_parameters():
                    if name not in params_not_to_freeze:
                        param.requires_grad = False

            def forward(self, x):
                x = self.bb(x)
                x = self.fc(x)
                return x        

        ###############################################################
        # Init transforms, dataset, optim etc
        transform = torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize(256),
                                                    torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                                    std=[0.229, 0.224, 0.225])
                                                    ])

        train_dataset = torchvision.datasets.ImageFolder('/content/hymenoptera_data/train', 
                                                        transform = transform)

        val_dataset = torchvision.datasets.ImageFolder('/content/hymenoptera_data/val', 
                                                        transform = transform)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size = 64, 
                                                    num_workers = 1,
                                                    shuffle = True)

        val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                    batch_size = 64, 
                                                    num_workers = 1)

        model = Model() 
        model.to(device)
        optim = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
        loss_fn = torch.nn.CrossEntropyLoss()

        #########################################################################
        # Train loop

        model.train()
        for epoch in range(5):
            for batch in train_dataloader:
                input, target = batch
                input, target = input.to(device), target.to(device)

                logits = model(input)

                loss = loss_fn(logits, target)
                optim.zero_grad()
                loss.backward()
                optim.step()


        #########################################################################
        # Eval loop
        preds = []
        targets = []

        model.eval()
        for batch in val_dataloader:
            input, target = batch
            input = input.to(device)

            preds.extend(model(input).argmax(1).tolist())
            targets.extend(target.tolist())

        # Return accuracy
        return sum([preds[i] == targets[i] for i in range(len(preds))]) / len(preds)


    def evaluate(self):
        solution = self.solve()
        assert isinstance(solution, float), "Answer must be float number"
        return {self.task_name: {"answer": solution}}


class HW(object):
    def __init__(self, list_of_tasks: List[Task]):
        self.tasks = list_of_tasks
        self.hw_name = "dl_lesson_2_hw"

    def evaluate(self):
        aggregated_tasks = []

        for task in self.tasks:
            aggregated_tasks.append(task.evaluate())

        aggregated_tasks = {"tasks": aggregated_tasks}

        yaml_result = yaml.dump(aggregated_tasks)

        print(yaml_result)

        with open(f"{self.hw_name}.yaml", "w") as f:
            f.write(yaml_result)


hw = HW([Task1(), Task2(), Task3()])
hw.evaluate()
