import torch
import yaml

from abc import ABC
from typing import List


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
        Calculate, using PyTorch, the sum of the elements of the range from 0 to 10000.
    """
    def __init__(self) -> None:
        self.task_name = "task1"

    def solve(self):
        return torch.sum(torch.arange(0,10000)).item()

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.item()}}


class Task2(Task):
    """
        Solve optimization problem: find the minimum of the function f(x) = ||Ax^2 + Bx + C||^2, where
        - x is vector of size 8
        - A is identity matrix of size 8x8
        - B is matrix of size 8x8, where each element is 0
        - C is vector of size 8, where each element is -1

        Use PyTorch and autograd function. Relative error will be less than 1e-3
        
        Solution here is x, converted to the list(see submission.yaml).
    """
    def __init__(self) -> None:
        self.task_name = "task2"

    def solve(self):
        # needed tensors
        x = torch.randn((1,8), requires_grad = True)
        A = torch.eye(8)
        B = torch.zeros((8,8))
        C = torch.ones((8,8)) * -1

        # func
        def _f(x):
            return abs(A * x**2  + B * x + C)**2
        
        # grad descent
        for i in range(300):
            y = f(x)
            x.grad = torch.zeros(1,8)
            y.backward(torch.ones(8,8))
            x.data -= x.grad * 0.1 # lr

        return x[0]

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.tolist()}}


class Task3(Task):
    """
        Solve optimization problem: find the optimal parameters of the linear regression model, using PyTorch.
        train_X = [[0, 0], [1, 0], [0, 1], [1, 1]],
        train_y = [1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746]

        text_X = [[0, -1], [-1, 0]]

        User PyTorch. Relative error will be less than 1e-1
        
        Solution here is test_y, calculated from test_X, converted to the list(see submission.yaml).
    """
    def __init__(self) -> None:
        self.task_name = "task3"

    def solve(self):
        # train dataset
        train_X = [[0, 0], [1, 0], [0, 1], [1, 1]],
        train_y = [1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746]

        # define model, optim, loss_fn
        model = torch.nn.Linear(2, 1)
        loss_fn = torch.nn.MSELoss()
        optim = torch.optim.SGD(model.parameters(), lr=0.001)

        # training
        for epoch in range(12000):
            for i in range(3):
                optim.zero_grad()
                logit = model(torch.Tensor(train_X[0][i]))
                loss = loss_fn(logit, torch.tensor(train_y[i]))
                loss.backward()
                optim.step()

        # evaluate
        text_X = [[0, -1], [-1, 0]]

        return model(torch.Tensor(text_X)).squeeze(1)
    
    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.tolist()}}


class HW(object):
    def __init__(self, list_of_tasks: List[Task]):
        self.tasks = list_of_tasks
        self.hw_name = "dl_lesson_1_checker_hw"

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
