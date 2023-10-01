import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor, tp: int = 1):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return self.softmax(x / tp)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


class TSFrameworkMS(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(TSFrameworkMS, self).__init__()
        self.teacher = Net(input_dim, hidden_dim, output_dim)
        self.students = [
            Net(input_dim, hidden_dim, output_dim) for _ in range(input_dim)
        ]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor, tps: float = 1, tpt: float = 1):
        return self.teacher(x, tpt), [student(x, tps) for student in self.students]

    def predict(self, x: torch.Tensor, tps: float = 1, tpt: float = 1):
        with torch.no_grad():
            return self.forward(x, tps, tpt)

    def predict_teacher(self, x: torch.Tensor, tpt: float = 1):
        return self.predict(x, tpt=tpt)[0]

    def score(self, loader: DataLoader):
        with torch.no_grad():
            correct, total = 0, 0

            for X, y in loader:
                y_pred = self.predict_teacher(X).argmax(dim=1)
                correct += y_pred.eq(y).sum().item()
                total += len(y)

            return correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 0.01,
        w_teacher: float = 0.5,
        n_epochs: int = 300,
        patience: int = 30,
        verbose: bool = True,
        tps: float = 1,
        tpt: float = 1,
        alpha: float = 0.5,
    ):
        s_optimizers = [Adam(student.parameters(), lr=lr) for student in self.students]
        t_optimizer = Adam(self.teacher.parameters(), lr=lr)
        min_val_acc = -np.inf
        best_model = None
        cur_patience = 0

        train_acc, val_acc = [], []

        # initialize teacher parameters as average of student parameters
        for teacher_param, student_params in zip(
            self.teacher.parameters(),
            zip(*[student.parameters() for student in self.students]),
        ):
            teacher_param.data = torch.stack(student_params).mean(dim=0)

        for epoch in range(n_epochs):
            for X, y in train_loader:
                for so in s_optimizers:
                    so.zero_grad()

                t_optimizer.zero_grad()

                teacher_output, students_output = self.forward(X)
                student_losses = [
                    nn.CrossEntropyLoss()(student_output, teacher_output.detach())
                    for student_output in students_output
                ]
                teacher_loss = alpha * nn.CrossEntropyLoss()(teacher_output, y)

                for student_loss, so in zip(student_losses, s_optimizers):
                    student_loss.backward()
                    so.step()

                teacher_loss.backward()
                t_optimizer.step()

                # update teacher parameters as exponential moving average of student parameters
                for teacher_param, student_param in zip(
                    self.teacher.parameters(),
                    zip(*[student.parameters() for student in self.students]),
                ):
                    teacher_param.data = teacher_param.data * w_teacher + torch.mean(
                        torch.stack(student_param), dim=0
                    ) * (1 - w_teacher)

            train_score = self.score(train_loader)
            val_score = self.score(val_loader)

            train_acc.append(train_score)
            val_acc.append(val_score)

            if verbose:
                print(
                    f"Epoch {epoch + 1}:\n"
                    f"\tTrain accuracy: {train_score}\n"
                    f"\tVal accuracy: {val_score}"
                )

            if val_score > min_val_acc:
                min_val_acc = val_score
                best_model = self.state_dict()
                cur_patience = 0

            else:
                cur_patience += 1
                if cur_patience == patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.load_state_dict(best_model)

        # test accuracy
        full_test_acc = self.score(test_loader)

        partial_test_acc = []

        for i in range(len(self.students)):
            test_loader_copy = DataLoader(
                test_loader.dataset, batch_size=1, shuffle=False
            )
            test_loader_copy.dataset.X[:, i] = test_loader_copy.dataset.X[:, i].mean()
            partial_test_acc.append(self.score(test_loader_copy))

        return full_test_acc, np.mean(partial_test_acc), np.std(partial_test_acc)


class TSFrameworkOS(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(TSFrameworkOS, self).__init__()
        self.teacher = Net(input_dim, hidden_dim, output_dim)
        self.student = Net(input_dim, hidden_dim, output_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor, tps: float = 1, tpt: float = 1):
        return self.teacher(x, tpt), self.student(x, tps)

    def predict(self, x: torch.Tensor, tps: float = 1, tpt: float = 1):
        with torch.no_grad():
            return self.forward(x, tps, tpt)

    def predict_teacher(self, x: torch.Tensor, tpt: float = 1):
        return self.predict(x, tpt=tpt)[0]

    def score(self, loader: DataLoader):
        with torch.no_grad():
            correct, total = 0, 0

            for X, y in loader:
                y_pred = self.predict_teacher(X).argmax(dim=1)
                correct += y_pred.eq(y).sum().item()
                total += len(y)

            return correct / total

    @staticmethod
    def augment(x: torch.Tensor, cols_to_drop: np.array):
        x_ = x.clone()
        x_[np.arange(x_.shape[0]), cols_to_drop] = 0
        return x_

    @staticmethod
    def H(t, s, tps, tpt):
        t = t.detach()  # stop gradient

        s = F.softmax(s / tps, dim=1)
        t = F.softmax(t / tpt, dim=1)  # center + sharpen
        return -(t * torch.log(s)).sum(dim=1).mean()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        lr: float = 0.01,
        w_teacher: float = 0.2,
        n_epochs: int = 300,
        patience: int = 30,
        verbose: bool = True,
        tps: float = 1,
        tpt: float = 1,
        alpha: float = 1,
    ):
        s_optimizer = Adam(self.student.parameters(), lr=lr)
        t_optimizer = Adam(self.teacher.parameters(), lr=lr)
        min_val_acc = -np.inf
        best_model = None
        cur_patience = 0

        train_acc, val_acc = [], []

        criteria = nn.CrossEntropyLoss(reduction="mean")

        # initialize teacher parameters as average of student parameters
        for teacher_param, student_params in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_param.data = student_params.data

        for epoch in range(n_epochs):
            for X, y in train_loader:
                s_optimizer.zero_grad()
                t_optimizer.zero_grad()

                feats_to_drop = np.random.choice(
                    X.shape[1], size=X.shape[0] * 2, replace=True
                )

                x1 = self.augment(X, cols_to_drop=feats_to_drop[: X.shape[0]])
                x2 = self.augment(X, cols_to_drop=feats_to_drop[X.shape[0] :])

                [s1, t1], [s2, t2] = self.forward(x1, tps, tpt), self.forward(x2, tps, tpt)

                student_loss = (
                    criteria(s1, t2) + criteria(s2, t1)
                ) / 2 + alpha * criteria(self.teacher(X), y)

                # student_loss = (self.H(s1, t2, tps=tps, tpt=tpt) + self.H(s2, t1, tps=tps,
                #                                                           tpt=tpt)) / 2
                student_loss.backward()

                s_optimizer.step()

                del x1, x2, s1, s2, t1, t2

                # teacher_loss = alpha * criteria(self.predict_teacher(X), y)
                # teacher_loss.backward()
                #
                # t_optimizer.step()

                for teacher_param, student_param in zip(
                    self.teacher.parameters(), self.student.parameters()
                ):
                    teacher_param.data = (
                        teacher_param.data * w_teacher
                        + student_param.data * (1 - w_teacher)
                    )

            train_score = self.score(train_loader)
            val_score = self.score(val_loader)

            train_acc.append(train_score)
            val_acc.append(val_score)

            if verbose:
                print(
                    f"Epoch {epoch + 1}:\n"
                    f"\tTrain accuracy: {train_score}\n"
                    f"\tVal accuracy: {val_score}"
                )

            if val_score > min_val_acc:
                min_val_acc = val_score
                best_model = self.state_dict()
                cur_patience = 0

            else:
                cur_patience += 1
                if cur_patience == patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.load_state_dict(best_model)

        # test accuracy
        full_test_acc = self.score(test_loader)

        partial_test_acc = []

        for i in range(test_loader.dataset.X.shape[1]):
            test_loader_copy = DataLoader(
                test_loader.dataset, batch_size=1, shuffle=False
            )
            test_loader_copy.dataset.X[:, i] = test_loader_copy.dataset.X[:, i].mean()

            partial_test_acc.append(self.score(test_loader_copy))

        return full_test_acc, np.mean(partial_test_acc), np.std(partial_test_acc)
