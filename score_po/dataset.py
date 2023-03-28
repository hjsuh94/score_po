import torch


class Dataset:  # Unsupervised Dataset
    def __init__(self, dim_x, dim_u):
        """
        Dataset class that stores the observed values of (x,u).
        """
        self.dim_x = dim_x
        self.dim_u = dim_u

        self.data = torch.zeros(0, dim_x + dim_u)

    def add_to_dataset(self, batch_samples):
        """
        input:
            batch_samples of shape (B, dim_x + dim_u) or (dim_x + dim_u)
        """
        # If data is single dim, append a dimension.
        if len(batch_samples.shape) == 1:
            batch_samples = batch_samples[None, :]

        # Check for validity of data.
        if batch_samples.shape[1] is not self.dim_x + self.dim_u:
            raise ValueError("add_to_dataset: tried to insert data of wrong shape.")

        self.data = torch.vstack((self.data, batch_samples))

    def draw_from_dataset(self, num_samples):
        """
        randomly draw from dataset.
        """
        B = self.data.shape[0]
        # NOTE(terry-suh): seems inefficient.
        idx = torch.randperm(B)[:num_samples]
        return self.data[idx, :]


class DynamicsDataset(Dataset):  # Supervised Dataset
    def __init__(self, dim_x, dim_u):
        """
        Dataset class that stores the observed values of (x,u) as well as f(x,u).
        """
        self.dim_x = dim_x
        self.dim_u = dim_u

        self.data = torch.zeros(0, dim_x + dim_u)
        self.label = torch.zeros(0, dim_x)

    def add_to_dataset(self, batch_samples, batch_labels):
        """
        input:
            batch_samples of shape (B, dim_x + dim_u) or (dim_x + dim_u)
        """
        # If data is single dim, append a dimension.
        if len(batch_samples.shape) == 1:
            batch_samples = batch_samples[None, :]
        if len(batch_labels.shape) == 1:
            batch_labels = batch_labels[None, :]

        if batch_samples.shape[0] is not batch_samples.shape[1]:
            raise ValueError("wrong value for data and label.")

        # Check for validity of data.
        if (batch_samples.shape[1] is not self.dim_x + self.dim_u) or (
            batch_labels.shape[1] is not self.dim_x
        ):
            raise ValueError("add_to_dataset: tried to insert data of wrong shape.")

        self.data = torch.vstack((self.data, batch_samples))
        self.label = torch.vstack((self.laba, batch_labels))

    def draw_from_dataset(self, num_samples):
        """
        randomly draw from dataset.
        """
        B = self.data.shape[0]
        # NOTE(terry-suh): seems inefficient.
        idx = torch.randperm(B)[:num_samples]
        return self.data[idx, :], self.label[idx, :]


def test():
    dataset = Dataset(8, 3)
    dataset.add_to_dataset(torch.rand(10000, 11))
    dataset.add_to_dataset(torch.rand(11))
    print(dataset.data.shape)
    samples = dataset.draw_from_dataset(100)
    print(samples.shape)
    try:
        dataset.add_to_dataset(torch.rand(10000, 3))
    except ValueError as e:
        print(e)
