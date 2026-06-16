import numpy as np


class Logger:
    def __init__(
        self,
        name: str,
        dataset: str,
        window_size: int = 10,
        rounding: int = 4,
    ):
        """
        Args:
            name (str): Name of the metric
            dataset (str): Name of the dataset
            window_size (int, optional): Size of the moving average window. Defaults to 10.
            rounding (int, optional): Number of decimals to round to. Defaults to 4.
        """
        self.data = []
        self.name = name
        self.dataset = dataset
        self.rounding = rounding
        self.window_size = window_size

    def display(self) -> str:
        """
        Returns:
            str: latest value, moving average, and overall average, formatted for printing
        """
        latest = round(self.latest(), self.rounding)
        average = round(self.average(), self.rounding)
        moving_average = round(self.moving_average(), self.rounding)
        output = f"{self.full_name()}: {latest} ({self.window_size}pt moving_avg: {moving_average}) (avg: {average})"
        return output

    def log_data(self, data: float):
        """
        Append a value to the log, skipping NaNs.

        Args:
            data (float): value to record
        """
        if not np.isnan(data):
            self.data.append(data)

    def full_name(self) -> str:
        """
        Returns:
            str: metric name combined with the dataset it was logged on
        """
        return f"{self.name} ({self.dataset})"

    def latest(self) -> float:
        """
        Returns:
            float: most recently logged value, or NaN if nothing has been logged yet
        """
        if len(self.data) > 0:
            return self.data[-1]
        return np.nan

    def average(self) -> float:
        """
        Returns:
            float: mean of all logged values, or NaN if nothing has been logged yet
        """
        if len(self.data) > 0:
            return np.mean(self.data)
        return np.nan

    def moving_average(self) -> float:
        """
        Returns:
            float: mean of the last window_size logged values, falling back to the overall average if fewer have been logged
        """
        if len(self.data) > self.window_size:
            return np.mean(self.data[-self.window_size :])
        return self.average()