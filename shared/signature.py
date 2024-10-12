import itertools

import torch


class Zhang2DSignature:
    def _inner_double_sum(self, tensor: torch.Tensor, k1: int, k2: int) -> float:
        return torch.sum(
            (tensor[1:k1, 0 : k2 - 1] - tensor[0 : k1 - 1, 0 : k2 - 1])
            * (tensor[0 : k1 - 1, 1:k2] - tensor[0 : k1 - 1, 0 : k2 - 1])
        ).item()

    def _first_level_sig(self, image: torch.Tensor) -> list[float]:
        n = image[0].shape[0]
        return (
            torch.Tensor(
                [
                    (
                        (
                            tensor[n - 1, n - 1]
                            - tensor[0, n - 1]
                            - tensor[n - 1, 0]
                            + tensor[0, 0]
                        ).item(),
                        self._inner_double_sum(tensor=tensor, k1=n, k2=n),
                    )
                    for tensor in image
                ]
            )
            .flatten()
            .tolist()
        )

    def _second_level_sig(self, image: torch.Tensor) -> list[float]:
        channel = image.shape[0]
        # channel_pairs = list(
        #     itertools.product(range(channel), range(channel))
        # )  # TODO: this includes interaction terms
        channel_pairs = [(i, i) for i in range(channel)]  # for the case i = j

        sig = torch.zeros(
            (len(channel_pairs), 4)
        )  # col 0: x_i_j, col 1: x_i_hat_j_hat, col 2: x_i_hat_j, col 3: x_i_j_hat

        for i, pair in enumerate(channel_pairs):
            tensor_i, tensor_j = image[pair[0]], image[pair[1]]
            n = tensor_i.shape[0]

            x_i_j = torch.sum(
                (
                    tensor_i[0 : n - 1, 0 : n - 1]
                    - tensor_i[0, 0 : n - 1]
                    - tensor_i[0 : n - 1, 0].reshape(
                        -1, 1
                    )  # must be reshaped to match column definition; the term extracts column
                    + tensor_i[0, 0]
                )
                * (
                    tensor_j[1:n, 1:n]
                    - tensor_j[0 : n - 1, 1:n]
                    - tensor_j[1:n, 0 : n - 1]
                    + tensor_j[0 : n - 1, 0 : n - 1]
                )
            ).item()

            x_i_j_hat = torch.sum(
                (
                    tensor_i[0 : n - 1, 0 : n - 1]
                    - tensor_i[0, 0 : n - 1]
                    - tensor_i[0 : n - 1, 0].reshape(
                        -1, 1
                    )  # must be reshaped to match column definition; the term extracts column
                    + tensor_i[0, 0]
                )
                * (tensor_j[1:n, 0 : n - 1] - tensor_j[0 : n - 1, 0 : n - 1])
                * (tensor_j[0 : n - 1, 1:n] - tensor_j[0 : n - 1, 0 : n - 1])
            ).item()

            sig[i, 0] = x_i_j
            sig[i, 3] = x_i_j_hat

            x_i_hat_j_hat = 0.0
            x_i_hat_j = 0.0

            for k1 in range(n - 1):
                for k2 in range(n - 1):
                    x_i_hat_j_hat += (
                        self._inner_double_sum(tensor=tensor_i, k1=k1 + 1, k2=k2 + 1)
                        * (tensor_j[k1 + 1, k2] - tensor_j[k1, k2])
                        * (tensor_j[k1, k2 + 1] - tensor_j[k1, k2])
                    )
                    x_i_hat_j += self._inner_double_sum(
                        tensor=tensor_i, k1=k1 + 1, k2=k2 + 1
                    ) * (
                        tensor_j[k1 + 1, k2 + 1]
                        - tensor_j[k1, k2 + 1]
                        - tensor_j[k1 + 1, k2]
                        + tensor_j[k1, k2]
                    )

            sig[i, 1] = x_i_hat_j_hat
            sig[i, 2] = x_i_hat_j

        return sig.flatten().tolist()

    def _all_sig_levels(self, image: torch.Tensor) -> list[float]:
        all_levels = self._first_level_sig(image=image)
        all_levels.extend(self._second_level_sig(image=image))

        return all_levels

    def calculate_batch_sig(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        assert depth in (1, 2), "Depth can only take value 1 or 2."

        if depth == 1:
            return torch.Tensor(list(map(self._first_level_sig, x)))

        return torch.Tensor(list(map(self._all_sig_levels, x)))

    def calculate_feature_dim(self, channels: int, depth: int) -> int:
        # level 1 + level 2 number of features
        assert depth in (1, 2), "Depth can only take value 1 or 2."
        # return (
        #     channels * 2 if depth == 1 else 2 * channels * (1 + 2 * channels)
        # )  # TODO: includes iteraction terms
        return (
            channels * 2 if depth == 1 else 6 * channels
        )  # 2c + 4c when no interaction terms are considered


class Diehl2DSignature:
    def _brute_force(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        result = 0.0
        n = tensor1.shape[0]

        for k1 in range(n - 1):
            for k2 in range(n - 1):
                diff1 = (
                    tensor1[k1, k2] - tensor1[k1, 0] - tensor1[0, k2] + tensor1[0, 0]
                )
                diff2 = (
                    tensor2[k1 + 1, k2 + 1]
                    - tensor2[k1, k2 + 1]
                    - tensor2[k1 + 1, k2]
                    + tensor2[k1, k2]
                )

                result += diff1 * diff2

        return result

    def _sum_expression(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        n = tensor1.shape[0]

        return torch.sum(
            (
                tensor1[0 : n - 1, 0 : n - 1]
                - tensor1[0 : n - 1, 0].reshape(
                    -1, 1
                )  # must be reshaped to match column definition; the term extracts column
                - tensor1[0, 0 : n - 1]
                + tensor1[0, 0]
            )
            * (
                tensor2[1:n, 1:n]
                - tensor2[0 : n - 1, 1:n]
                - tensor2[1:n, 0 : n - 1]
                + tensor2[0 : n - 1, 0 : n - 1]
            )
        ).item()

    def _expression(self, tensor: torch.Tensor) -> float:
        n = tensor.shape[0]

        return (
            tensor[n - 1, n - 1] - tensor[n - 1, 0] - tensor[0, n - 1] + tensor[0, 0]
        ).item()

    def _first_level_sig(self, image: torch.Tensor) -> list[float]:
        return [self._expression(tensor=arr) for arr in image]

    def _second_level_sig(self, image: torch.Tensor) -> list[float]:
        channel = image.shape[0]
        channel_pairs = list(itertools.product(range(channel), range(channel)))

        return [
            self._sum_expression(tensor1=image[i], tensor2=image[j])
            for (i, j) in channel_pairs
        ]

    def _all_sig_levels(self, image: torch.Tensor) -> list[float]:
        all_levels = self._first_level_sig(image=image)
        all_levels.extend(self._second_level_sig(image=image))

        return all_levels

    def calculate_batch_sig(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        assert depth in (1, 2), "Depth can only take value 1 or 2."

        if depth == 1:
            return torch.Tensor(list(map(self._first_level_sig, x)))

        return torch.Tensor(list(map(self._all_sig_levels, x)))

    def calculate_feature_dim(self, channels: int, depth: int) -> int:
        # level 1 + level 2 number of features
        assert depth in (1, 2), "Depth can only take value 1 or 2."
        return (
            channels
            if depth == 1
            else channels
            + len(list(itertools.product(range(channels), range(channels))))
        )


if __name__ == "__main__":
    batch = torch.ones(1, 2, 8, 8)

    zs = Zhang2DSignature()
    res = zs.calculate_batch_sig(batch, 2)
    print(res)
    print(res.shape)
    print(zs.calculate_feature_dim(channels=2, depth=2))
