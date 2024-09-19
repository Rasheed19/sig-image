import itertools

import numpy as np


class Zhang2DSignature:
    def _brute_force(self, arr: np.ndarray) -> float:
        n = arr.shape[0]
        result = 0.0
        result_hat = 0.0
        for k1 in range(n - 1):
            for k2 in range(n - 1):
                result += self._expression(arr=arr, k1=k1, k2=k2)
                result_hat += self._expression_hat(arr=arr, k1=k1, k2=k2)

        return result, result_hat

    def _expression(self, arr: np.ndarray, k1: int, k2: int) -> float:
        return arr[k1 + 1, k2 + 1] - arr[k1, k2 + 1] - arr[k1 + 1, k2] + arr[k1, k2]

    def _expression_hat(self, arr: np.ndarray, k1: int, k2: int) -> float:
        return (arr[k1 + 1, k2] - arr[k1, k2]) * (arr[k1, k2 + 1] - arr[k1, k2])

    def _sum_expression(self, arr: np.ndarray, n: int, m: int) -> float:
        return np.sum(
            arr[1:n, 1:m]  # arr[k1+1, k2+1]
            - arr[0 : n - 1, 1:m]  # arr[k1, k2+1]
            - arr[1:n, 0 : m - 1]  # arr[k1+1, k2]
            + arr[0 : n - 1, 0 : m - 1]  # arr[k1, k2]
        )

    def _sum_expression_hat(self, arr: np.ndarray, n: int, m: int) -> float:
        return np.sum(
            (arr[1:n, 0 : m - 1] - arr[0 : n - 1, 0 : m - 1])
            * (arr[0 : n - 1, 1:m] - arr[0 : n - 1, 0 : m - 1])
        )

    def first_level_sig(self, image: np.ndarray) -> np.ndarray:
        sig = np.zeros(shape=2 * image.shape[0], dtype="float")

        for i, arr in enumerate(image):
            n = arr.shape[0]
            sig[2 * i : 2 * (i + 1)] = (
                self._sum_expression(arr=arr, n=n, m=n),
                self._sum_expression_hat(arr=arr, n=n, m=n),
            )
        return sig

    def second_level_sig(self, image: np.ndarray) -> np.ndarray:
        channel = image.shape[0]
        sig = np.zeros(shape=4 * channel**2, dtype="float")
        channel_pairs = list(itertools.product(range(channel), range(channel)))

        for i, pair in enumerate(channel_pairs):
            arr_sum = np.zeros(4)

            for k1 in range(image[pair[0]].shape[0] - 1):
                for k2 in range(image[pair[0]].shape[0] - 1):
                    if (
                        k1 == 0 or k2 == 0
                    ):  # FIXME: unclear how to fix/implement the inner double sum
                        continue
                    arr_sum += np.array(
                        [
                            self._sum_expression(arr=image[pair[0]], n=k1, m=k2)
                            * self._expression(arr=image[pair[1]], k1=k1, k2=k2),
                            self._sum_expression_hat(arr=image[pair[0]], n=k1, m=k2)
                            * self._expression_hat(arr=image[pair[1]], k1=k1, k2=k2),
                            self._sum_expression_hat(arr=image[pair[0]], n=k1, m=k2)
                            * self._expression(arr=image[pair[1]], k1=k1, k2=k2),
                            self._sum_expression(arr=image[pair[0]], n=k1, m=k2)
                            * self._expression_hat(arr=image[pair[1]], k1=k1, k2=k2),
                        ]
                    )
            sig[4 * i : 4 * (i + 1)] = arr_sum

        return sig


class Giusti2DSignature:
    def _permutation(self, x: int, deg: int) -> int:
        assert (x in [1, 2]) and (deg in [1, 2]), "x and deg must be in [1, 2]."

        if deg == 2:
            if x == 1:
                return 2
            else:
                return 1
        else:
            return x

    def _injection(self, x: int, deg: int) -> int:
        assert (x in [1, 2]) and (
            deg in [1, 2, 3]
        ), "x must be in [1, 2] and deg must be in [1, 2, 3]."

        if deg == 2:
            if x == 2:
                return 3
            else:
                return 1
        elif deg == 3:
            if x == 1:
                return 2
            else:
                return 3
        else:
            return x

    def _expression(
        self, arr1: np.ndarray, arr2: np.ndarray, k1: int, k2: int
    ) -> float:
        return (
            (arr1[k1 + 1, k2] - arr1[k1, k2]) * (arr2[k1, k2 + 1] - arr2[k1, k2])
        ) - ((arr1[k1, k2 + 1] - arr1[k1, k2]) * (arr2[k1 + 1, k2] - arr2[k1, k2]))

    def _brute_force(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        result = 0.0
        n = arr1.shape[0]

        for k1 in range(n - 1):
            for k2 in range(n - 1):
                result += self._expression(arr1=arr1, arr2=arr2, k1=k1, k2=k2)

        return result

    def _sum_expression(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        n = arr1.shape[0]
        return np.sum(
            (
                (arr1[1:n, 0 : n - 1] - arr1[0 : n - 1, 0 : n - 1])
                * (arr2[0 : n - 1, 1:n] - arr2[0 : n - 1, 0 : n - 1])
            )
            - (
                (arr1[0 : n - 1, 1:n] - arr1[0 : n - 1, 0 : n - 1])
                * (arr2[1:n, 0 : n - 1] - arr2[0 : n - 1, 0 : n - 1])
            )
        )

    def first_level_sig(self, image: np.ndarray) -> np.ndarray:
        assert (
            image.shape[0] == 3
        ), f"Number of channels in image must be 3 but {image.shape[0]} is given."

        channel_pairs = list(itertools.combinations([0, 1, 2], 2))

        return np.array(
            [
                self._sum_expression(arr1=image[c1], arr2=image[c2])
                for (c1, c2) in channel_pairs
            ]
        )

    def second_level_sig(self, image: np.ndarray) -> np.ndarray:
        assert (
            image.shape[0] == 3
        ), f"Number of channels in image must be 3 but {image.shape[0]} is given."

        # FIXME: need clarification on how to generate all possible combinations

        raise NotImplementedError()


class Diehl2DSignature:
    def _brute_force_sum_expression(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        result = 0.0
        n = arr1.shape[0]

        for k1 in range(n - 1):
            for k2 in range(n - 1):
                diff1 = arr1[k1, k2] - arr1[k1, 0] - arr1[0, k2] + arr1[0, 0]
                diff2 = (
                    arr2[k1 + 1, k2 + 1]
                    - arr2[k1, k2 + 1]
                    - arr2[k1 + 1, k2]
                    + arr2[k1, k2]
                )

                result += diff1 * diff2

        return result

    def _sum_expression(self, arr1: np.ndarray, arr2: np.ndarray):
        n = arr1.shape[0]

        return np.sum(
            (
                arr1[0 : n - 1, 0 : n - 1]
                - arr1[0 : n - 1, 0].reshape(
                    -1, 1
                )  # must be reshaped to match column definition; the term extracts column
                - arr1[0, 0 : n - 1]
                + arr1[0, 0]
            )
            * (
                arr2[1:n, 1:n]
                - arr2[0 : n - 1, 1:n]
                - arr2[1:n, 0 : n - 1]
                + arr2[0 : n - 1, 0 : n - 1]
            )
        )

    def _expression(self, arr: np.ndarray) -> float:
        n = arr.shape[0]

        return arr[n - 1, n - 1] - arr[n - 1, 0] - arr[0, n - 1] + arr[0, 0]

    def first_level_sig(self, image: np.ndarray) -> np.ndarray:
        return np.array([self._expression(arr=arr) for arr in image])

    def second_level_sig(self, image: np.ndarray) -> np.ndarray:
        channel = image.shape[0]
        channel_pairs = list(itertools.product(range(channel), range(channel)))

        return np.array(
            [
                self._sum_expression(arr1=image[i], arr2=image[j])
                for (i, j) in channel_pairs
            ]
        )


if __name__ == "__main__":
    N = 5  # Example size
    xi = np.random.randn(N, N)

    zh = Zhang2DSignature()
    print("brute force", zh._brute_force(xi))
    print("vectorized", zh._sum_expression_hat(xi, N, N))
