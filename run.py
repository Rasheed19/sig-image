if __name__ == "__main__":
    # from sklearn.datasets import load_digits

    # mnist = load_digits()
    # features = mnist["data"]
    # labels = mnist["target"]

    # zhn = Zhang2DSignature()

    # for i in range(features.shape[0]):
    #     sample = features[i].reshape(3, 4, 4)
    #     print(
    #         zhang_sig(sample, channel=3),
    #     )

    # Import necessary libraries
    import numpy as np
    from tensorflow.keras.datasets import cifar10

    from shared.signature import Diehl2DSignature

    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Display the shape of the dataset
    print(f"Train images shape: {train_images[0].shape}")
    print(train_images[0].reshape(3, 32, 32).shape)

    sample = train_images[0].reshape(3, 32, 32)
    sample = sample / np.mean(sample)
    # zh = Zhang2DSignature()
    # gs = Giusti2DSignature()

    ds = Diehl2DSignature()

    print(ds.first_level_sig(image=sample))
    print(ds.second_level_sig(image=sample))

    # channel_pairs = list(itertools.product([1, 2, 3], repeat=2))
    # from shared.signature import Giusti2DSignature

    # gs = Giusti2DSignature()

    # all_comb = [
    #     (
    #         gs._injection(1, deg[0]),
    #         gs._injection(2, deg[0]),
    #         gs._injection(1, deg[1]),
    #         gs._injection(2, deg[1]),
    #     )
    #     for deg in channel_pairs
    # ]
    # all_comb = set(all_comb)

    # all_comb2 = [
    #     (
    #         gs._injection(1, deg[0]),
    #         gs._injection(1, deg[1]),
    #         gs._injection(2, deg[1]),
    #         gs._injection(2, deg[0]),
    #     )
    #     for deg in channel_pairs
    # ]
    # all_comb2 = set(all_comb2)

    # all_comb3 = [
    #     (
    #         gs._injection(2, deg[1]),
    #         gs._injection(1, deg[0]),
    #         gs._injection(1, deg[1]),
    #         gs._injection(2, deg[0]),
    #     )
    #     for deg in channel_pairs
    # ]
    # all_comb4 = [
    #     (
    #         gs._injection(2, deg[1]),
    #         gs._injection(1, deg[0]),
    #         gs._injection(2, deg[0]),
    #         gs._injection(1, deg[1]),
    #     )
    #     for deg in channel_pairs
    # ]
    # all_comb4 = set(all_comb4)

    # print(all_comb)
    # print(all_comb2)

    # print(len(all_comb))
    # print(len(all_comb2))

    # print(all_comb.intersection(all_comb4))
