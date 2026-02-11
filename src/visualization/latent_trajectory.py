from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_latent_trajectory(z, dates, title=""):
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(z)

    plt.figure(figsize=(6, 6))
    plt.plot(z2[:, 0], z2[:, 1], alpha=0.7)
    plt.scatter(z2[:, 0], z2[:, 1], c=range(len(z2)), cmap="viridis", s=10)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Time")
    plt.show()
