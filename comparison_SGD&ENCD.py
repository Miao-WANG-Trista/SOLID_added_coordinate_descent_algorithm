{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled3.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOV6rU5bYBPtasB08/VBn1X",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/felixbastian/SOLID_add-on_coordinate-descent/blob/main/comparison_SGD%26ENCD.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yNiy-v7eBJcT"
      },
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import random\n",
        "import time\n",
        "import ElasticNetCD as encd\n",
        "import sklearn\n",
        "from sklearn.datasets import fetch_california_housing\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "def sgd(samples, y, step_size=0.005, max_iteration_count=100):\n",
        "    sample_num, dimension = samples.shape\n",
        "    w = np.ones((dimension,1), dtype=np.float32)\n",
        "    loss_collection = []\n",
        "    loss = 1\n",
        "    iteration_count = 0\n",
        "    while loss > 0.001 and iteration_count < max_iteration_count:\n",
        "        loss = 0\n",
        "        gradient = np.zeros((dimension,1), dtype=np.float32)\n",
        "        \n",
        "        #Randomly choose a sample to update the weights\n",
        "        sample_index = random.randint(0, sample_num-1)\n",
        "        predict_y = np.dot(w.T, samples[sample_index])\n",
        "        for j in range(dimension):\n",
        "            gradient[j] += (predict_y - y[sample_index]) * samples[sample_index][j]\n",
        "            w[j] -= step_size * gradient[j]\n",
        "\n",
        "        for i in range(sample_num):\n",
        "            predict_y = np.dot(w.T, samples[i])\n",
        "            loss += np.power((predict_y - y[i]), 2)\n",
        "\n",
        "        loss_collection.append(loss / (2 * len(y)))\n",
        "        iteration_count += 1\n",
        "    return w,loss_collection\n",
        "\n",
        "#  california house-prices dataset\n",
        "data = fetch_california_housing(as_frame=True)\n",
        "X, y = data.data, data.target\n",
        "X = StandardScaler().fit_transform(X)\n",
        "\n",
        "#Application of EN\n",
        "start1 = time.time()\n",
        "B_hat, cost_history, objective = encd.ElasticNetCD.elastic_net(X, y, 0.8, 0.3, 1e-4, 100)\n",
        "end1 = time.time()\n",
        "\n",
        "#Application of SGD\n",
        "start2 = time.time()\n",
        "bret, bxret = sgd(X, y, step_size=0.005, max_iteration_count=100)\n",
        "end2 = time.time()\n",
        "\n",
        "#Print the running time for each algorithm\n",
        "print (\"The running time of EN is \" + str(end1-start1))\n",
        "print (\"The running time of SGD is \" + str(end2-start2))\n",
        "\n",
        "#plot the loss function\n",
        "plt.plot(range(len(cost_history)), cost_history, label=\"EN\", color='r')\n",
        "plt.title('SGD & EN Loss')\n",
        "plt.xlabel('Iterations (Path length)')\n",
        "plt.ylabel('Loss function')\n",
        "plt.plot(range(len(bxret)), bxret, label=\"SGD\")\n",
        "plt.legend()\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "XNiHS8WpBP0S"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lXJj2BuuBP10"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KWvmZdpSBP3w"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Wus4IMA3Ba8f"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Cj5PZhrrCBbd"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iyUHX9FjBa-X"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}