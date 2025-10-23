import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.special import binom

import time as time
from utils.utils import pairs_to_sqform, select_pairs

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------------
# Classification function and gradient
# -------------------------------------
# Classification function and gradient
def S_multi_class_fixed(Is, N, p=2):
    # Get intra-class indices in squareform
    pairs = []
    for I in Is:
        pairs.extend(pairs_to_sqform(I, N))

    # Size of squareform matrix
    N_sq = int(binom(N, 2))

    # Define score function
    def S(D, p=p):
        if D.ndim == 2:
            D = squareform(D, checks=False)
        elif D.ndim not in [1, 2]:
            raise ValueError(
                f"D must be a 2d distance matrix or, preferably, a 1d condensed matrix. ndims={D.ndim}"
            )

        score = np.sum(D[pairs] ** p)
        return score / np.sum(D**p)

    # Gradient
    def dS(D, p=p):
        if D.ndim == 2:
            D = squareform(D, checks=False)
        elif D.ndim not in [1, 2]:
            raise ValueError(
                f"D must be a 2d distance matrix or, preferably, a 1d condensed matrix. ndims={D.ndim}"
            )

        score = S(D)

        # Matrix of intra-class ones
        blocks = np.zeros(N_sq)
        blocks[pairs] = 1

        grad = blocks - score
        grad = (p * D ** (p - 1) / np.sum(D**p)) * grad

        return squareform(grad)

    # Return functions
    return S, dS


# Returns a function that computes a classification score
# given the distance matrix of a set with two classes.
# Specifically, the score is the squared sum of interclass
# distances divided by the squared sum of all distances
# I = list with indices of first class
# N = total number of points
def S_two_class_generic(N):
    def S(D, y):
        if D.ndim == 2:
            D = squareform(D)
        elif D.ndim not in [1, 2]:
            raise ValueError(
                f"D must be a 2d distance matrix or, preferably, a 1d condensed matrix. ndims={D.ndim}"
            )

        I = np.where(y)[0]
        J = np.setdiff1d(range(len(y)), I)

        pairs = select_pairs(I, J)

        score = np.sum(D[pairs] ** 2)

        return score / np.sum(D**2)

    return S


# Classification function and gradient
def S_two_class_fixed(I, N, p=2):
    # Complement class
    J = np.setdiff1d(range(N), I)

    # Call multiclass function on two classes
    S, dS = S_multi_class_fixed([I, J], N, p=p)
    return S, dS


# -------------------------------------
# Generating classes of dynamic metric spaces
# -------------------------------------
def moving_point_grid(shape, dx, obstacle, nSteps):
    """
    shape: tuple of 2 integers with shape of robot grid
    dx: float. Distance that robots move at each step
    center: tuple with the center of the ellipse
    radii:  tuple with the radii (x, y) of the ellipse
    nSteps: number of steps robots take in the room
    """
    nx, ny = shape

    # Create template for robot/drone grid
    # --------------------
    # Create x coordinates
    xx = np.linspace(0, 1, nx)
    yy = np.linspace(-1, 1, ny)

    # Combine x and y and turn into a 2D array
    # Meshgrid combines xx and yy, and I reshape the result
    P = np.meshgrid(xx, yy)  # P has 2 arrays of shape (ny, nx)
    P = [v.flatten() for v in P]  # P has 2 arrays of shape (nx*ny,)
    P = np.array(P).T  # Shape [nx*ny,2]

    # # Move drone grid across a room without obstacles
    # # --------------------
    # P_seq_1 = np.zeros((nSteps, nx*ny, 2))
    # for idt in range(nSteps):
    #     X = P[:,0] + dx*idt
    #     Y = P[:,1]

    #     P_seq_1[idt,:,0] = X
    #     P_seq_1[idt,:,1] = Y

    # Move drone grid across a room with a circular obstacle
    # NOTE: If the I produced by obstacle is empty, we sample
    # as if there were no obstacle
    # --------------------
    # Each page of the arrays below is in the same format as
    # the result of np.meshgrid.
    # Later we'll merge them as we did above
    X_seq_2 = np.zeros((nSteps, nx, ny))
    Y_seq_2 = np.zeros((nSteps, nx, ny))

    # Top and half of the drones (0 is in the bottom half)
    J_bot = np.arange(ny / 2).astype(int)
    J_top = np.arange(np.ceil(ny / 2), ny).astype(int)

    for idx in range(shape[0]):
        x0 = xx[idx]
        x_move = x0 + dx * np.arange(nSteps)

        # Store X
        for idy in range(ny):
            X_seq_2[:, idx, idy] = x_move

        # Find position of obstacle at all the x coordinates we visit
        top, bot, I = obstacle(x_move)

        # For each x position, create a list of y coordinates that
        # avoid the obstacle
        for idt in range(nSteps):
            # No obstacle
            if idt not in I:
                Y_seq_2[idt, idx, :] = yy

            # Obstacle present
            else:
                # Put half of the points above the obstacle
                # and half below
                y_bot = np.linspace(-1, bot[idt], len(J_bot))

                # If there is only one point in the top, we set it
                # as 1
                if len(J_top) == 1:
                    y_top = 1
                # Otherwise, we avoid the obstacle
                else:
                    y_top = np.linspace(top[idt], 1, len(J_top))

                Y_seq_2[idt, idx, J_top] = y_top
                Y_seq_2[idt, idx, J_bot] = y_bot

    # At each step, gather x and y coordinates into a single array
    P_seq_2 = np.zeros((nSteps, nx * ny, 2))
    for idt in range(nSteps):
        # We just transpose X and Y to have the exact same format as meshgrid
        P_2 = [X_seq_2[idt, :, :].T, Y_seq_2[idt, :, :].T]
        P_2 = [v.flatten() for v in P_2]

        P_seq_2[idt, :, 0] = P_2[0]
        P_seq_2[idt, :, 1] = P_2[1]

    return P_seq_2


def add_noise_avoid_obstacle(P_seq, std, center, obstacle, rng=None):
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    # Apply noise
    # P_seq_noise = P_seq + rng.uniform(-std, std, P_seq.shape)
    P_seq_noise = P_seq + rng.normal(0, std, size=P_seq.shape)

    # Find position of obstacle at each x
    top, bot, I = obstacle(P_seq_noise[:, 0])

    # Are we above or below the center of the obstacle
    if center is not None:
        cx, cy = center
    else:
        cx, cy = 0, 0
    I_pos = np.where(P_seq_noise[:, 1] > cy)[0]
    I_neg = np.where(P_seq_noise[:, 1] < cy)[0]

    # Ensure we are above the obstacle in positive heights
    P_seq_noise[I_pos, 1] = np.maximum(P_seq_noise[I_pos, 1], top[I_pos])

    # ... and below the obstacle in negative heights
    P_seq_noise[I_neg, 1] = np.minimum(P_seq_noise[I_neg, 1], bot[I_neg])

    return P_seq_noise


def create_obstacle_fun(center, radii):
    # If we have don't have center or radii, we return an empty obstacle
    if center is None or radii is None:

        def obstacle(x):
            top = -np.ones_like(x)
            bot = np.ones_like(x)
            I = np.zeros(0, dtype=int)

            return top, bot, I

    # If we have center and radii, we return an obstacle function
    else:
        cx, cy = center
        a, b = radii

        # Position of obstacle (x should be a 1D array)
        # Also returns an array indicating when x is inside of the domain
        def obstacle(x):
            # Domain of the obstacle
            # Don't include the boundaries of the x-axis
            I = np.where(np.logical_and(cx - a < x, x < cx + a))[0]

            # Find the top and bottom of the obstacle
            # Note: top=-1 and bot=1 outside of the domain
            top = -1 * np.ones_like(x)
            bot = 1 * np.ones_like(x)

            # Inside of the domain
            top[I] = cy + (b / a) * np.sqrt(a**2 - (x[I] - cx) ** 2)
            bot[I] = cy - (b / a) * np.sqrt(a**2 - (x[I] - cx) ** 2)

            return top, bot, I

    # Return the function we created
    return obstacle


# -------------------------------------
# Function to run several classification experiments
# -------------------------------------
def classification_experiment(n_neighbors, n_classes, dms_train, dms_test=None):
    # If we don't have test data, we use the train data as test
    if dms_test is None:
        dms_test = dms_train

    # Extract info from training and test sets
    dGWs_train, dMSs_train, y_train = dms_train
    dGWs_test, dMSs_test, y_test = dms_test

    # Shapes in the experiment
    nSteps = dGWs_train.shape[0]
    n_train = dGWs_train.shape[1]
    n_test = dGWs_test.shape[1]

    # Set up classifier
    knn = KNeighborsClassifier(n_neighbors)

    # Clustering accuracy
    predictions = np.zeros((nSteps, n_test))
    pred_probs = np.zeros((nSteps, n_test, n_classes))
    accuracy = np.zeros(nSteps)
    for t in range(nSteps):
        # Fit KNN with a GW matrix from the training set
        dm_train = dGWs_train[t, :, :]
        clusters = knn.fit(dm_train, y_train)

        # Predict on the corresponding GW matrix from the test set
        dm_test = dGWs_test[t, :, :]
        y_pred = clusters.predict(dm_test)
        predictions[t, :] = y_pred
        accuracy[t] = accuracy_score(y_test, y_pred)

        # Store probabilities for the ensemble
        pred_probs[t, :, :] = clusters.predict_proba(dm_test)

    # Fit KNN with the MS matrix from the training set
    clusters = knn.fit(dMSs_train, y_train)

    # Predict with the MS matrix from the test set
    prediction_ms = clusters.predict(dMSs_test)
    accuracy_ms = accuracy_score(y_test, prediction_ms)

    # Ensembles
    # Hard max
    max_probs = np.max(pred_probs, axis=2)
    winners = pred_probs == max_probs[:, :, None]

    votes = np.sum(winners, axis=0)
    y_pred_hard = votes.argmax(axis=1)

    # Soft max
    probs_soft = np.sum(pred_probs, axis=0)
    y_pred_soft = probs_soft.argmax(axis=1)

    accuracy_hard = accuracy_score(y_test, y_pred_hard)
    accuracy_soft = accuracy_score(y_test, y_pred_soft)

    # Print results
    with np.printoptions(precision=3):
        print("True labels:")
        print(y_test)
        print()

        print("Individual invariants")
        print(predictions)
        print(accuracy)
        print()

        print("Learned multiscale")
        print(prediction_ms)
        print(accuracy_ms)
        print()

        print("GW Ensemble -- hard voting")
        print(accuracy_hard)
        print()

        print("GW Ensemble -- soft voting")
        print(accuracy_soft)

    return pred_probs, accuracy, accuracy_ms, accuracy_hard, accuracy_soft


# -------------------------------------
# Graphing functions
# -------------------------------------
def plot_drone_grid(P_seq, room_length, center, radii, plot_obstacle=False, seq_id=1):
    # Functions to plot the obstacle
    if plot_obstacle:
        cx, cy = center
        a, b = radii

        tt = np.linspace(0, 2 * np.pi, 50)
        xx = cx + a * np.cos(tt)
        yy = cy + b * np.sin(tt)

    nSteps = P_seq.shape[0]
    fig, axes = plt.subplots(1, nSteps, figsize=(10, 10))
    for t in range(nSteps):
        axes[t].plot(P_seq[t, :, 0], P_seq[t, :, 1], ".")

        axes[t].set_xlim(-0.5, room_length + 1)
        axes[t].set_ylim(-2, 2)
        axes[t].set_aspect("equal", "box")

        axes[t].set_title(f"Seq. {seq_id}, t={t}")

        if plot_obstacle:
            axes[t].plot(xx, yy, "red", linestyle="dotted")

    fig.tight_layout()

    return fig, axes


def plot_score_and_deltas(
    score_list, abs_delta_list, rel_delta_list, nu_list, plot_squares=True
):
    nu = nu_list[-1]

    # Part 1: Classification score and deltas
    fig1, axes1 = plt.subplots(1, 4, figsize=(15, 3))
    fig1.suptitle("Evolution of classification score")

    tt = np.arange(len(score_list))
    axes1[0].plot(tt, score_list)
    axes1[0].plot(tt[1::2], score_list[1::2], "r*--")
    axes1[0].set_title("Score")

    axes1[1].plot(tt, abs_delta_list)
    axes1[1].plot(tt[1::2], abs_delta_list[1::2], "r*--")
    axes1[1].set_title("Absolute Delta")
    axes1[1].set_yscale("symlog", linthresh=1e-10)

    axes1[2].plot(tt, rel_delta_list)
    axes1[2].plot(tt[1::2], rel_delta_list[1::2], "r*--")
    axes1[2].set_title("Relative Delta")
    axes1[2].set_yscale("symlog", linthresh=1e-10)

    # Compute difference between even (nu) and odd (pi) steps
    score_list = np.array(score_list)
    even_odd_diff = score_list[1::2] - score_list[::2]
    axes1[3].plot(even_odd_diff)
    axes1[3].set_title("Score change after fitting nu")
    axes1[3].set_yscale("symlog", linthresh=1e-10)

    # Part 2: evolution of nu
    if plot_squares:
        nu_arr = np.array(nu_list).T

        # Compute differences
        nu_abs = np.abs(np.diff(nu_arr, axis=1))

        # Plot
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 3))
        fig2.suptitle("Evolution of nu")

        # axes2[0].plot(nu_list)
        im1 = axes2[0].imshow(nu_arr)
        axes2[0].set_title("nu")
        plt.colorbar(im1)

        # axes2[1].plot(nu_abs)
        im2 = axes2[1].imshow(nu_abs)
        axes2[1].set_title("Delta on nu")
        plt.colorbar(im2)

    else:
        # Convert to array
        nu_arr = np.array(nu_list)

        # Compute differences
        nu_abs = np.abs(np.diff(nu_arr, axis=0))
        nu_abs = np.pad(nu_abs, [[1, 0], [0, 0]], mode="constant")

        # Plot
        fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3))
        fig2.suptitle("Evolution of nu")

        axes2[0].plot(nu_arr)
        axes2[0].set_title("nu")
        axes2[0].set_yscale("symlog", linthresh=1e-10)
        axes2[0].legend(np.arange(len(nu)) + 1)

        axes2[1].plot(nu_abs)
        axes2[1].set_title("Delta on nu")
        axes2[1].set_yscale("symlog", linthresh=1e-3)
        axes2[1].legend(np.arange(len(nu)) + 1)

    return fig1, axes1, fig2, axes2


def MS_output(dGWs, dMSs, nu, nSteps, S, invariant_names):
    # Part 1: Compare classification scores
    with np.printoptions(precision=3):
        for t in range(nSteps):
            print(invariant_names[t])
            print("Score: {:3f}".format(S(dGWs[t, :, :], p=1)))
            print()

        print("Metric learning:")
        print(nu)
        print("Score: {:3f}".format(S(dMSs, p=1)))

    # Part 2: Show distance matrices
    fig, axes = plt.subplots(1, nSteps + 1, figsize=(5 * (nSteps + 1), 5))
    fig.suptitle("GW and MS distance matrices")

    for t in range(nSteps):
        im = axes[t].imshow(dGWs[t, :, :], vmin=0)
        axes[t].set_title(invariant_names[t])
        plt.colorbar(im, ax=axes[t])

    im = axes[nSteps].imshow(dMSs, vmin=0)
    axes[nSteps].set_title(f"Learned")
    plt.colorbar(im, ax=axes[nSteps])

    return fig, axes, im
