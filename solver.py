from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np
from numpy.typing import NDArray
import torch
import sys


################################################################################
# Circuits
################################################################################


def ansatz_block(four_params: NDArray[np.float64]) -> QuantumCircuit:
    assert len(four_params) == 4

    qc = QuantumCircuit(2)
    qc.ry(four_params[0], 0)
    qc.ry(four_params[1], 1)
    qc.cz(0, 1)
    qc.ry(four_params[2], 0)
    qc.ry(four_params[3], 1)

    return qc


def ansatz_layer(n: int, params: NDArray[np.float64]) -> QuantumCircuit:
    assert len(params) == 4 * n

    qc = QuantumCircuit(n)

    for i in range(0, n - 1, 2):
        four_params = params[:4]
        params = params[4:]
        block = ansatz_block(four_params)
        qc.compose(block, (i, i + 1), inplace=True)

    for i in range(1, n, 2):
        four_params = params[:4]
        params = params[4:]
        block = ansatz_block(four_params)
        qc.compose(block, (i, (i + 1) % n), inplace=True)

    return qc


def ansatz(n: int, layers: int, params: NDArray[np.float64]) -> QuantumCircuit:
    assert len(params) == 4 * n * layers

    qc = QuantumCircuit(n)

    for _ in range(layers):
        layer_params = params[:4 * n]
        params = params[4 * n:]
        layer = ansatz_layer(n, layer_params)
        qc.compose(layer, range(n), inplace=True)

    return qc


################################################################################
# Algorithms
################################################################################


def get_counts(qc: QuantumCircuit, shots: int = 1) -> dict[str, int]:
    backend = AerSimulator()
    job = backend.run(qc, shots=shots)
    return job.result().get_counts()


def counts_to_vec(
    size: int,
    shots: int,
    counts: dict[str, int],
) -> NDArray[np.float64]:
    vec = np.zeros(size, dtype=np.float64)
    for k, v in counts.items():
        i = int(k, base=2)
        vec[i] = np.sqrt(v / shots)
    return vec


def cost(
    n: int,
    ansatz: QuantumCircuit,
    A: NDArray[np.float64],
    m: int,
    E: NDArray[np.float64],
    shots: int,
) -> float:
    rows_A, cols_A = A.shape
    assert rows_A == cols_A
    assert rows_A == 2 ** n
    assert len(E) == m

    total_cost = 0

    for i in range(cols_A):
        Ai = A[:, i]
        Ai_norm_factor = np.linalg.norm(Ai)
        Ai = Ai / Ai_norm_factor

        qc = QuantumCircuit(n, n)
        qc.initialize(list(Ai))
        qc.compose(ansatz, range(n), inplace=True)
        qc.measure(range(n), range(n))

        counts = get_counts(qc, shots)
        col = counts_to_vec(rows_A, shots, counts) * Ai_norm_factor

        col_cost = np.sum(col[:m] ** 2 * E)

        total_cost += col_cost

    return total_cost


def grad(
    n: int,
    ansatz_layers: int,
    params: NDArray[np.float64],
    A: NDArray[np.float64],
    m: int,
    E: NDArray[np.float64],
    shots: int,
    param_idx: int,
) -> float:
    params_plus = params.copy()
    params_plus[param_idx] += np.pi / 2
    ansatz_plus = ansatz(n, ansatz_layers, params_plus)
    cost_plus = cost(n, ansatz_plus, A, m, E, shots)

    params_minus = params.copy()
    params_minus[param_idx] -= np.pi / 2
    ansatz_minus = ansatz(n, ansatz_layers, params_minus)
    cost_minus = cost(n, ansatz_minus, A, m, E, shots)

    return (cost_plus - cost_minus) / 2


def read_ansatz(n: int, ansatz: QuantumCircuit, m: int, shots: int) -> NDArray[np.float64]:
    assert ansatz.num_qubits == n

    rows = np.zeros((2 ** n, m))

    for i in range(m):
        qc = QuantumCircuit(ansatz.num_qubits, ansatz.num_qubits)
        qc.initialize(i)
        qc.compose(ansatz, range(n), inplace=True)
        qc.measure(range(n), range(n))

        counts = get_counts(qc, shots)
        vec = counts_to_vec(2 ** n, shots, counts)
        rows[:, i] = np.flip(vec)   # Not sure why we need to flip

    return rows


def cos_sim(m_true: NDArray[np.float64], m_test: NDArray[np.float64]):
    assert m_true.shape == m_test.shape
    return np.mean(np.diag(np.abs(m_true).T @ m_test))


def train(
    n: int,
    ansatz_layers: int,
    const_params: NDArray[np.float64],
    params: NDArray[np.float64],
    A: NDArray[np.float64],
    m: int,
    E: NDArray[np.float64],
    shots: int,
    lr: float,
    momentum: float,
    iters: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tensor = torch.tensor(params, requires_grad=False)
    optimizer = torch.optim.SGD([tensor], lr=lr, momentum=momentum)

    true_eigenvectors = np.linalg.eig(A @ A.T).eigenvectors[:, :m]

    for i in range(iters):
        print(f"Iteration {i}")

        # Zero out gradients
        optimizer.zero_grad()

        all_params = np.concat((const_params, tensor.numpy()))

        full_grad = torch.tensor([
            grad(n, ansatz_layers, all_params, A,
                 m, E, shots, i + len(const_params))
            for i in range(len(params))
        ])

        print(f"grad magnitude: {full_grad.norm()}")

        eigenvectors = read_ansatz(
            n, ansatz(n, ansatz_layers, all_params), m, shots)

        sim = cos_sim(true_eigenvectors, eigenvectors)
        print(f"cosine similarity: {sim}")

        tensor.grad = full_grad

        # Update the weights
        optimizer.step()

        print(tensor.numpy())

    all_params = np.concat((const_params, tensor.numpy()))

    return (
        all_params,
        read_ansatz(n, ansatz(n, ansatz_layers, all_params), m, shots),
    )


################################################################################
# Testing
################################################################################


def test(
    n: int,
    ansatz_layers: int,
    A: NDArray[np.float64],
    shots: int,
    iters: int,
    lr: float,
    momentum: float = 0.9,
    params: NDArray[np.float64] | None = None
):
    if params is None:
        # Initialize parameters to cover a wide range of rotations
        params = np.linspace(
            0,
            4 * np.pi * n * ansatz_layers,
            4 * n * ansatz_layers,
            endpoint=False,
            dtype=np.float64,
        )

    m = 2 ** n
    E = np.arange(m, dtype=np.float64) + 1

    _, eigenvectors = train(
        n, ansatz_layers,
        np.array([], dtype=np.float64), params,
        A, m, E, shots, lr, momentum, iters,
    )

    true_eigenvectors = np.linalg.eig(A @ A.T).eigenvectors

    print("Eigenvectors:")
    print(eigenvectors)

    print("True eigenvectors:")
    print(true_eigenvectors)

    sim = cos_sim(true_eigenvectors, eigenvectors)
    print(f"Final cosine similarity: {sim}")


def train_incremental(
    n: int,
    A: NDArray[np.float64],
    shots: int,
    layers_iters_lr: list[tuple[int, int, float]],
    momentum: float = 0.9,
):
    m = 2 ** n
    E = np.arange(m, dtype=np.float64) + 1

    const_params = np.array([], dtype=np.float64)
    layers_sum = 0

    for layers, iters, lr in layers_iters_lr:
        layers_sum += layers

        params = np.linspace(
            0,
            4 * np.pi * n * layers,
            4 * n * layers,
            endpoint=False,
            dtype=np.float64,
        )

        const_params, eigenvectors = train(
            n, layers_sum,
            const_params, params,
            A, m, E, shots, lr, momentum, iters,
        )

        true_eigenvectors = np.linalg.eig(A @ A.T).eigenvectors

        print("Eigenvectors:")
        print(eigenvectors)

        print("True eigenvectors:")
        print(true_eigenvectors)

        sim = cos_sim(true_eigenvectors, eigenvectors)
        print(f"Final cosine similarity: {sim}")


def test2():
    n = 2
    ansatz_layers = 2

    A = np.array([
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4],
    ], dtype=np.float64)

    test(n, ansatz_layers, A, 10000, 100, lr=0.01)


# n = 3
def test3():
    n = 3
    ansatz_layers = 6

    A = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 0, 0, 0, 0, 0],
        [0, 0, 3, 1, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 0],
        [0, 0, 0, 0, 0, 0, 0, 8],
    ], dtype=np.float64)

    # Params after 100 iters
    # grad magnitude: 9.52843938192399
    # cosine similarity: 0.9991150297869384
    params = np.array([
        -0.97464396, 3.04049296, 7.06562946, 9.31177225, 12.48045653,
        15.71551651, 17.86453538, 21.90552757, 25.1661304, 28.30081124,
        31.42073696, 34.6314746, 38.43611086, 39.72212573, 44.14067598,
        47.30732977, 50.52806169, 53.50729799, 56.0010889, 59.65488192,
        62.76881423, 66.02134083, 69.04316737, 72.36579187, 75.51845176,
        77.72038359, 81.47505567, 84.65373763, 87.71107811, 91.13712673,
        95.84823835, 95.86768176, 100.53193717, 103.69754392, 106.80394134,
        110.01717396, 112.93501166, 117.92655424, 119.54591566, 122.92107603,
        126.14526743, 127.19264062, 131.91192816, 135.15007874, 138.24461136,
        141.25723207, 144.49947149, 147.53766108, 151.01813822, 153.86706298,
        158.03395371, 160.56071514, 163.70632551, 166.4692982, 171.29320431,
        172.75615002, 175.97927871, 179.12150944, 182.20571002, 185.39802064,
        189.37472683, 193.25556547, 193.95221963, 198.01841855, 201.16466421,
        204.22480902, 207.34530437, 210.51794775, 213.66710177, 216.72079534,
        220.05703974, 223.0358057,
    ], dtype=np.float64)

    # test(n, ansatz_layers, A, 10000, 150, lr=0.001)

    test(n, ansatz_layers, A, 10000, 0, lr=0.001, params=params)

    # test_layer_by_layer(n, A, 10000, [(4, 100, 0.001), (2, 100, 0.001)])


def test4():
    n = 4
    ansatz_layers = 16

    A = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 9, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16],
    ], dtype=np.float64)

    # Params after 200 iters with lr = 0.0001
    # grad magnitude: 567.7006648075786
    # cosine similarity: 0.9732283516181264
    params = np.array([
        -0.85922591, 3.90767643, 6.09194422, 9.61549642, 11.13343381,
        12.59152669, 17.78360708, 18.37357903, 25.4450314, 26.78126208,
        31.8073893, 35.16867688, 34.43034026, 40.683621, 46.80282729,
        46.53591493, 49.65079576, 53.60908147, 56.16531392, 59.41403147,
        63.48864523, 68.66375587, 68.86832114, 68.36053932, 75.08273471,
        78.03015541, 81.21904746, 83.5123189, 84.28650112, 90.71259415,
        91.14703467, 96.77744981, 100.05355178, 103.11961255, 106.66442596,
        110.63052141, 111.63207048, 112.69444032, 121.1118971, 119.69396252,
        125.97390235, 130.71019992, 131.96998722, 134.59623447, 135.10065069,
        141.35503873, 147.99467595, 146.78752837, 150.42341059, 154.03485904,
        157.25665611, 159.86787081, 162.88268125, 169.94656498, 170.16313534,
        168.70498497, 175.6145596, 179.60697912, 182.29762753, 185.22246137,
        184.31474607, 191.91147584, 190.67308229, 197.5745408, 200.4654356,
        204.00888507, 208.0324516, 210.26872333, 213.4052902, 212.17175009,
        221.15905325, 220.79399164, 226.18904769, 230.27863251, 232.42119504,
        236.22476985, 236.0728947, 243.1365156, 247.43883629, 248.1440624,
        251.66035087, 254.02457362, 257.02340591, 260.58237271, 264.46287383,
        269.01309384, 270.37054704, 269.86620244, 276.04378272, 279.97056747,
        282.19260351, 285.47775742, 285.80544261, 291.39283063, 290.73145386,
        298.5058601, 302.10185136, 304.35649527, 307.28177675, 310.87403837,
        314.11844712, 312.84008408, 318.77940345, 320.80098159, 326.51134102,
        327.8742558, 332.6976425, 336.0306517, 336.71902744, 341.98740921,
        347.39030153, 349.14604018, 351.72988364, 354.56018382, 358.79569891,
        361.97803262, 363.77865428, 369.16479592, 370.67749373, 372.33911734,
        377.81404149, 380.06657716, 383.27007142, 385.52045741, 388.43018314,
        393.59378607, 390.97707446, 399.05367849, 402.18991015, 405.15956998,
        408.09736145, 411.3390222, 413.59922086, 412.78557732, 421.3538357,
        421.02218727, 426.87541449, 430.56247771, 432.79607676, 437.19038124,
        436.29865568, 442.45941715, 447.67329793, 449.93330511, 452.72906534,
        455.22586371, 458.24676422, 462.39796625, 465.52939742, 469.79262681,
        471.78918422, 473.60086258, 478.55036908, 481.41108094, 483.9871598,
        486.04879588, 489.072407, 492.52361651, 491.2963397, 498.55683825,
        502.10962536, 506.20884173, 508.71515359, 513.35439567, 514.22573072,
        512.87979597, 523.18839667, 521.43358565, 528.81859711, 532.5467531,
        534.62849413, 538.12214899, 536.67376723, 543.05946166, 548.79231369,
        548.70585578, 551.79031305, 556.54260284, 558.40008814, 562.558765,
        566.91083398, 570.60513224, 571.12075716, 573.17176382, 578.35079964,
        580.67972525, 584.97426811, 587.98284784, 588.7117478, 592.96590668,
        592.9702602, 599.68941267, 602.67596199, 606.91034579, 609.09722407,
        612.77977448, 616.32537712, 615.10277128, 621.06022616, 623.42214624,
        628.20662398, 630.49278438, 634.13516747, 638.4450013, 639.17431759,
        643.14432911, 650.17267047, 650.96824665, 653.91342937, 655.94587526,
        659.09354144, 661.76210776, 666.12705573, 672.44143665, 672.50703761,
        671.24510388, 677.25698379, 682.49634044, 684.46270416, 687.59517118,
        687.13154465, 694.04836578, 694.04691746, 699.91502107, 702.69212649,
        706.77106173, 710.82086303, 714.01968805, 715.57955298, 716.18364171,
        722.0611768, 724.24370967, 729.97872269, 731.71200659, 735.4171417,
        737.82908493, 739.46780822, 745.24337656, 750.27590326, 750.25480465,
        753.47187573, 757.54570737, 760.16862113, 764.16880495, 766.39301625,
        771.7020565, 774.10880039, 773.91843802, 779.96457349, 783.50239893,
        785.44469905, 788.30555418, 790.2421297, 795.13261447, 795.25331192,
        801.10372909
    ], dtype=np.float64)

    # test(n, ansatz_layers, A, 10000, 300, lr=0.0001)

    test(n, ansatz_layers, A, 10000, 0, lr=0.0001, params=params)

    # test_layer_by_layer(n, A, 10000, [
    #     (4, 200, 0.0001),
    #     (4, 200, 0.0001),
    #     (4, 200, 0.00001),
    #     (4, 200, 0.00001),
    # ])


np.set_printoptions(linewidth=10000)

match sys.argv[1]:
    case "2":
        test2()
    case "3":
        test3()
    case "4":
        test4()
