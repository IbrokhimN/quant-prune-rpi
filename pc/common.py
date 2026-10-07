# общее для скриптов на пк. пути, устройство, загрузка модели и дообучение.
#
# два момента в finetune(), на которых легко споткнуться (подробности в README).
# optimizer создаём внутри finetune() после прунинга и переноса на устройство. прунинг и
# m.to("cuda") подменяют тензоры весов новыми объектами, и optimizer, созданный раньше,
# продолжает обновлять старые, так что дообучаются одни свёртки.
# при неструктурном прунинге маски применяем после каждого шага, иначе AdamW возвращает
# занулённым весам ненулевые значения и разреженность тает
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, "..", "shared"))
for p in (HERE, SHARED):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                   # noqa: E402
import tinytensor as tt                              # noqa: E402
from tinytensor.core.tensor import get_array_module  # noqa: E402
from tinytensor.data import load_mnist, TensorDataset, DataLoader  # noqa: E402
from tinytensor.models import LeNet                  # noqa: E402
from tinytensor.nn import CrossEntropyLoss           # noqa: E402
from tinytensor.optim import AdamW                   # noqa: E402

MODELS_DIR = os.path.join(SHARED, "models")
BASELINE = os.path.join(SHARED, "baseline.tt")
TESTSET = os.path.join(SHARED, "testset.npz")
CALIB = os.path.join(SHARED, "calib.npz")
SEED = 600613
FINETUNE_EPOCHS = 3


def detect_cuda():
    try:
        import cupy  # noqa
        cupy.zeros(1)
        return True
    except Exception:
        return False


USE_CUDA = detect_cuda()
tt.set_device("cpu")  # по умолчанию всегда cpu


def load_data():
    return load_mnist(os.path.join(HERE, "data_mnist"))


def fresh_model(path=BASELINE):
    # свежий LeNet на cpu с весами из файла, веса сохранены как numpy и грузятся и на Pi
    tt.set_device("cpu")
    m = LeNet(num_classes=10, in_channels=1)
    m.load(path)
    return m


def linear_masks(model):
    # маски ненулевых весов Linear-слоёв, ключ это индекс слоя
    from tinytensor.nn.linear import Linear
    return {i: (l.weight.data != 0) for i, l in enumerate(model.net.layers)
            if isinstance(l, Linear)}


def finetune(m, x_train, y_train, epochs=FINETUNE_EPOCHS, seed=SEED, masks=None,
             lr=1e-3, batch_size=64, log=print):
    # дообучение у всех вариантов одинаковое, иначе сравнение нечестное. AdamW с lr=1e-3 и один
    # и тот же seed, значит и порядок батчей тот же. masks нужны для неструктурного прунинга,
    # после каждого шага обнуляем запрещённые веса
    dev = "cuda" if USE_CUDA else "cpu"
    tt.set_device(dev)  # DataLoader кладёт батчи на устройство по умолчанию
    try:
        m.to(dev)
        tt.set_seed(seed)
        opt = AdamW(m.parameters(), lr=lr)  # после прунинга и .to(dev), не раньше
        loss_fn = CrossEntropyLoss()
        dev_masks = None
        if masks:
            dev_masks = {}
            for i, mk in masks.items():
                w = m.net.layers[i].weight
                xp = get_array_module(w.data)
                dev_masks[i] = xp.asarray(mk.astype(np.float32))
        m.train()
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        for ep in range(epochs):
            tot, nb = 0.0, 0
            for xb, yb in loader:
                loss = loss_fn(m(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if dev_masks:
                    for i, mk in dev_masks.items():
                        m.net.layers[i].weight.data *= mk
                tot += float(loss.data)
                nb += 1
            log(f"    дообучение: эпоха {ep + 1}/{epochs}  loss={tot / max(nb, 1):.4f}")
    finally:
        m.to("cpu")
        tt.set_device("cpu")
        m.eval()
    return m


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
