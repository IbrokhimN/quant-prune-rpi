# обучаем эталонную LeNet на MNIST, запускать на пк (можно на cuda).
# на выходе ../shared/baseline.tt с float32 весами, они сохранены как numpy и грузятся на Pi без cupy
#   python 01_train_baseline.py [--epochs 15]
import argparse
import numpy as np
import tinytensor as tt
from tinytensor.models import LeNet
from tinytensor.nn import CrossEntropyLoss
from tinytensor.optim import AdamW
from common import USE_CUDA, load_data, BASELINE, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--patience", type=int, default=4)
args = ap.parse_args()

print(f"обучение на: {'cuda' if USE_CUDA else 'cpu'}")
tt.set_device("cuda" if USE_CUDA else "cpu")
tt.set_seed(SEED)

(x_train, y_train), (x_test, y_test) = load_data()

# валидацию для early stopping берём из обучающей выборки, а не из теста, чтобы тест не влиял на выбор эпохи
x_val, y_val = x_train[-5000:], y_train[-5000:]
x_fit, y_fit = x_train[:-5000], y_train[:-5000]

model = LeNet(num_classes=10, in_channels=1)
# compile тут безопасен, между ним и fit тензоры весов не подменяются
model.compile(lambda p: AdamW(p, lr=1e-3), CrossEntropyLoss())
model.fit(x_fit, y_fit, epochs=args.epochs, batch_size=64,
          validation_data=(x_val, y_val), patience=args.patience)

acc = (model.predict(x_test) == y_test).mean()
print(f"\nэталон обучен, test accuracy {acc * 100:.2f}%")

model.to("cpu")
tt.set_device("cpu")
model.save(BASELINE)
print(f"веса сохранены в {BASELINE}")
