# готовим данные для Pi и для калибровки full-integer квантизации. запускать первым, на пк.
#   ../shared/testset.npz весь тест MNIST (10000), нужен на Pi для accuracy и latency
#   ../shared/calib.npz 200 объектов из обучающей выборки для калибровки int8 в LiteRT
# по тесту калибровать нельзя, это была бы утечка тестовых данных
import numpy as np
from common import load_data, TESTSET, CALIB, SEED

(x_train, y_train), (x_test, y_test) = load_data()

np.savez_compressed(TESTSET, x=x_test, y=y_test)
print(f"тест сохранён, {x_test.shape[0]} примеров в {TESTSET}")

rng = np.random.RandomState(SEED)
idx = rng.choice(len(x_train), size=200, replace=False)
np.savez_compressed(CALIB, x=x_train[idx])
print(f"калибровочный набор из train, {len(idx)} примеров в {CALIB}")
