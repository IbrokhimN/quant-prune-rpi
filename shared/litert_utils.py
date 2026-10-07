# обёртка над интерпретатором LiteRT. нужна и при экспорте на пк, и в бенчмарке на pi.
# рантайм ищем по очереди, сначала ai_edge_litert, потом старый tflite_runtime, потом полный tensorflow
# (последний только как запасной вариант на пк)
import numpy as np


def get_interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter
        import ai_edge_litert as pkg
        return Interpreter, "ai-edge-litert", getattr(pkg, "__version__", "?")
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        import tflite_runtime as pkg
        return Interpreter, "tflite-runtime", getattr(pkg, "__version__", "?")
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter, "tensorflow", tf.__version__
    except ImportError:
        raise ImportError(
            "не найден ни ai-edge-litert, ни tflite-runtime, ни tensorflow. "
            "На Pi с 64-bit ОС он ставится через pip install ai-edge-litert, про 32-bit смотри README.")


def make_litert_fn(path, batch, threads=1):
    # возвращает fn(xb) для входа (B,1,28,28) float32, как в testset.npz.
    # батч в моделях динамический, так что размер фиксируем через resize_tensor_input.
    # при одном канале NCHW и NHWC лежат в памяти одинаково, поэтому хватает обычного reshape
    Interpreter, _, _ = get_interpreter_class()
    it = Interpreter(model_path=path, num_threads=threads)
    inp = it.get_input_details()[0]
    it.resize_tensor_input(inp["index"], [batch, 28, 28, 1])
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    assert inp["dtype"] == np.float32, "ожидается float32-вход (модели экспортируются с float I/O)"
    i_in, i_out = inp["index"], out["index"]

    def fn(xb):
        it.set_tensor(i_in, xb.reshape(batch, 28, 28, 1))
        it.invoke()
        return np.argmax(it.get_tensor(i_out), axis=1)
    return fn


def runtime_info():
    _, name, ver = get_interpreter_class()
    return {"litert_package": name, "litert_version": ver}
