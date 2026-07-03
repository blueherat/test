import argparse
import io
import os
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple

import numpy as np
import requests
import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

INCEPTION_V3_URL = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb"
INCEPTION_V3_PATH = "classify_image_graph_def.pb"

FID_POOL_NAME = "pool_3:0"
FID_SPATIAL_NAME = "mixed_6/conv:0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", help="参考批次 npz 文件路径")
    parser.add_argument("--sample_batch", help="样本批次 npz 文件路径")
    args = parser.parse_args()

    config = tf.ConfigProto(
        allow_soft_placement=True  # 允许 DecodeJpeg 在 Inception 图中的 CPU 上运行
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    print("正在预热 TensorFlow...")
    # 这会让 TF 现在就打印冗长信息，而不是在下一次 print() 后再打印，以免造成混淆。
    evaluator.warmup()

    print("正在计算参考批次激活...")
    ref_acts = evaluator.read_activations(args.ref_batch)
    print("正在计算/读取参考批次统计量...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

    print("正在计算样本批次激活...")
    sample_acts = evaluator.read_activations(args.sample_batch)
    print("正在计算/读取样本批次统计量...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(args.sample_batch, sample_acts)

    print("正在计算评估指标...")
    print("Inception Score:", evaluator.compute_inception_score(sample_acts[0]))
    print("FID:", sample_stats.frechet_distance(ref_stats))
    print("sFID:", sample_stats_spatial.frechet_distance(ref_stats_spatial))
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
    print("Precision:", prec)
    print("Recall:", recall)


class InvalidFIDException(Exception):
    pass


class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        计算两组统计量之间的 Frechet 距离。
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"训练集与测试集均值向量长度不同：{mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"训练集与测试集协方差矩阵维度不同：{sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # 矩阵乘积可能近似奇异
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "FID 计算产生奇异乘积；向协方差估计的对角线添加 %s"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # 数值误差可能产生很小的虚部
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("虚部 {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class Evaluator:
    def __init__(
        self,
        session,
        batch_size=64,
        softmax_batch_size=512,
    ):
        self.sess = session
        self.batch_size = batch_size
        self.softmax_batch_size = softmax_batch_size
        self.manifold_estimator = ManifoldEstimator(session)
        with self.sess.graph.as_default():
            self.image_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.softmax_input = tf.placeholder(tf.float32, shape=[None, 2048])
            self.pool_features, self.spatial_features = _create_feature_graph(self.image_input)
            self.softmax = _create_softmax_graph(self.softmax_input)

    def warmup(self):
        self.compute_activations(np.zeros([1, 8, 64, 64, 3]))

    def read_activations(self, npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open_npz_array(npz_path, "arr_0") as reader:
            return self.compute_activations(reader.read_batches(self.batch_size))

    def compute_activations(self, batches: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算下游评估使用的图像特征。

        :param batches: 遍历 [0, 255] 范围内 NHWC numpy 数组的迭代器。
        :return: 形状为 [N x X] 的 numpy 数组元组，其中 X 是特征维度。
                 元组内容为 (pool_3, spatial)。
        """
        preds = []
        spatial_preds = []
        for batch in tqdm(batches):
            batch = batch.astype(np.float32)
            pred, spatial_pred = self.sess.run(
                [self.pool_features, self.spatial_features], {self.image_input: batch}
            )
            preds.append(pred.reshape([pred.shape[0], -1]))
            spatial_preds.append(spatial_pred.reshape([spatial_pred.shape[0], -1]))
        return (
            np.concatenate(preds, axis=0),
            np.concatenate(spatial_preds, axis=0),
        )

    def read_statistics(
        self, npz_path: str, activations: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[FIDStatistics, FIDStatistics]:
        obj = np.load(npz_path)
        if "mu" in list(obj.keys()):
            return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
                obj["mu_s"], obj["sigma_s"]
            )
        return tuple(self.compute_statistics(x) for x in activations)

    def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)

    def compute_inception_score(self, activations: np.ndarray, split_size: int = 5000) -> float:
        softmax_out = []
        for i in range(0, len(activations), self.softmax_batch_size):
            acts = activations[i : i + self.softmax_batch_size]
            softmax_out.append(self.sess.run(self.softmax, feed_dict={self.softmax_input: acts}))
        preds = np.concatenate(softmax_out, axis=0)
        # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L46
        scores = []
        for i in range(0, len(preds), split_size):
            part = preds[i : i + split_size]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return float(np.mean(scores))

    def compute_prec_recall(
        self, activations_ref: np.ndarray, activations_sample: np.ndarray
    ) -> Tuple[float, float]:
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
        pr = self.manifold_estimator.evaluate_pr(
            activations_ref, radii_1, activations_sample, radii_2
        )
        return (float(pr[0][0]), float(pr[1][0]))


class ManifoldEstimator:
    """
    用于比较特征向量流形的辅助类。

    改编自 https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        session,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        估计给定特征向量的流形。

        :param session: TensorFlow session。
        :param row_batch_size: 计算成对距离时的行批大小
                               （用于在内存占用和性能之间权衡）。
        :param col_batch_size: 计算成对距离时的列批大小。
        :param nhood_sizes: 估计流形时使用的近邻数量。
        :param clamp_to_percentile: 剪除半径大于给定百分位数的超球。
        :param eps: 用于数值稳定的小数。
        """
        self.distance_block = DistanceBlock(session)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def warmup(self):
        feats, radii = (
            np.zeros([1, 2048], dtype=np.float32),
            np.zeros([1, 1], dtype=np.float32),
        )
        self.evaluate_pr(feats, radii, feats, radii)

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)

        # 通过计算每个样本到 k-NN 的距离来估计特征流形。
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # 计算批次之间的距离。
                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(row_batch, col_batch)

            # 从当前批次中寻找 k 近邻。
            radii[begin1:end1, :] = np.concatenate(
                [
                    x[:, self.nhood_sizes]
                    for x in _numpy_partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)
                ],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate(self, features: np.ndarray, radii: np.ndarray, eval_features: np.ndarray):
        """
        评估新的特征向量是否位于该流形上。
        """
        num_eval_images = eval_features.shape[0]
        num_ref_images = radii.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = features[begin2:end2]

                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(feature_batch, ref_batch)

            # 对新的特征向量小批次，判断它们是否位于估计出的流形中。
            # 如果某个特征向量位于某个参考样本的超球内，则该新样本位于估计流形上。
            # 超球半径由近邻大小 k 对应的距离决定。
            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= radii
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                radii[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0 : end1 - begin1, :], axis=1)

        return {
            "fraction": float(np.mean(batch_predictions)),
            "batch_predictions": batch_predictions,
            "max_realisim_score": max_realism_score,
            "nearest_indices": nearest_indices,
        }

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        高效评估 precision 和 recall。

        :param features_1: 参考批次的 [N1 x D] 特征向量。
        :param radii_1: 参考向量的 [N1 x K1] 半径。
        :param features_2: 另一批次的 [N2 x D] 特征向量。
        :param radii_2: 另一批向量的 [N x K2] 半径。
        :return: (precision, recall) 数组元组：
                 - precision: 长度为 K1 的 np.ndarray
                 - recall: 长度为 K2 的 np.ndarray
        """
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )


class DistanceBlock:
    """
    计算向量之间的成对距离。

    改编自 https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L34
    """

    def __init__(self, session):
        self.session = session

        # 初始化用于计算成对距离的 TF 图。
        with session.graph.as_default():
            self._features_batch1 = tf.placeholder(tf.float32, shape=[None, None])
            self._features_batch2 = tf.placeholder(tf.float32, shape=[None, None])
            distance_block_16 = _batch_pairwise_distances(
                tf.cast(self._features_batch1, tf.float16),
                tf.cast(self._features_batch2, tf.float16),
            )
            self.distance_block = tf.cond(
                tf.reduce_all(tf.math.is_finite(distance_block_16)),
                lambda: tf.cast(distance_block_16, tf.float32),
                lambda: _batch_pairwise_distances(self._features_batch1, self._features_batch2),
            )

            # less_thans 的额外逻辑。
            self._radii1 = tf.placeholder(tf.float32, shape=[None, None])
            self._radii2 = tf.placeholder(tf.float32, shape=[None, None])
            dist32 = tf.cast(self.distance_block, tf.float32)[..., None]
            self._batch_1_in = tf.math.reduce_any(dist32 <= self._radii2, axis=1)
            self._batch_2_in = tf.math.reduce_any(dist32 <= self._radii1[:, None], axis=0)

    def pairwise_distances(self, U, V):
        """
        评估两批特征向量之间的成对距离。
        """
        return self.session.run(
            self.distance_block,
            feed_dict={self._features_batch1: U, self._features_batch2: V},
        )

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        return self.session.run(
            [self._batch_1_in, self._batch_2_in],
            feed_dict={
                self._features_batch1: batch_1,
                self._features_batch2: batch_2,
                self._radii1: radii_1,
                self._radii2: radii_2,
            },
        )


def _batch_pairwise_distances(U, V):
    """
    计算两批特征向量之间的成对距离。
    """
    with tf.variable_scope("pairwise_dist_block"):
        # U 和 V 中每一行的平方范数。
        norm_u = tf.reduce_sum(tf.square(U), 1)
        norm_v = tf.reduce_sum(tf.square(V), 1)

        # 将 norm_u 作为列向量，将 norm_v 作为行向量。
        norm_u = tf.reshape(norm_u, [-1, 1])
        norm_v = tf.reshape(norm_v, [1, -1])

        # 成对平方欧氏距离。
        D = tf.maximum(norm_u - 2 * tf.matmul(U, V, False, True) + norm_v, 0.0)

    return D


class NpzArrayReader(ABC):
    @abstractmethod
    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def remaining(self) -> int:
        pass

    def read_batches(self, batch_size: int) -> Iterable[np.ndarray]:
        def gen_fn():
            while True:
                batch = self.read_batch(batch_size)
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // batch_size + int(rem % batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


class BatchIterator:
    def __init__(self, gen_fn, length):
        self.gen_fn = gen_fn
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen_fn()


class StreamingNpzArrayReader(NpzArrayReader):
    def __init__(self, arr_f, shape, dtype):
        self.arr_f = arr_f
        self.shape = shape
        self.dtype = dtype
        self.idx = 0

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.shape[0]:
            return None

        bs = min(batch_size, self.shape[0] - self.idx)
        self.idx += bs

        if self.dtype.itemsize == 0:
            return np.ndarray([bs, *self.shape[1:]], dtype=self.dtype)

        read_count = bs * np.prod(self.shape[1:])
        read_size = int(read_count * self.dtype.itemsize)
        data = _read_bytes(self.arr_f, read_size, "array data")
        return np.frombuffer(data, dtype=self.dtype).reshape([bs, *self.shape[1:]])

    def remaining(self) -> int:
        return max(0, self.shape[0] - self.idx)


class MemoryNpzArrayReader(NpzArrayReader):
    def __init__(self, arr):
        self.arr = arr
        self.idx = 0

    @classmethod
    def load(cls, path: str, arr_name: str):
        with open(path, "rb") as f:
            arr = np.load(f)[arr_name]
        return cls(arr)

    def read_batch(self, batch_size: int) -> Optional[np.ndarray]:
        if self.idx >= self.arr.shape[0]:
            return None

        res = self.arr[self.idx : self.idx + batch_size]
        self.idx += batch_size
        return res

    def remaining(self) -> int:
        return max(0, self.arr.shape[0] - self.idx)


@contextmanager
def open_npz_array(path: str, arr_name: str) -> NpzArrayReader:
    with _open_npy_file(path, arr_name) as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
            header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
            header = np.lib.format.read_array_header_2_0(arr_f)
        else:
            yield MemoryNpzArrayReader.load(path, arr_name)
            return
        shape, fortran, dtype = header
        if fortran or dtype.hasobject:
            yield MemoryNpzArrayReader.load(path, arr_name)
        else:
            yield StreamingNpzArrayReader(arr_f, shape, dtype)


def _read_bytes(fp, size, error_template="ran out of data"):
    """
    复制自：https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/format.py#L788-L886

    从类文件对象读取，直到读满 size 字节。
    如果在读满 size 字节之前没有遇到 EOF，则抛出 ValueError。
    非阻塞对象只有派生自 io 对象时才受支持。
    这是必需的，例如 Python 2.6 中 ZipExtFile 可能返回少于请求数量的数据。
    """
    data = bytes()
    while True:
        # io 文件（python3 默认）会在 would-block 时返回 None 或抛错；
        # python2 文件会截断，对此可能无能为力。注意，普通文件不能是非阻塞的。
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF：读取 %s 时，期望 %d 字节，实际得到 %d 字节"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


@contextmanager
def _open_npy_file(path: str, arr_name: str):
    with open(path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_f:
            if f"{arr_name}.npy" not in zip_f.namelist():
                raise ValueError(f"npz 文件中缺少 {arr_name}")
            with zip_f.open(f"{arr_name}.npy", "r") as arr_f:
                yield arr_f


def _download_inception_model():
    if os.path.exists(INCEPTION_V3_PATH):
        return
    print("正在下载 InceptionV3 模型...")
    with requests.get(INCEPTION_V3_URL, stream=True) as r:
        r.raise_for_status()
        tmp_path = INCEPTION_V3_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        os.rename(tmp_path, INCEPTION_V3_PATH)


def _create_feature_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    pool3, spatial = tf.import_graph_def(
        graph_def,
        input_map={f"ExpandDims:0": input_batch},
        return_elements=[FID_POOL_NAME, FID_SPATIAL_NAME],
        name=prefix,
    )
    _update_shapes(pool3)
    spatial = spatial[..., :7]
    return pool3, spatial


def _create_softmax_graph(input_batch):
    _download_inception_model()
    prefix = f"{random.randrange(2**32)}_{random.randrange(2**32)}"
    with open(INCEPTION_V3_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    (matmul,) = tf.import_graph_def(
        graph_def, return_elements=[f"softmax/logits/MatMul"], name=prefix
    )
    w = matmul.inputs[1]
    logits = tf.matmul(input_batch, w)
    return tf.nn.softmax(logits)


def _update_shapes(pool3):
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L50-L63
    ops = pool3.graph.get_operations()
    for op in ops:
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:  # pylint: disable=protected-access
                # shape = [s.value for s in shape] TF 1.x
                shape = [s for s in shape]  # TF 2.x
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__["_shape_val"] = tf.TensorShape(new_shape)
    return pool3


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


if __name__ == "__main__":
    main()
