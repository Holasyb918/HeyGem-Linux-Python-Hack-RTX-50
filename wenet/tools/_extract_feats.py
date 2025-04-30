import librosa
# import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy import signal

import torchaudio.compliance.kaldi as kaldi
import torchaudio
# torchaudio.set_audio_backend("sox_io")


def _extract_feature(wav_path):
    """ Extract acoustic fbank feature from origin waveform.

    Speed perturbation and wave amplitude distortion is optional.

    Args:
        batch: a list of tuple (wav id , wave path).
        speed_perturb: bool, whether or not to use speed pertubation.
        wav_distortion_conf: a dict , the config of wave amplitude distortion.
        feature_extraction_conf:a dict , the config of fbank extraction.

    Returns:
        (keys, feats, labels)
    """
    waveform, sample_rate = torchaudio.load_wav(wav_path)

    mat = kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.1,
                energy_floor=0.0,
                sample_frequency=sample_rate)
    mat = mat.detach().numpy()

    return mat

def _extract_feature_norm(wav_path):
    """ Extract acoustic fbank feature from origin waveform.

    Speed perturbation and wave amplitude distortion is optional.

    Args:
        batch: a list of tuple (wav id , wave path).
        speed_perturb: bool, whether or not to use speed pertubation.
        wav_distortion_conf: a dict , the config of wave amplitude distortion.
        feature_extraction_conf:a dict , the config of fbank extraction.

    Returns:
        (keys, feats, labels)
    """

    waveform, sample_rate = torchaudio.load_wav(wav_path)

    mat = kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.1,
                energy_floor=0.0,
                sample_frequency=sample_rate)
    mat = mat.detach().numpy()

    return mat


hparams = {
    'sample_rate': 16000,#一秒16000个采样点
    'preemphasis': 0.97,
    'n_fft': 1024,
    'hop_length': 200,#80个采样点为帧移动步长 5ms
    'win_length': 800,#400个采样点为帧宽度，25ms
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 0.,
    'fmax': 8000.,
    'ref_db': 20,  #
    'min_db': -80.0,  # restrict the dynamic range of log power
    'iterations': 100,  # griffin_lim #iterations
    'silence_db': -28.0,
    'center': True,#是否将MFCC作为当前帧中间向量的结果。（数个向量作为一帧生成一个mfcc)
}

_mel_basis = None


def load_wav(wav_f, sr=None):
    # wav_arr, _ = librosa.load(wav_f, sr=sr)
    # return wav_arr
    if type(wav_f)==str:
        wav_arr, _ = librosa.load(wav_f, sr=sr)
    else:
        wav_arr = wav_f
    return wav_arr

def write_wav(write_path, wav_arr, sr):
    wav_arr *= 32767 / max(0.01, np.max(np.abs(wav_arr)))
    wavfile.write(write_path, sr, wav_arr.astype(np.int16))
    return

def preempahsis(wav_arr, pre_param=hparams['preemphasis']):
    return signal.lfilter([1, -pre_param], [1], wav_arr)

def deemphasis(wav_arr, pre_param=hparams['preemphasis']):
    return signal.lfilter([1], [1, -pre_param], wav_arr)

def split_wav(wav_arr, top_db=-hparams['silence_db']):
    intervals = librosa.effects.split(wav_arr, top_db=top_db)
    return intervals

def mulaw_encode(wav_arr, quantization_channels):
    mu = float(quantization_channels - 1)
    safe_wav_abs = np.minimum(np.abs(wav_arr), 1.0)
    encoded = np.sign(wav_arr) * np.log1p(mu * safe_wav_abs) / np.log1p(mu)
    return encoded

def mulaw_encode_quantize(wav_arr, quantization_channels):
    mu = float(quantization_channels - 1)
    safe_wav_abs = np.minimum(np.abs(wav_arr), 1.0)
    encoded = np.sign(wav_arr) * np.log1p(mu * safe_wav_abs) / np.log1p(mu)
    return ((encoded + 1.) / 2 * mu + 0.5).astype(np.int32)

def mulaw_decode(encoded, quantization_channels):
    mu = float(quantization_channels - 1)
    magnitude = (1 / mu) * ((1 + mu) ** abs(encoded) - 1.)
    return np.sign(encoded) * magnitude

def mulaw_decode_quantize(encoded, quantization_channels):
    mu = float(quantization_channels - 1)
    signal = 2 * (encoded.astype(np.float32) / mu) - 1.
    magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1.)
    return np.sign(signal) * magnitude

def mulaw_encode_quantize_tf(wav_batch, quantization_channels):
    with tf.variable_scope('mulaw_encode'):
        mu = tf.cast(quantization_channels - 1, tf.float32)
        safe_wav_abs = tf.minimum(tf.abs(wav_batch), 1.0)
        encoded = tf.sign(wav_batch) * tf.log1p(mu * safe_wav_abs) / tf.log1p(mu)
        return tf.cast((encoded + 1.) / 2 * mu + 0.5, tf.int32)

# def mulaw_encode_tf(wav_batch, quantization_channels):
#     with tf.variable_scope('mulaw_encode'):
#         mu = tf.cast(quantization_channels - 1, tf.float32)
#         safe_wav_abs = tf.minimum(tf.abs(wav_batch), 1.0)
#         encoded = tf.sign(wav_batch) * tf.log1p(mu * safe_wav_abs) / tf.log1p(mu)
#         return encoded

# def mulaw_decode_quantize_tf(encoded, quantization_channels):
    with tf.variable_scope('mulaw_decode'):
        mu = tf.cast(quantization_channels - 1, tf.float32)
        signal = 2 * (tf.cast(encoded, tf.float32) / mu) - 1.
        magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1.)
        return tf.sign(signal) * magnitude

# def mulaw_decode_tf(encoded, quantization_channels):
    with tf.variable_scope('mulaw_decode'):
        mu = tf.cast(quantization_channels - 1, tf.float32)
        magnitude = (1 / mu) * ((1 + mu) ** abs(encoded) - 1.)
        return tf.sign(encoded) * magnitude

def stft(wav_arr, n_fft=hparams['n_fft'],#短时傅里叶变化
         hop_len=hparams['hop_length'],
         win_len=hparams['win_length'],
         window=hparams['window'],
         center=hparams['center']):
    # return shape: [n_freqs, time]
    return librosa.core.stft(wav_arr, n_fft=n_fft, hop_length=hop_len,
                             win_length=win_len, window=window, center=center)

# def stft_tf(wav_arr, n_fft=hparams['n_fft'],
#             hop_len=hparams['hop_length'],
#             win_len=hparams['win_length'],
#             window=hparams['window']):
#     window_f = {'hann': tf.contrib.signal.hann_window,
#                 'hamming': tf.contrib.signal.hamming_window}[window]
#     # returned value is of shape [..., frames, fft_bins] and complex64 value
#     return tf.contrib.signal.stft(signals=wav_arr, frame_length=win_len,
#                                   frame_step=hop_len, fft_length=n_fft,
#                                   window_fn=window_f)

def istft(stft_matrix, hop_len=hparams['hop_length'],
          win_len=hparams['win_length'], window=hparams['window']):
    # stft_matrix should be complex stft results instead of magnitude spectrogram
    # or power spectrogram, and of shape [n_freqs, time]
    return librosa.core.istft(stft_matrix, hop_length=hop_len,
                              win_length=win_len, window=window)

# def istft_tf(stft_matrix, hop_len=hparams['hop_length'], n_fft=hparams['n_fft'],
#              win_len=hparams['win_length'], window=hparams['window']):
#     window_f = {'hann': tf.contrib.signal.hann_window,
#                 'hamming': tf.contrib.signal.hamming_window}[window]
#     # stft_matrix should be of shape [..., frames, fft_bins]
#     return tf.contrib.signal.inverse_stft(stft_matrix, frame_length=win_len,
#                                           frame_step=hop_len, fft_length=n_fft,
#                                           window_fn=window_f)

def spectrogram(wav_arr, n_fft=hparams['n_fft'],
                hop_len=hparams['hop_length'],
                win_len=hparams['win_length'],
                window=hparams['window'],
                center=hparams['center']):
    # return shape: [time, n_freqs]
    s = stft(wav_arr, n_fft=n_fft, hop_len=hop_len,
             win_len=win_len, window=window, center=center).T
    magnitude = np.abs(s)       #幅度谱
    power = magnitude ** 2             #能量谱                         #经过短时傅里叶变换得到magnitude(?)和其平方  为什么不是快速傅里叶变化
    return {'magnitude': magnitude,
            'power': power,
            'stft':s.T}

def power_spec2mel(power_spec, sr=hparams['sample_rate'], n_fft=hparams['n_fft'],
                   num_mels=hparams['num_mels'], fmin=hparams['fmin'], fmax=hparams['fmax']):
    # power_spec should be of shape [time, 1+n_fft/2]
    power_spec_t = power_spec.T
    global _mel_basis
    _mel_basis = (librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
                  if _mel_basis is None else _mel_basis)  # [n_mels, 1+n_fft/2]
    mel_spec = np.dot(_mel_basis, power_spec_t)  # [n_mels, time]
    return mel_spec.T   # mel谱

def wav2melspec(wav_arr, sr=hparams['sample_rate'], n_fft=hparams['n_fft'],
                hop_len=hparams['hop_length'], win_len=hparams['win_length'],
                window=hparams['window'], num_mels=hparams['num_mels'],
                fmin=hparams['fmin'], fmax=hparams['fmax']):
    power_spec = spectrogram(wav_arr, n_fft, hop_len, win_len, window)['power']
    melspec = power_spec2mel(power_spec.T, sr, n_fft, num_mels, fmin, fmax)
    return melspec  # [time, num_mels]

def wav2mfcc(wav_arr, sr=hparams['sample_rate'], n_mfcc=hparams['n_mfcc'],
             n_fft=hparams['n_fft'], hop_len=hparams['hop_length'],
             win_len=hparams['win_length'], window=hparams['window'],
             num_mels=hparams['num_mels'], fmin=0.0,
             fmax=None, ref_db=hparams['ref_db']):
    from scipy.fftpack import dct
    print("wav_arr1:",wav_arr.shape)
    wav_arr = preempahsis(wav_arr)
    print("wav_arr2:",wav_arr.shape)

    mag_spec = spectrogram(wav_arr, n_fft=n_fft, hop_len=hop_len,
                           win_len=win_len, window=window)['magnitude']
    mel_spec = power_spec2mel(mag_spec, sr=sr, n_fft=n_fft, num_mels=num_mels,
                              fmin=fmin, fmax=fmax)
    # log_melspec = power2db(mel_spec, ref_db=ref_db)
    log_melspec = librosa.amplitude_to_db(mel_spec)
    mfcc = dct(x=log_melspec.T, axis=0, type=2, norm='ortho')[:n_mfcc]
    # mfcc = np.dot(librosa.filters.dct(n_mfcc, log_melspec.shape[1]), log_melspec.T)
    deltas = librosa.feature.delta(mfcc)
    delta_deltas = librosa.feature.delta(mfcc, order=2)
    mfcc_feature = np.concatenate((mfcc, deltas, delta_deltas), axis=0)

    return mfcc_feature.T

def wav2mfcc_v2(wav_arr, sr=hparams['sample_rate'], n_mfcc=hparams['n_mfcc'],#使用这个
                n_fft=hparams['n_fft'], hop_len=hparams['hop_length'],
                win_len=hparams['win_length'], window=hparams['window'],
                num_mels=hparams['num_mels'], fmin=0.0,
                fmax=None, ref_db=hparams['ref_db'],
                center=hparams['center']):
    from scipy.fftpack import dct
    wav_arr = preempahsis(wav_arr)
    #经过一次滤波
    power_spec = spectrogram(wav_arr, n_fft=n_fft, hop_len=hop_len,
                             win_len=win_len, window=window, center=center)['power']
    mel_spec = power_spec2mel(power_spec, sr=sr, n_fft=n_fft, num_mels=num_mels,
                              fmin=fmin, fmax=fmax)             # mel谱
    log_melspec = power2db(mel_spec, ref_db=ref_db)             #对数mel谱


    """下面是MFCC"""
    # mfcc = dct(x=log_melspec.T, axis=0, type=2, norm='ortho')[:n_mfcc]
    # deltas = librosa.feature.delta(mfcc)
    # delta_deltas = librosa.feature.delta(mfcc, order=2)
    # mfcc_feature = np.concatenate((mfcc, deltas, delta_deltas), axis=0)
    # return mfcc_feature.T
    x_stft = spectrogram(wav_arr, n_fft=n_fft, hop_len=hop_len,
                             win_len=win_len, window=window, center=center)['stft']
    # print("log_melspec:", x_stft.shape)
    return log_melspec,x_stft


def wav2linear_v2(wav_arr, sr=hparams['sample_rate'], n_mfcc=hparams['n_mfcc'],  # 使用这个
                n_fft=hparams['n_fft'], hop_len=hparams['hop_length'],
                win_len=hparams['win_length'], window=hparams['window'],
                num_mels=hparams['num_mels'], fmin=0.0,
                fmax=None, ref_db=hparams['ref_db'],
                center=hparams['center']):
    from scipy.fftpack import dct
    wav_arr = preempahsis(wav_arr)
    # 经过一次滤波
    power_spec = spectrogram(wav_arr, n_fft=n_fft, hop_len=hop_len,
                             win_len=win_len, window=window, center=center)['power']
    linear = _amp_to_db(power_spec, ref_db=ref_db)  # 对数mel谱
    normalized_linear = _db_normalize(linear, min_db=hparams['min_db'])
    x_stft = spectrogram(wav_arr, n_fft=n_fft, hop_len=hop_len,
                             win_len=win_len, window=window, center=center)['stft']


    return normalized_linear,x_stft

def _amp_to_db(x,ref_db=20):
    return 20 * np.log10(np.maximum(1e-5, x)) + ref_db


def mel2log_mel(mel_spec, ref_db=hparams['ref_db'], min_db=hparams['min_db']):
    log_mel = power2db(mel_spec, ref_db)
    normalized = log_power_normalize(log_mel, min_db)
    return normalized

def power2db(power_spec, ref_db=hparams['ref_db'], tol=1e-5):
    # power spectrogram is stft ** 2
    # returned value: (10. * log10(power_spec) - ref_db)
    return 10. * np.log10(power_spec + tol) - ref_db

def db2power(power_db, ref_db=hparams['ref_db']):
    return np.power(10.0, 0.1 * (power_db + ref_db))
#
# def db2power_tf(power_db, ref_db=hparams['ref_db']):
#     return tf.pow(10.0, 0.1 * (power_db + ref_db))

def log_power_normalize(log_power, min_db=hparams['min_db']):
    """
    :param log_power: in db, computed by power2db(spectrogram(wav_arr)['power'])
    :param min_db: minimum value of log_power in db
    :return: log_power normalized to [0., 1.]
    """
    assert min_db < 0. or "min_db should be a negative value like -80.0 or -100.0"
    return np.clip((log_power - min_db) / -min_db, 0., 1.)

def log_power_denormalize(normalized_logpower, min_db=hparams['min_db']):
    return np.clip(normalized_logpower, 0., 1.) * -min_db + min_db

# def log_power_denormalize_tf(normalized_logpower, min_db=hparams['min_db']):
#     return tf.clip_by_value(normalized_logpower, 0., 1.) * -min_db + min_db

def griffin_lim(magnitude_spec, iterations=hparams['iterations']):
    """
    :param magnitude_spec: magnitude spectrogram of shape [time, n_freqs]
                           obtained from spectrogram(wav_arr)['magnitude]
    :param iterations: number of iterations to estimate phase
    :return: waveform array
    """
    mag = magnitude_spec.T  # transpose to [n_freqs, time]
    angles = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    complex_mag = np.abs(mag).astype(np.complex)
    stft_0 = complex_mag * angles
    y = istft(stft_0)
    for i in range(iterations):
        angles = np.exp(1j * np.angle(stft(y)))
        y = istft(complex_mag * angles)
    return y

# def grinffin_lim_tf(magnitude_spec, iterations=hparams['iterations']):
#     # magnitude_spec: [frames, fft_bins], of type tf.float32
#     angles = tf.cast(
#         tf.exp(2j * np.pi * tf.cast(
#             tf.random_uniform(
#                 tf.shape(magnitude_spec)),
#             dtype=tf.complex64)),
#         dtype=tf.complex64)
#     complex_mag = tf.cast(tf.abs(magnitude_spec), tf.complex64)
#     stft_0 = complex_mag * angles
#     y = istft_tf(stft_0)
#     for i in range(iterations):
#         angles = tf.exp(1j * tf.cast(tf.angle(stft_tf(y)), tf.complex64))
#         y = istft_tf(complex_mag * angles)
#     return y

def griffin_lim_test(wav_f, n_fft=hparams['n_fft'],
                     hop_len=hparams['hop_length'],
                     win_len=hparams['win_length'],
                     window=hparams['window']):
    wav_arr = load_wav(wav_f)
    spec_dict = spectrogram(wav_arr, n_fft=n_fft, hop_len=hop_len,
                            win_len=win_len, window=window)
    mag_spec = spec_dict['magnitude']
    y = griffin_lim(mag_spec)
    write_wav('reconstructed1.wav', y, sr=16000)

def stft2wav_test(stft_f, mean_f, std_f):
    spec = np.load(stft_f)
    mean = np.load(mean_f)
    std = np.load(std_f)
    spec = spec * std + mean
    spec = log_power_denormalize(spec)
    power_spec = db2power(spec)
    mag_spec = power_spec ** 0.5
    y = griffin_lim(mag_spec)
    y = deemphasis(y)
    write_wav('reconstructed2.wav', y, sr=16000)
    return y
#
# def stft2wav_tf_test(stft_f, mean_f, std_f):
#     # get inputs
#     spec = np.load(stft_f)
#     mean = np.load(mean_f)
#     std = np.load(std_f)
#     spec = spec * std + mean
#     # build graph
#     spec_pl = tf.placeholder(tf.float32, [None, None, 513])
#     denormalized = log_power_denormalize_tf(spec_pl)
#     mag_spec = tf.pow(db2power_tf(denormalized), 0.5)
#     wav = grinffin_lim_tf(mag_spec)
#     # set session and run
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     wav_arr = sess.run(wav, feed_dict={spec_pl: np.expand_dims(spec, axis=0)})
#     sess.close()
#     y = deemphasis(np.squeeze(wav_arr))
#     write_wav('reconstructed_tf.wav', y, sr=16000)
#     return y

# 超参数个数：1
# return: db normalized to [0., 1.]
def _db_normalize(db, min_db):
    return np.clip((db - min_db) / -min_db, 0., 1.)




def mfcc_test():
    wav_f = './test.wav'
    wav_arr = load_wav(wav_f)


    mfcc = wav2mfcc_v2(wav_arr)
    mfcc1 = np.load('test.npy')
    print(mfcc.min(), mfcc1.min())
    print(mfcc.max(), mfcc1.max())
    print(mfcc.mean(), mfcc1.mean())
    print(np.abs(mfcc - mfcc1))
    print(np.mean(np.abs(mfcc - mfcc1)))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(211)
    plt.imshow(mfcc.T, origin='lower')
    # plt.colorbar()
    plt.subplot(212)
    plt.imshow(mfcc1.T, origin='lower')
    # plt.colorbar()
    plt.tight_layout()
    plt.show()
    return



if __name__ == '__main__':
    mfcc_test()
