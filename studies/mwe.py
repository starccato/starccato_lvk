from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import os
import numpy as np
import matplotlib.pyplot as plt
from starccato_jax.waveforms import  StarccatoCCSNe


FS, N, T = 4096.0, 512, 512 / 4096.0
DT = 1.0 / FS
DF = 1.0 / T  # Frequency resolution
PSD_N = 4096
TIMES = np.arange(0, T, DT)[:N]
FREQS = np.fft.rfftfreq(N, d=DT)
FLOW = 100.0
FMASK = (FREQS >= FLOW) & (FREQS <= 2048)
N_BURNIN, N_STEPS = 100, 100


data_fn = 'test_data/analysis_chunk_1256676910.hdf5'
psd_fn = 'test_data/psd_1256676910.hdf5'
z_injection = np.zeros(32)
outdir = "testing_datagen"

os.makedirs(outdir, exist_ok=True)
noise_data = TimeSeries.read(data_fn, format='hdf5')
psd_data = FrequencySeries.read(psd_fn, format='hdf5')

DT = 1.0 / 4096
N = 512  # we only want 512 data points
freqs = np.fft.rfftfreq(N, d=DT)
freq_mask = (freqs >= 20) & (freqs <= 1024)  # filter frequencies between 20 Hz and 1024 Hz

noise = noise_data.value
noise_fft = np.fft.rfft(noise)
freq = np.fft.rfftfreq(len(noise), d=DT)
df = freq[1] - freq[0]  # Frequency resolution
mask = (freq >= 20) & (freq <= 1024)
psd = np.interp(
    freq[mask],
    xp=psd_data.frequencies.value,
    fp=psd_data.value,
)
noise_amp = np.median(noise)
signal = StarccatoCCSNe().generate(n=1)[0] * noise_amp # len = 512
# pad the signal to match the noise length
signal = np.pad(signal, (0, len(noise) - len(signal)), mode='constant', constant_values=0)

signal_fft = np.fft.rfft(signal)

data = noise + signal
data_fft = np.fft.rfft(data)



def get_snr(signal_fft, data_fft, psd, df, mask):
    h_d = 4 * np.real(np.sum(np.conj(signal_fft[mask]) * data_fft[mask] / psd) * df)
    h_h = 4 * np.real(np.sum(np.conj(signal_fft[mask]) * signal_fft[mask] / psd) * df)
    return h_d / np.sqrt(h_h)


snr = get_snr(signal_fft, data_fft, psd, DF, mask)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(noise_data.psd(), color='gray', alpha=0.5, label='GWPY simulated noise welch PSD')
ax.plot(psd_data, color='blue', label='GWPY PSD')
ax.loglog(freq[mask], np.abs(noise_fft[mask]) ** 2, color='black', lw=2, label='Noise pdgrm')
ax.loglog(freq[mask], np.abs(signal_fft[mask]) ** 2, color='C1', lw=2, label='signal pdgrm')
ax.loglog(freq[mask], np.abs(data_fft[mask]) ** 2, color='C2', lw=2, label='Data pdgrm')
ax.loglog(freq[mask], psd, color='skyblue', label='Analysis PSD', )
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [1/Hz]")
plt.title(f"SNR = {snr:.2f} (20 Hz - 1024 Hz)")
plt.xlim(20, 1024)
plt.tight_layout()
plt.show()

