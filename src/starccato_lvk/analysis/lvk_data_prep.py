import numpy as np
import jax

from typing import Dict
import bilby
from bilby.core.utils import nfft
from bilby.gw.detector import PowerSpectralDensity

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

# Constants
FMAX = 1024.0
FLOW = 100.0
SAMPLING_FREQUENCY = 4096.0

# Rescaling factors to avoid float32 precision issues
STRAIN_SCALE = 1e21  # Scale strain from ~1e-22 to ~1e-1
PSD_SCALE = STRAIN_SCALE ** 2  # PSD scales as strain^2

# Plot styling
DATA_COL = "tab:gray"
SIGNAL_COL = "tab:orange"
PSD_COL = "black"
POSTERIOR_COL = "tab:blue"


class LvkDataPrep:
    """
    Class for preparing LIGO-Virgo-KAGRA data for gravitational wave analysis.
    Handles data loading, PSD estimation, injection, and rescaling for numerical stability.
    """

    def __init__(self, detector='H1', waveform_model=None, injection_params: Dict = None,
                 strain_scale=STRAIN_SCALE, psd_scale=PSD_SCALE, roll_off=0.1):
        """
        Initialize LVK data preparation.

        Parameters:
        -----------
        detector : str
            Detector name (e.g., 'H1', 'L1', 'V1')
        waveform_model : object, optional
            Waveform model for injection (e.g., StarccatoCCSNe instance)
        injection_params : dict, optional
            Parameters for injection {'amplitude': float, 'z': array, 'rng_key': int}
        strain_scale : float
            Rescaling factor for strain data
        psd_scale : float
            Rescaling factor for PSD data
        roll_off : float
            Roll-off parameter for bilby
        """
        self.detector = detector
        self.waveform_model = waveform_model
        self.injection_params = injection_params or {}
        self.strain_scale = strain_scale
        self.psd_scale = psd_scale
        self.roll_off = roll_off

        self.ifo = None
        self.rescaled_data = None
        self.injection_signal = None
        self.window = None
        self.snr_optimal = None
        self.snr_matched = None
        self.lnz_noise = None

    @classmethod
    def load(cls, data_path, psd_path, **kwargs):
        """
        Load data from HDF5 files.

        Parameters:
        -----------
        data_path : str
            Path to strain data HDF5 file
        psd_path : str
            Path to PSD HDF5 file
        **kwargs : dict
            Additional arguments for LvkDataPrep constructor
        """
        instance = cls(**kwargs)

        # Load data
        data = TimeSeries.read(data_path, format="hdf5")
        psd_data = FrequencySeries.read(psd_path, format="hdf5")

        instance._setup_interferometer(data, psd_data)
        return instance

    @classmethod
    def download(cls, start_time, end_time, **kwargs):
        """
        Download data from GWOSC.

        Parameters:
        -----------
        start_time : int
            GPS start time
        end_time : int
            GPS end time
        **kwargs : dict
            Additional arguments for LvkDataPrep constructor
        """
        instance = cls(**kwargs)

        # Download strain data
        data = TimeSeries.fetch_open_data(instance.detector, start_time, end_time)

        # Download data for PSD estimation (32 seconds before)
        psd_data = TimeSeries.fetch_open_data(instance.detector, start_time - 32, start_time)

        # Estimate PSD from the pre-data
        psd = psd_data.psd(fftlength=4, method='median')

        instance._setup_interferometer(data, psd)
        return instance

    def _setup_interferometer(self, data, psd_data):
        """Set up bilby interferometer with data and optional injection."""
        # Generate injection if model and parameters provided
        injection = None
        if self.waveform_model is not None:
            injection = self._generate_injection()

        # Setup interferometer
        self.ifo, self.injection_signal, self.window = \
            self._set_interferometer_from_data(data, psd_data, injection)

        self.snr_optimal, self.snr_matched = self._set_ifo_snrs()

        # Prepare rescaled data
        self.rescaled_data = self._prepare_rescaled_data()

        # Compute noise log-evidence
        self.lnz_noise = noise_log_evidence(self.ifo, self.ifo.strain_data.duration)


    def _generate_injection(self):
        """Generate injection waveform from the waveform model."""
        amplitude = self.injection_params.get('amplitude', 5e-22)
        z = self.injection_params.get('z', None)
        rng_key = self.injection_params.get('rng_key', 0)

        if z is not None:
            waveform = self.waveform_model.generate(z=z, rng=jax.random.PRNGKey(rng_key))
        else:
            waveform = self.waveform_model.generate(n=1, rng=jax.random.PRNGKey(rng_key))[0]

        return np.array(waveform * amplitude, dtype=np.float64)

    def _set_interferometer_from_data(self, data, psd_data, injection=None):
        """Set up interferometer with data, PSD, and optional injection."""
        if injection is not None:
            duration = data.duration.value
            if 2 * self.roll_off > duration:
                raise ValueError("2 * roll-off is longer than segment duration.")

            # Crop data to match injection
            data = data[:len(injection)]

        # Create interferometer
        ifo = bilby.gw.detector.get_empty_interferometer(self.detector)
        ifo.strain_data.roll_off = self.roll_off

        # Set strain data
        ifo.strain_data.set_from_gwpy_timeseries(data)
        td_window = ifo.strain_data.time_domain_window(roll_off=self.roll_off)

        # Handle PSD - check if it's TimeSeries or FrequencySeries
        if hasattr(psd_data, 'frequencies'):
            # FrequencySeries case
            psd_interp = np.interp(ifo.frequency_array, psd_data.frequencies.value, psd_data.value)
        else:
            # TimeSeries case - need to compute PSD
            psd_freq_series = psd_data.psd(fftlength=4, method='median')
            psd_interp = np.interp(ifo.frequency_array, psd_freq_series.frequencies.value, psd_freq_series.value)

        ifo.power_spectral_density = PowerSpectralDensity(
            frequency_array=ifo.frequency_array, psd_array=psd_interp
        )

        # Handle injection
        snr_optimal = None
        snr_matched = None
        injection_aligned = injection

        if injection is not None:
            # Inject signal (additive model)
            ifo.strain_data.set_from_gwpy_timeseries(data + injection)


        ifo.minimum_frequency = FLOW
        ifo.maximum_frequency = FMAX

        return ifo, injection_aligned, td_window

    def _prepare_rescaled_data(self):
        """Prepare and rescale interferometer data for numerical stability."""
        # Extract original data
        strain_td_orig = self.ifo.strain_data.time_domain_strain
        strain_fd_orig = self.ifo.strain_data.frequency_domain_strain
        psd_orig = self.ifo.power_spectral_density_array
        frequency_array = self.ifo.frequency_array

        # Apply rescaling
        strain_td_scaled = (strain_td_orig * self.strain_scale).astype(np.float32)
        strain_fd_scaled = (strain_fd_orig * self.strain_scale).astype(np.complex64)
        psd_scaled = (psd_orig * self.psd_scale).astype(np.float32)

        # Package into dictionary for easy access
        rescaled_data = {
            'strain_td': strain_td_scaled,
            'strain_fd': strain_fd_scaled,
            'psd_array': psd_scaled,
            'frequency_array': frequency_array.astype(np.float32),
            'sampling_frequency': self.ifo.strain_data.sampling_frequency,
            'df': frequency_array[1] - frequency_array[0],
            'strain_scale': self.strain_scale,
            'psd_scale': self.psd_scale
        }

        return rescaled_data

    def get_snrs(self):
        """Get computed SNR values."""
        return {
            'optimal_snr': self.snr_optimal,
            'matched_snr': self.snr_matched
        }

    def compute_snr(self, waveform:np.ndarray):
        """Compute optimal and matched SNR for a given waveform."""
        freq_waveform, _ = nfft(waveform, SAMPLING_FREQUENCY)
        snr_optimal = np.sqrt(self.ifo.optimal_snr_squared(signal=freq_waveform)).real
        snr_matched = self.ifo.matched_filter_snr(signal=freq_waveform)
        return snr_optimal, np.abs(snr_matched)

    def _set_ifo_snrs(self):
        if self.injection_signal is None:
            return None, None
        snr_optimal, snr_matched = self.compute_snr(self.injection_signal)
        self.ifo.meta_data.update(
            optimal_SNR=snr_optimal,
            matched_SNR=snr_matched,
        )
        return snr_optimal, snr_matched


def noise_log_evidence(ifo, waveform_duration):
    """
    The noise hypothesis (ie data = noise) has no parameters.
    Hence, the log-evidence is just the lnLikelihood.
    """
    log_l = -2. / waveform_duration * np.sum(
        abs(ifo.frequency_domain_strain) ** 2 /
        ifo.power_spectral_density_array)
    return log_l.real