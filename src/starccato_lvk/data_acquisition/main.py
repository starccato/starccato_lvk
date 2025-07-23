# from config import DATA_DIR
# from io.strain_loader import load_analysis_chunk_and_psd
# from segments.trigger_generator import get_random_noise_triggers_from_files

#
#
#
#
# def generate_noiseless_trigger_times(N, seed=None):
#     return get_random_noise_triggers_from_files(DATA_DIR, N, seed=seed)
#
#
# def get_data_with_injection(GPS_time, injection, injection_sample_rate=4096):
#     segment = load_strain_segment(GPS_time - 64, 65)
#     psd = compute_psd_before(GPS_time, duration=64, n_chunks=32)
#
#     # Resample and inject
#     inj = np.interp(
#         np.linspace(0, len(segment) / injection_sample_rate, num=len(segment)),
#         np.linspace(0, len(injection) / injection_sample_rate, num=len(injection)),
#         injection
#     )
#
#     injected = segment + inj
#     return injected, psd
