import numpy as np  # Matrice computations
from pylsl import StreamInlet, resolve_byprop # Library responsible for connection
import utils  # pylsl utilities
import pyautogui #handles mouse curosor movements

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

class ABT:
    alpha = 0
    beta = 0
    theta = 0

# Lengths (in Seconds)
BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Calibrations [Alpha, Beta, Theta]
north = [0, 0, 0]
south = [0, 0, 0]
east = [0, 0, 0]
west = [0, 0, 0]

# Muse 2 Electrodes
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

def dataCollection(inlet, fs, np, eeg_buffer, filter_state, band_buffer, SHIFT_LENGTH=SHIFT_LENGTH, INDEX_CHANNEL=INDEX_CHANNEL, EPOCH_LENGTH=EPOCH_LENGTH):

    """ ACQUIRE DATA """
    eeg_data, timestamp = inlet.pull_chunk(
        timeout=1, max_samples=int(SHIFT_LENGTH * fs))

    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

    eeg_buffer, filter_state = utils.update_buffer(
        eeg_buffer, ch_data, notch=True,
        filter_state=filter_state)

    """ COMPUTE BAND POWERS """
    data_epoch = utils.get_last_data(eeg_buffer,
                                     EPOCH_LENGTH * fs)

    band_powers = utils.compute_band_powers(data_epoch, fs)
    band_buffer, _ = utils.update_buffer(band_buffer,
                                         np.asarray([band_powers]))

    smooth_band_powers = np.mean(band_buffer, axis=0)


    """ COMPUTE NEUROFEEDBACK METRICS """
    alpha_metric = smooth_band_powers[Band.Alpha] / \
        smooth_band_powers[Band.Delta]

    beta_metric = smooth_band_powers[Band.Beta] / \
        smooth_band_powers[Band.Theta]

    theta_metric = smooth_band_powers[Band.Theta] / \
       smooth_band_powers[Band.Alpha]

    """ COMPUTER RESPONSE """
    ABT.alpha = alpha_metric
    ABT.beta = beta_metric
    ABT.theta = theta_metric

    return ABT

if __name__ == "__main__":

    """ MUSE 2 CONNECTION """
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    info = inlet.info()
    description = info.desc()

    fs = int(info.nominal_srate())

    """ INITIALIZE BUFFERS """
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ CALIBRATE FOR USER"""

    """ GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        while True:

##            """ ACQUIRE DATA """
##            eeg_data, timestamp = inlet.pull_chunk(
##                timeout=1, max_samples=int(SHIFT_LENGTH * fs))
##
##            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
##
##            eeg_buffer, filter_state = utils.update_buffer(
##                eeg_buffer, ch_data, notch=True,
##                filter_state=filter_state)
##
##            """ COMPUTE BAND POWERS """
##            data_epoch = utils.get_last_data(eeg_buffer,
##                                             EPOCH_LENGTH * fs)
##
##            band_powers = utils.compute_band_powers(data_epoch, fs)
##            band_buffer, _ = utils.update_buffer(band_buffer,
##                                                 np.asarray([band_powers]))
##
##            smooth_band_powers = np.mean(band_buffer, axis=0)
##
##
##            """ COMPUTE NEUROFEEDBACK METRICS """
##            alpha_metric = smooth_band_powers[Band.Alpha] / \
##                smooth_band_powers[Band.Delta]
##
##            beta_metric = smooth_band_powers[Band.Beta] / \
##                smooth_band_powers[Band.Theta]
##
##            theta_metric = smooth_band_powers[Band.Theta] / \
##               smooth_band_powers[Band.Alpha]
##            print('A: ', alpha_metric, ' B: ', beta_metric, ' T: ', theta_metric)
            ABT = dataCollection(inlet, fs, np, eeg_buffer, filter_state, band_buffer)
            print('A: ', ABT.alpha, ' B: ', ABT.beta, ' T: ', ABT.theta)

            """ COMPUTER RESPONSE """

    except KeyboardInterrupt:
        print('Closing!')
