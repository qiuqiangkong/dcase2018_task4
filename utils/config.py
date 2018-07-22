sample_rate = 32000
"""number: Target sample rate during feature extraction."""

window_size = 2048
"""int: Size of FFT window."""

overlap = 720
"""int: Amount of overlap between frames."""

seq_len = 240

mel_bins = 64
"""int: Number of Mel bins."""

labels = ['Speech', 'Dog', 'Cat', 'Alarm_bell_ringing', 'Dishes', 'Frying', 'Blender', 'Running_water', 'Vacuum_cleaner', 'Electric_shaver_toothbrush']

lb_to_ix = {lb: i for i, lb in enumerate(labels)}
ix_to_lb = {i: lb for i, lb in enumerate(labels)}

num_classes = len(labels)