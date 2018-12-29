class GlobalVariables :

    #options for running different experiments
    use_samples = 0
    use_pitch = 0
    use_spectrogram = 1
    use_raw_data = 0

    use_dense=0
    use_CNN=1
    #Grid Size
    nRow = 4
    nCol = 4

    #parameters
    sample_state_size = 100
    pitch_state_size= 114 #87
    spectrogram_length=129
    spectrogram_state_size= 259
    raw_data_state_size= 100
    action_size = 4
    batch_size = 32
    Number_of_episodes=50
    timesteps=(nRow+nCol+nRow)
    how_many_times = 10 #How many times to run the same experiment

