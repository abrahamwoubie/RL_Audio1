class GlobalVariables :

    #options for running different experiments
    use_pitch=0
    use_samples = 1
    use_spectrogram = 0
    use_raw_data = 0

    #Grid Size
    nRow = 20
    nCol = 20

    #parameters
    state_size = 100
    action_size = 4
    batch_size = 32
    Number_of_episodes=100
    timesteps=50
    how_many_times = 10 #How many times to run the same experiment

