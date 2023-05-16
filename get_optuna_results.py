import numpy as np
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt

for study_name, storage_name in [
                        #['optimize_HME_100k_reduced_SAN', 'sqlite:///optimize_HME_100k_reduced_SAN.db'],
                        ['optimize_Base_Soma_Subtr_CAN', 'sqlite:///optimize_Base_Soma_Subtr_CAN.db'],
                    ]:

    print(study_name)
    experiment = optuna.load_study(study_name=study_name,
                                   storage=storage_name)

    print('Total trials:', len(experiment.trials))
    #assert len(experiment.trials) >= 100

    plt.figure()

    values_list = []
    params_list = []

    valid_trials = 0

    #for i, trial in enumerate(experiment.trials[0:100]):
    for i, trial in enumerate(experiment.trials):
        if trial.state == TrialState.COMPLETE:
            valid_trials += 1

            values_list.append(trial.values[0])
            params_list.append(str(trial.params))

    params_values = {}
    for i, params in enumerate(params_list):
        if params in params_values.keys():
            params_values[params].append(values_list[i])
        else:
            params_values[params] = [values_list[i]]

    avg_params_list = []

    values_list_new = []
    for param in params_values.keys():
        if len(params_values[param]) > 0:
            avg_params_list.append([param, np.average(params_values[param]), np.std(params_values[param]), len(params_values[param]), np.amin(params_values[param]), np.amax(params_values[param])])
            values_list_new.append(np.average(params_values[param]))
            #values_list_new.append(np.amin(params_values[param]))
            #values_list_new.append(np.amax(params_values[param]))

    values_list_new = np.array(values_list_new)

    if experiment.direction == optuna.study.StudyDirection.MINIMIZE:
        order = np.argsort(-values_list_new)
    else:
        order = np.argsort(values_list_new)

    print('Unique configs:', len(params_values.keys()))

    print('Total valid trials:', valid_trials)

    # print(experiment.trials[order[-3]])
    # print(experiment.trials[order[-2]])
    # print(experiment.trials[order[-1]])
    for i in range(1, min(50, len(avg_params_list))):
        print(i, avg_params_list[order[-i]][1], avg_params_list[order[-i]][2], avg_params_list[order[-i]][4], avg_params_list[order[-i]][5], avg_params_list[order[-i]][3])

    print()

    for i in range(1, min(50, len(avg_params_list))):
        print(i, avg_params_list[order[-i]][0])
    # print(experiment.trials[order[-3]].params)
    # print(experiment.trials[order[-2]].params)
    # print(experiment.trials[order[-1]].params)
    print()

    plt.plot(values_list)
    plt.xlabel('Trial #')
    plt.ylabel('Score')
    plt.show()

