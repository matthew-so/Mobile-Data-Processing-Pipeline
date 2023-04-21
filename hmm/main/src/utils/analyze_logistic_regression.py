import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import glob
import sklearn

user_sets = [i for i in range(1)]
verification_logs = "../../../main/projects/Mediapipe/data/"
train_lls = {}
test_lls = {}

# kappa_choices = np.arange(5.0, -5.0, -0.1)
# accuracy = np.zeros(kappa_choices.shape)
# false_positive_rate = np.zeros(kappa_choices.shape)
# false_negative_rate = np.zeros(kappa_choices.shape)

for user in user_sets:
    curr_verification_log = glob.glob(f"{verification_logs}{user}/*.pkl")
    curr_user = curr_verification_log[0].split("/")[-1].split("_")[0]
    print(curr_user)

    train_lls[curr_user] = None
    test_lls[curr_user] = None

    for i in range(len(curr_verification_log)):
        if "correct" not in curr_verification_log[i] and "UI" in curr_verification_log[i]:
            train_lls[curr_user] = pkl.load(open(curr_verification_log[i], 'rb'))
        if "test" in curr_verification_log[i]:
            test_lls[curr_user] = pkl.load(open(curr_verification_log[i], 'rb'))
    
    phrase = 'alligator_above_bed'
    incorrect_train_data = train_lls[curr_user]['incorrect'][phrase]
    print(incorrect_train_data)
    correct_train_data = train_lls[curr_user]['correct'][phrase]
    print(correct_train_data)
    training_data = np.concatenate((incorrect_train_data, correct_train_data))
    # print(training_data)
    labels = np.concatenate((np.zeros((len(incorrect_train_data, ))), np.ones((len(correct_train_data)))))
    print(labels)


