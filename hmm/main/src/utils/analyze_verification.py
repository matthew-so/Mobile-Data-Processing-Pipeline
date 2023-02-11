import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import glob


user_sets = [i for i in range(12)]
verification_logs = "../../../main/projects/Mediapipe/data/"
train_lls = {}
test_lls = {}
kappa_choices = np.arange(5.0, -5.0, -0.1)
accuracy = np.zeros(kappa_choices.shape)
false_positive_rate = np.zeros(kappa_choices.shape)
false_negative_rate = np.zeros(kappa_choices.shape)
for user in user_sets:
    curr_verification_log = glob.glob(f"{verification_logs}{user}/*.pkl")
    curr_user = curr_verification_log[0].split("/")[-1].split("_")[0]

    train_lls[curr_user] = pkl.load(open(curr_verification_log[0], 'rb')) if "correct" in curr_verification_log[0] else pkl.load(open(curr_verification_log[1], 'rb'))
    test_lls[curr_user] = pkl.load(open(curr_verification_log[0], 'rb')) if "test" in curr_verification_log[0] else pkl.load(open(curr_verification_log[1], 'rb'))
    
    for sign in train_lls[curr_user]:
        curr_probs = train_lls[curr_user][sign]
        mean, std = np.mean(curr_probs), np.std(curr_probs)
        train_lls[curr_user][sign] = [mean, std]
    
    for idx, curr_kappa in enumerate(kappa_choices):

        classified_positive_correctly = 0
        classified_negative_correctly = 0
        classified_positive_incorrectly = 0
        classified_negative_incorrectly = 0

        negative_phrases = test_lls[curr_user]["incorrect"]
        positive_phrases = test_lls[curr_user]["correct"]

        for phrase in negative_phrases:
            mean, std = train_lls[curr_user][phrase]
            for probability in negative_phrases[phrase]:
                if probability >= mean + curr_kappa * std:
                    #classified as positive
                    classified_negative_incorrectly += 1
                else:
                    #classified as negative
                    classified_negative_correctly += 1
        
        for phrase in positive_phrases:
            mean, std = train_lls[curr_user][phrase]
            for probability in positive_phrases[phrase]:
                if probability >= mean + curr_kappa * std:
                    #classified as positive
                    classified_positive_correctly += 1
                else:
                    #classified as negative
                    classified_positive_incorrectly += 1
        
        accuracy[idx] += (classified_negative_correctly + classified_positive_correctly)/(classified_negative_incorrectly + classified_negative_correctly + classified_positive_correctly + classified_positive_incorrectly)
        false_negative_rate[idx] += (classified_positive_incorrectly)/(classified_positive_incorrectly + classified_negative_correctly + 1e-100)
        false_positive_rate[idx] += (classified_negative_incorrectly)/(classified_negative_incorrectly + classified_positive_correctly + 1e-100)

accuracy *= 100.0/len(user_sets)
false_positive_rate *= 100.0/len(user_sets)
false_negative_rate *= 100.0/len(user_sets)
best_acc_index = np.argmax(accuracy)
print("Kappa = " + str(kappa_choices[best_acc_index]))
print("Best acc = " + str(accuracy[best_acc_index]))
print("False positive there = " + str(false_positive_rate[best_acc_index]))
print("False negative there = " + str(false_negative_rate[best_acc_index]))


plt.plot(kappa_choices, accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Kappa')
plt.title('Accuracy vs Kappa')
plt.show()

plt.plot(kappa_choices, false_positive_rate)
plt.ylabel('False Positive')
plt.xlabel('Kappa')
plt.title('False Positive vs Kappa')
plt.show()

plt.plot(kappa_choices, false_negative_rate)
plt.ylabel('False Negative')
plt.xlabel('Kappa')
plt.title('False Negative vs Kappa')
plt.show()