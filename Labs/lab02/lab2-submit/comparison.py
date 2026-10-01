'''
Model: Random Forest

How it works:
The model is composed of multiple decision trees,
and each of them is a bunch of if-else questions.
Each tree is deliberately made different by feeding them different data/training sets
and letting each tree select from a few randomly drawn features at each step.
Since randomness makes each tree make different mistakes, voting can drown out these
diverse errors, leaving the true rules behind.

Why I chose it:
The logic of this sklearn model is pretty straightforward.
Random forest is constructed by combining if-else statements, and it can express the rule 
of "depending on the combination". And it can express rules where the effect of one feature
depends on the value  of another feature, which my from-scratch model cannot do because
it gives each feature one fixed weight.
By using it for comparison, we can determine whether this data requires a more complex boundary.
'''

# Borrow 4 functions from Part 1: load data, train, predict, score
from binary_classification import load_data, train, predict, accuracy

# Borrow the random forest classifier from sklearn
from sklearn.ensemble import RandomForestClassifier

# load_data() returns 5 things; the 5th (feature names) is not needed, so catch it with _
X_train, X_test, y_train, y_test, _ = load_data()

# ---------- From-scratch model ----------
# Retrain with the Part 1 train function; default alpha=0.01 and n_epochs=100, same as Part 1
# verbose=False turns off the loss printout every 10 epochs
# train returns w, b, losses; losses is only needed for plotting, so catch it with _
w, b, _ = train(X_train, y_train, verbose=False)

# Use the learned w, b to predict every test sample (a list of 0/1)
scratch_pred = predict(X_test, w, b)

# Compare predictions with the true labels to get the fraction correct
scratch_acc = accuracy(y_test, scratch_pred)

# ---------- Random Forest ----------
# A forest of 100 trees; random_state fixes the random seed so every run gives the same result
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train: sklearn expects numpy arrays, so convert the tensors with .numpy() first
rf.fit(X_train.numpy(), y_train.numpy())

# score = predict on the test set, compare with true labels, return accuracy
rf_acc = rf.score(X_test.numpy(), y_test.numpy())

# ---------- Comparison ----------
print(f"From-scratch model test accuracy: {scratch_acc:.4f}")
print(f"Random Forest test accuracy:      {rf_acc:.4f}")
print(f"Random Forest train accuracy:     {rf.score(X_train.numpy(), y_train.numpy()):.4f}")

# Discussion:
# My from-scratch model scored higher on the test set (0.9912 vs. 0.9649),
# which is 1 error versus 4 out of 114 test samples. The cancer data appears to
# be close to linearly separable: my model can only draw one flat boundary, yet
# it already reaches 0.9868 training accuracy, so a single weighted sum of the
# features matches the true shape of the data well.
# Random Forest instead can only ask whether a single feature is greater than a
# certain value at each split, so its boundary is made of axis-aligned steps.
# It is hard to approximate a diagonal boundary with these steps, so a few test points near
# the border end up on the wrong side. Since the gap is only 3 samples, a
# different train/test split could change the result, so this shows my
# from-scratch model fits this dataset better, but both models are good in general.