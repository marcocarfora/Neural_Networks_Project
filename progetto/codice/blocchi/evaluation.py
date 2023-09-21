# Evaluation of the model


# Accuracy on test set
loss, acc = model.evaluate(test_generator,verbose=1,steps=val_steps)
print('Test loss: %f' %loss)
print('Test accuracy: %f' %acc)
print("---------------------------------------------------------------\n")


# Precision, recall and F1-score
preds = model.predict(test_generator,verbose=1,steps=val_steps)
Ypred = np.argmax(preds, axis=1)
Ytest = test_generator.classes  # shuffle=False in test_generator
print("---------------------------------------------------------------\n")

print(classification_report(Ytest, Ypred, labels=None, target_names=classnames, digits=3))
print("---------------------------------------------------------------\n")


# Confusion Matrix
cm = confusion_matrix(Ytest, Ypred)
print(cm, end="\n\n")
print("---------------------------------------------------------------\n")

pl.matshow(cm)
#pl.title("Confusion matrix of the classifier")
pl.colorbar()
pl.show()
print("---------------------------------------------------------------\n")


# Plot results

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

