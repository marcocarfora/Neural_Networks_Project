# Training process

# Implementing an Early Stopping callback to terminate early in case of no or decreasing improvement 
#stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

steps_per_epoch = train_generator.n // train_generator.batch_size
val_steps = test_generator.n // test_generator.batch_size + 1
epochs = 30

try:
    history = model.fit(
    	train_generator,
    	epochs = epochs,
    	verbose = 1,
      #callbacks = [stopping],
      steps_per_epoch = steps_per_epoch,
      validation_data = test_generator,
      validation_steps = val_steps
)
except KeyboardInterrupt:
    pass

print("\n---------------------------------------------------------------\n")

