# VGG19 model implementation

def VGG19(input_shape, num_classes, lr=0.0001):
    # Initializing a Sequential model
    model = Sequential()

    # Creating first block: (2 Convolution + 1 Max pool)
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides= (2,2)))

    # Creating second block: (2 Convolution + 1 Max pool)
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides= (2,2)))

    # Creating third block: (4 Convolution + 1 Max pool)
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides= (2,2)))

    # Creating fourth block: (4 Convolution + 1 Max pool)
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides= (2,2)))

    # Creating fifth block: (4 Convolution + 1 Max pool)
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides= (2,2)))

    # Flattening the pooled image pixels
    model.add(Flatten())

    # Creating 2 Dense Layers with 0.5 Dropout
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # Creating an output layer
    model.add(Dense(units=num_classes, activation='softmax'))

    # Compiling
    adam = optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model


# Take the model
model = VGG19(input_shape, num_classes)
model.summary()
print("\n---------------------------------------------------------------\n")

