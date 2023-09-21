# Load data (4774 images from 20 classes)
# Get data from https://drive.google.com/drive/folders/0Bxxqx_AAp2u2Zkp4cGxoNVEzb3M
# Place data in a Google Drive folder and set the datadir variable below with the path of this folder

datadir = '/content/drive/My Drive/Colab Notebooks/ARGOS_public'
trainingset = datadir + '/train/'
testset = datadir + '/test/'

data_augmentation_level = 2   # Level of data augmentation [0: none, 1: low, 2: high]
batch_size = 32               # 32 or 64 depending on situation
input_shape = ()

# Create training set below (with data augmentation)

if data_augmentation_level == 0:
    train_datagen = ImageDataGenerator(rescale = 1. / 255)
    train_shuffle = False
elif data_augmentation_level == 1:
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        zoom_range = 0.1,
        rotation_range = 10
    )
    train_shuffle = False
else:    
    train_datagen = ImageDataGenerator(
    	rescale = 1. / 255,
        zoom_range = 0.1,
        rotation_range = 20,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        vertical_flip = False
    )
    train_shuffle = True

train_generator = train_datagen.flow_from_directory(
    directory = trainingset,
    target_size = (224,224),
    color_mode = "rgb",
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = train_shuffle
)

# Create validation set below

test_datagen = ImageDataGenerator(rescale = 1. / 255)

test_generator = test_datagen.flow_from_directory(
    directory = testset,
    target_size = (224,224),
    color_mode = "rgb",
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = False
)

# Resulting numbers
num_samples = train_generator.n
num_classes = train_generator.num_classes
input_shape = train_generator.image_shape
classnames = [k for k,v in train_generator.class_indices.items()]

print("Image input %s" %str(input_shape))
print("Classes: %r" %classnames)
print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
print('Loaded %d test samples from %d classes.' %(test_generator.n,test_generator.num_classes))
print('Data augmentation level: %d' %(data_augmentation_level))
print("\n---------------------------------------------------------------\n")



# Show n random images
'''
n = 3
x,y = train_generator.next()	# x,y size is train_generator.batch_size

for i in range(0,n):
    image = x[i]
    label = y[i].argmax()  		# Categorical from one-hot-encoding
    print(classnames[label])
    plt.imshow(image)
    plt.show()
print("\n---------------------------------------------------------------\n")
'''

