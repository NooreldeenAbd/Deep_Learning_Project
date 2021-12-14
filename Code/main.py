import Neural
import Images
import matplotlib.pyplot as plt
import numpy as np

# Define some house keeping items
classes = {0: "cloud", 1: "rain", 2: 'shine', 3: 'sunrise'}
path_train = 'Replace With File Path'
path_test = 'Replace With File Path'
path_camera = 'Replace With File Path'
dim = (100, 100)

# import images
train_images, train_targets = Images.import_and_resize_all(path_train, dim)
test_images, test_targets = Images.import_and_resize_all(path_test, dim)
camera_images, camera_targets = Images.import_and_resize_all(path_camera, dim)


# Normalize and convert to np array
train_images_normal = np.array(train_images)/255
train_targets = np.array(train_targets)
test_images_normal = np.array(test_images)/255
test_targets = np.array(test_targets)
camera_images_normal = np.array(camera_images)/255
camera_targets = np.array(camera_targets)

# Show some training images
plt.figure(figsize=(5,5))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(train_targets[i])
plt.show()

# Train the network and get accuracy
test_loss, test_acu = Neural.train(train_images_normal,
                                   train_targets, test_images_normal,
                                   test_targets, len(classes))

# Use network to classify images
res_test = Neural.test(test_images_normal)
res_cam = Neural.test(camera_images_normal)
res_test = res_test.astype(int)
res_cam = res_cam.astype(int)

# Compute and print the confusion matrix of the test images
print('Accuracy: ', test_acu)
print('Test Data Confusion Matrix: ')
conf_matrix = Neural.compute_confusion(test_targets, res_test, len(classes))
print(conf_matrix)

# Display images from my camera and their classes
for i in range(len(camera_images)):
    plt.imshow(camera_images[i])
    plt.xlabel(classes[res_cam[i]])
    plt.show()
