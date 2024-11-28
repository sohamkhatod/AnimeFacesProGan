
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import argparse

parser = argparse.ArgumentParser(
                    prog='ProGan64',
                    description='Generates anime faces trained on tpu for about 6-7 hours',
                    epilog='Made by Soham Khatod')

parser.add_argument('-n','--number-images',default = 1,help = 'No of images to display')
parser.add_argument('-s','--size',default = 3,help = 'size of the complete plot')
parser.add_argument('-g','--Generator',default='./train/Generator.keras',help = 'Location of generator model. Note need file should have name Generator.keras')
parser.add_argument('-d','--Discriminator',default='./train/Disciminator.keras',help = 'Location of Discriminator model. Note need file should have name Discriminator.keras')
parser.add_argument('--Display-disc',default=False,help = 'Display Discriminator score')
parser.add_argument('--save',default = None,help = 'save the image as given name')

@tf.keras.utils.register_keras_serializable()
class PixelNormalization(tf.keras.layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # computing pixel values

        #Perform l2 norm
        values = inputs**2.0
        mean_values = tf.reduce_mean(values, axis=-1, keepdims=True)
        mean_values += 1.0e-8
        l2 = tf.sqrt(mean_values)
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape


@tf.keras.utils.register_keras_serializable()
class MinibatchStdev(tf.keras.layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        squ_diffs = tf.square(inputs - mean)
        mean_sq_diff = tf.reduce_mean(squ_diffs, axis=0, keepdims=True)
        mean_sq_diff += 1e-8
        stdev = tf.sqrt(mean_sq_diff)

        mean_pix = tf.reduce_mean(stdev, keepdims=True)
        shape = tf.shape(inputs)
        output = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

        combined = tf.keras.backend.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)

args, unknown_args = parser.parse_known_args()
no_images =int(args.number_images)
figsize = int(args.size)
GENERATOR_LOCATION = args.Generator
DISC_LOCATION = args.Discriminator
DISPLAY_DISC = args.Display_disc
save_image_name = args.save

gen = tf.keras.models.load_model(GENERATOR_LOCATION,custom_objects = {'PixelNormalization':PixelNormalization})
if DISPLAY_DISC:
    disc = tf.keras.models.load_model(DISC_LOCATION,custom_objects = {'MinibatchStdev':MinibatchStdev})

def display_image(num_images=1, individual_figsize=(6, 6)):
    cols = min(num_images, 5)  # Max columns are 5
    rows = math.ceil(num_images / cols)

    figsize_total = (individual_figsize[0] * cols, individual_figsize[1] * rows)

    fig, axs = plt.subplots(rows, cols, figsize=figsize_total)
    axs = axs.flatten() if rows > 1 else [axs]  # Ensure axs is iterable

    for idx in range(num_images):
        noise = tf.random.normal((1, 256))
        generate_image = gen(noise)  # Replace `gen` with your generator function

        if DISPLAY_DISC:
            disc_score = disc(generate_image)  # Replace `disc` with your discriminator function
            axs[idx].text(0.5, -0.1, f'Discriminator rating: {float(disc_score):.2f}', ha='center', transform=axs[idx].transAxes)

        axs[idx].imshow(((generate_image[0].numpy() + 1) * 127.5).astype(int))
        axs[idx].axis('off')

    for idx in range(num_images, len(axs)):
        axs[idx].axis('off')
    if save_image_name is not None:
        fig.savefig(save_image_name  ,dpi=400)
    plt.tight_layout()
    plt.show()

# Example usage
display_image(no_images, individual_figsize=(figsize, figsize))