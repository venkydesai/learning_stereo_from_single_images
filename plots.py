import os
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import uuid

def save_scalar_plots(logdir, save_dir):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tags = event_acc.Tags()

    for tag_type in ['scalars']:
        if tag_type in tags:
            scalar_tags = tags[tag_type]
            for tag in scalar_tags:
                event_data = event_acc.Scalars(tag)
                if event_data:
                    # Fetching all steps and values for the scalar plot
                    steps = [event.step for event in event_data]
                    values = [event.value for event in event_data]

                    # Replace '/' in tag with '-'
                    tag_name = tag.replace('/', '-')

                    plt.figure(figsize=(8, 6))
                    plt.plot(steps, values, marker='o', markersize=4, label=tag_name)
                    plt.xlabel('Steps')
                    plt.ylabel(tag_name)
                    plt.title(f'{tag_name} over Steps')
                    plt.legend()
                    save_path = os.path.join(save_dir, f'{tag_name}_plot_{uuid.uuid4()}.png')

                    plt.savefig(save_path)
                    plt.close()

def save_images(logdir, save_dir):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tags = event_acc.Tags()

    for tag_type in ['images']:
        if tag_type in tags:
            image_tags = tags[tag_type]
            for tag in image_tags:
                image_data = event_acc.Images(tag)
                if image_data:
                    # Fetching only the last image
                    image = image_data[-1]

                    # Replace '/' in tag with '-'
                    tag_name = tag.replace('/', '-')

                    image_path = os.path.join(save_dir, f'{tag_name}_image_{uuid.uuid4()}.png')
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image.encoded_image_string)

# Example usage:
log_directory = '/home/patel.aryam/stereo-from-mono/out/baseline/train'
save_directory = '/home/patel.aryam/stereo-from-mono/plots'

save_scalar_plots(log_directory, save_directory)
save_images(log_directory, save_directory)
