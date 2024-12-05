import os
import napari
import numpy as np
import tifffile
import glob
from qtpy.QtWidgets import QPushButton


class ImageAnnotator:
    """
    A class for annotating images using Napari with a single label.

    Parameters
    ----------
    folder_images : str
        Path to the folder containing the images to be annotated.
    output_folder : str
        Path to the folder where the annotated labels will be saved.
    labels_folder : str, optional
        Path to the folder containing preliminary labels.
    label_name : str, optional
        Name of the label layer, by default 'Z-bands'.
    brush_size : int, optional
        Size of the brush for annotation, by default 10.
    threshold : int, optional
        Threshold value for binarizing preliminary labels, by default None.
    """

    def __init__(self, folder_images, output_folder, labels_folder=None, label_name='Z-bands', brush_size=10,
                 threshold=None):
        self.folder_images = folder_images
        self.output_folder = output_folder
        self.labels_folder = labels_folder
        self.label_name = label_name
        self.brush_size = brush_size
        self.threshold = threshold
        self.list_images = glob.glob(os.path.join(self.folder_images, '*.tif'))
        self.current_index = 0

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        self.viewer = napari.Viewer()
        self.labels_layer = None

        self.setup_viewer()

    def save_labels(self, labels, filename):
        """
        Save the labels as a uint8 TIFF file.

        Parameters
        ----------
        labels : np.ndarray
            The label data to be saved.
        filename : str
            The path to the file where labels will be saved.
        """
        tifffile.imwrite(filename, (labels.astype(np.uint8) * 255))

    def load_image(self, image_path):
        """
        Load an image into the Napari viewer.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        np.ndarray
            The image data.
        """
        data = tifffile.imread(image_path)
        if len(self.viewer.layers) > 0:
            self.viewer.layers[0].data = data
        else:
            self.viewer.add_image(data)
        return data

    def load_preliminary_labels(self, image_name):
        """
        Load preliminary labels if available and apply threshold if necessary.

        Parameters
        ----------
        image_name : str
            The name of the image file to match the preliminary label file.

        Returns
        -------
        np.ndarray
            The preliminary label data, binarized if a threshold is set.
        """
        if self.labels_folder:
            label_path = os.path.join(self.labels_folder, image_name)
            if os.path.exists(label_path):
                labels = tifffile.imread(label_path)
                if self.threshold is not None:
                    labels = (labels > self.threshold).astype(np.uint8)
                return labels
        return None

    def load_next_image(self):
        """
        Save the current labels and load the next image for annotation.
        """
        # Save current labels
        output_path = os.path.join(self.output_folder, os.path.basename(self.list_images[self.current_index]))
        self.save_labels(self.labels_layer.data, output_path)

        # Increment the index and load the next image
        self.current_index += 1
        if self.current_index < len(self.list_images):
            image_path = self.list_images[self.current_index]
            data = self.load_image(image_path)
            preliminary_labels = self.load_preliminary_labels(os.path.basename(image_path))
            if preliminary_labels is not None:
                self.labels_layer.data = preliminary_labels
            else:
                self.labels_layer.data = np.zeros_like(data)
            self.labels_layer.mode = 'paint'  # Select the brush tool by default
        else:
            print("All images annotated")
            self.viewer.close()

    def setup_viewer(self):
        """
        Set up the Napari viewer with the initial image and annotation tools.
        """
        image_path = self.list_images[self.current_index]
        data = self.load_image(image_path)

        # Load preliminary labels if available
        preliminary_labels = self.load_preliminary_labels(os.path.basename(image_path))
        if preliminary_labels is not None:
            labels_data = preliminary_labels
        else:
            labels_data = np.zeros_like(data)

        # Set brush size and select brush tool by default
        self.labels_layer = self.viewer.add_labels(labels_data, name=self.label_name)
        self.labels_layer.brush_size = self.brush_size  # Adjust the brush size as needed
        self.labels_layer.mode = 'paint'  # Select the brush tool by default

        # Define the save and next function
        def save_and_next(event=None):
            self.load_next_image()

        # Add a button to the viewer
        save_button = QPushButton('Save and Next')
        save_button.clicked.connect(save_and_next)
        self.viewer.window.add_dock_widget(save_button, area='left', name='Save and Next')

        # Add a keybinding to save and next (e.g., pressing 'n')
        @self.viewer.bind_key('n')
        def save_and_next_binding(viewer):
            save_and_next()

        # Run the napari viewer
        napari.run()
