import re
from typing import Literal, Tuple, Union

import cv2
import numpy as np
from PIL import Image

## Constants and types #############
RGB = 'RGB'
GREY = 'GREY'
RGBA = 'RGBA'
ColourMode = Literal['RGB', 'GREY', 'RGBA']

####################################


class NpImage:
    """
    Class to load an image from a path into a Numpy array and store associated metadata
    
    Example use:
    ```
    im = NpImage(sample_path, scale_factor=0.8, colour=RGB).get_np_image_exif()
    ```
    """

    colour: ColourMode
    '''(ColourMode): colour profile to load the image in'''

    exif_transpose: Union[Image.Transpose, None] = None
    '''(Image.Transpose | None): EXIF orientation transpose method'''

    im_path: str
    '''(str): the path to the image file'''

    im: np.ndarray = None
    '''(np.ndarray): the final loaded image (after resizing and recolouring)'''

    PIL_image: Image = None
    '''(Image): the resized PIL image with EXIF transpose applied if needed'''

    new_dim: Union[Tuple[int, int], int, None]
    '''(int | Tuple(int, int) | None): the input final image dimensions to use (if set in __init__ function)'''

    original_PIL_image: Image
    '''(Image): the original PIL image'''

    scale_factor: Union[float, None]
    '''(float | None): the factor to scale the image by on load (if set in __init__ function)'''

    EXIF_ORIENTATION_IDX = 0x0112
    '''the index of EXIF orientation information in PIL images'''

    def __init__(self, im_path: str, new_dim: Tuple[int, int] = None,
                 scale_factor: float = None, colour: ColourMode = RGB) -> np.ndarray:
        """
        Read an image using Pillow, then optionally resize after processing EXIF metadata.
        cv2.imread doesn't work well with UTF characters, hence using Pillow.
        Args:
            im_path (string)                :  path to the image to load.
            new_dim (int | (int, int))*     :  new shortest edge length (int) or new image dimensions (width, height) to resize image to. If one dimension is -1, calculate it from the other dimension keeping aspect ratio.
            scale_factor (float)*           :  amount to scale image by (0.1 -> image will be 1/10 original size).
            colour (Literal['RGB', 'RGBA', 'GREY']) :  whether to load the image as a 3-channel rgb, 1-channel greyscale image, or 4-channel RGBA image.
            *Exactly one of new_dim or scale_factor should be specified
        Returns:
            (np.ndarray[np.uint8]): the loaded image - a (h,w,3) or (h,w) numpy array.
        Examples:
        ```
        # Load RGB PIL image with EXIF transform applied if needed
        pil_image = NpImage(path).get_PIL_image_exif()
        # Load numpy image at half size
        resized_np_image = NpImage(path, scale_factor=0.5).get_np_image_exif()

        # Scale image to have width 900 (preserves aspect ratio)
        resized_np_image = NpImage(path, new_dim=(900, -1)).get_np_image_exif() 
        # Scale image to have shape 900x800 px (not preserving aspect ratio)
        resized_np_image = NpImage(path, new_dim=(900, 800)).get_np_image_exif()
        # Scale image to have shortest size length 900 (preserves aspect ratio)
        resized_np_image = NpImage(path, new_dim=900).get_np_image_exif()
        ```
        """

        self.colour = colour
        self.im_path = im_path

        self.new_dim = new_dim
        self.scale_factor = scale_factor

        # Open image using PIL
        PIL_image = Image.open(im_path)
        if self.colour == RGB:
            self.original_PIL_image = PIL_image.convert(self.colour)
        else:
            self.original_PIL_image = PIL_image

        # Load image EXIF transpose metadata into class
        self._create_exif_transpose(self.original_PIL_image)

    def get_PIL_image_exif(self) -> Image:
        """Return PIL image, transposed according to EXIT metadata with resizing handled if needed"""
        if self.PIL_image is None:
            exif_pil_image = self._handle_exif_transpose(
                self.original_PIL_image)
            # Resize image if required
            self.PIL_image = self._handle_resize_img(
                exif_pil_image, new_dim=self.new_dim, scale_factor=self.scale_factor)

        return self.PIL_image

    def get_np_image_exif(self) -> np.ndarray:
        """
        Loads the final image object into the Class.
        Handles EXIF orientation adjustment of self.PIL_image, resizing if needed and colour profile adjustment.
        Should be called at most once per Class instance.
        Args:
            None.
        Returns:
            The final loaded, rescaled, recoloured numpy image (np.ndarray).
        """
        if (self.im is None):
            image = self._handle_exif_transpose(self.original_PIL_image)

            # Resize image if required
            image = self._handle_resize_img(
                image, new_dim=self.new_dim, scale_factor=self.scale_factor)

            # Convert image to a numpy array
            np_image: np.ndarray = np.array(image, dtype=np.uint8)

            # Correct image colour profile
            np_image = self._handle_recolour_img(np_image)

            self.im = np_image

        return self.im

    def _handle_resize_img(self, image: Image, new_dim: Tuple[int, int] = None,
                           scale_factor: float = None) -> Image:
        """
        Resize image if required, otherwise return the unchanged image
        Args:
            image (Image)               :  the PIL image to resize.
            new_dim (int | (int, int))* :  if an integer is specified, set to the new shortest edge length (keeping askect ratio). If a tuple is given, treat as new image dimensions (width, height) to resize image. If one dimension is -1, calculate it from the other dimension keeping aspect ratio.
            img_scale (float)*          :  amount to scale image by (0.1 -> image will be 1/10 original size).
            *Exactly one of new_dim or img_scale should be specified.
        Returns:
            (Image): the resized PIL image.
        """
        if new_dim is None and scale_factor is None:
            return image  # Return unchanged image

        assert new_dim is None or scale_factor is None, "At most one of new_dim and scale_factor may be specified."

        # Get image dimensions pre-resizing
        (old_width, old_height) = image.size
        if scale_factor is not None and scale_factor != 1:
            new_width = round(old_width*scale_factor)
            new_height = round(old_height*scale_factor)
            image = image.resize((new_width, new_height), Image.BOX)
        else:
            # If new_dim is an inteer, set it to the shortest side width
            if isinstance(new_dim, int):
                if old_width < old_height:
                    new_dim = (new_dim, -1)
                else:
                    new_dim = (-1, new_dim)

            # Set new dimensions
            new_width = new_dim[0]
            new_height = new_dim[1]
            # If new_width or new_height are -1, calculate the missing dimension
            if new_height == -1:
                scale_factor = new_width/old_width
                new_height = round(scale_factor*old_height)
            elif new_width == -1:
                scale_factor = new_height/old_height
                new_width = round(scale_factor*old_width)

            image = image.resize((new_width, new_height), Image.BOX)

        return image

    def _handle_recolour_img(self, image: np.ndarray) -> np.ndarray:
        """
        Recolour image if needed according to the value of self.colour
        Args:
            image (np.ndarray) : the image to recolour according to self.colour.
        Returns:
            (np.ndarray): the recoloured image.
        """
        im_dimension = image.ndim

        if self.colour == GREY:
            # Return a greyscale image
            if im_dimension == 2:
                return image  # If image is already greyscale

            _, _, chans = image.shape
            if chans == 3:
                # Assume image is in RGB format
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif chans == 4:
                # Convert an RGBA image (e.g. from a png) to greyscale
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                raise ValueError("Unexpected number of image channels. Image " +
                                 self.im_path + " has shape " + str(image.shape))
        elif self.colour == RGB:
            # Return an RGB image
            if im_dimension == 2:
                # If image is greyscale, convert to RGB
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif im_dimension == 3:
                _, _, chans = image.shape
                if chans == 3:
                    return image
                elif chans == 4:
                    # If there are 4 channels, e.g. for a png image in RGBA format, convert to rgb
                    return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    raise ValueError("Unexpected number of image channels. Image " +
                                     self.im_path + " has shape " + str(image.shape))

            else:
                raise Exception("Unexpected colour value " + str(self.colour))
        elif self.colour == RGBA:
            # Return an RGB image
            if im_dimension == 2:
                # If image is greyscale, convert to RGB
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif im_dimension == 3:
                _, _, chans = image.shape
                if chans == 3:
                    return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
                elif chans == 4:
                    return image
                else:
                    raise ValueError("Unexpected number of image channels. Image " +
                                     self.im_path + " has shape " + str(image.shape))

            else:
                raise Exception("Unexpected colour value " + str(self.colour))

    def _create_exif_transpose(self, image: Image):
        """
        If the input image has EXIF orientation information available,
        set self.exif_transpose to the Image.Transpose orientation
        Args:
            image (Image): the PIL image to get the exif transpose information for.
        Returns:
            None.
        """
        exif = image.getexif()
        orientation: Union[int, None] = exif.get(self.EXIF_ORIENTATION_IDX)
        # Map EXIF orientation to PIL transpose value
        exif_transpose: Union[Image.Transpose, None] = {
            2: Image.Transpose.FLIP_LEFT_RIGHT,
            3: Image.Transpose.ROTATE_180,
            4: Image.Transpose.FLIP_TOP_BOTTOM,
            5: Image.Transpose.TRANSPOSE,
            6: Image.Transpose.ROTATE_270,
            7: Image.Transpose.TRANSVERSE,
            8: Image.Transpose.ROTATE_90,
        }.get(orientation)

        if exif_transpose is not None:
            self.exif_transpose = exif_transpose

    def _handle_exif_transpose(self, image: Image) -> Image:
        """ 
        If an image has an EXIF Orientation tag, return a new image that is
        transposed accordingly and save transormation information in the class. 
        Otherwise, return a copy of the image.
        N.B. the transpose logic is copied almost verbatim from 
            PIL v9.1.0, file PIL/ImageOps.py, function exif_transpose
        Arguments:
            image (Image): the image to transpose.
        Returns:
            (Image): the transposed image.
        """
        self._create_exif_transpose(image)

        if self.exif_transpose is not None:
            transposed_image: Image = image.transpose(self.exif_transpose)
            transposed_exif = transposed_image.getexif()
            # Update exif orientation information in returned image
            if self.EXIF_ORIENTATION_IDX in transposed_exif:
                del transposed_exif[self.EXIF_ORIENTATION_IDX]
                if "exif" in transposed_image.info:
                    transposed_image.info["exif"] = transposed_exif.tobytes()
                elif "Raw profile type exif" in transposed_image.info:
                    transposed_image.info[
                        "Raw profile type exif"
                    ] = transposed_exif.tobytes().hex()
                elif "XML:com.adobe.xmp" in transposed_image.info:
                    transposed_image.info["XML:com.adobe.xmp"] = re.sub(
                        r'tiff:Orientation="([0-9])"',
                        "",
                        transposed_image.info["XML:com.adobe.xmp"],
                    )
            return transposed_image

        # If image is not transposed, return a copy of the original image
        return image.copy()

    def exif_orientation_transpose(self, im: np.ndarray) -> np.ndarray:
        """Apply transpose to any numpy uint8 image using the original image EXIF orientation
        Arguments:
            im (np.ndarray[np.uint8]): the image to transform (if self.exif_transpose is not None).
        Returns:
            (np.ndarray): the transposed numpy image.
        """
        assert im.dtype == np.uint8, "Expected a uint8 image to allow conversion to PIL image"

        if self.exif_transpose is None:
            # If no EXIF transpose is needed, return a copy of the image
            return im.copy()
        else:
            # Convert image to PIL Image and transpose
            pil_im = Image.fromarray(im)
            transposed_im = pil_im.transpose(self.exif_transpose)
            # Convert Image back to numpy array
            return np.asarray(transposed_im)
