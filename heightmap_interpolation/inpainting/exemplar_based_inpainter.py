from exemplar_based_inpainting.inpainter import Inpainter
import numpy as np
import cv2

class ExemplarBasedInpainter():
    """Interphase for using the inpainter from the exemplar_based_inpainter package"""

    def __init__(self, **kwargs):
        self.patch_size = kwargs.pop("patch_size", 9)
        self.search_original_source_only = kwargs.pop("search_original_source_only", False)
        # self.search_color_space = kwargs.pop("search_color_space", "gray")
        self.search_color_space = "gray" # DevNote: no reason to use a color space, it will always be a single channel elevation map
        self.plot_progress = kwargs.pop("plot_progress", False)
        self.out_progress_dir = kwargs.pop("out_progress_dir", "")
        self.show_progress_bar = kwargs.pop("show_progress_bar", True)
        self.patch_preference = kwargs.pop("patch_preference", "closest")
        self.inpainter = Inpainter(
            patch_size=self.patch_size, 
            search_original_source_only=self.search_original_source_only,
            search_color_space=self.search_color_space,
            plot_progress=self.plot_progress, 
            out_progress_dir=self.out_progress_dir ,                  
            show_progress_bar=self.show_progress_bar, 
            patch_preference=self.patch_preference)

    def inpaint(self, f, mask):
        # Normalize data between 0 and 1
        min_f = np.min(f[mask])
        max_f = np.max(f[mask])
        f_norm = (f-min_f)/(max_f-min_f)
        f_norm[~mask] = 0

        # Inpaint 
        mask_inp = ~mask
        mask_inp = mask_inp.astype(np.uint8)*255 # as uint8 (255 == missing area)
        f_inp = self.inpainter.inpaint(f_norm, mask_inp)

        # De-normalize
        f_inp = min_f + f_inp * (max_f-min_f)

        return f_inp


