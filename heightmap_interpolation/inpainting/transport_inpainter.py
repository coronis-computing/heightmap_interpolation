from heightmap_interpolation.third_party.MATLAB_Python_inpainting_codes.lib.inpainting import transport



class TransportInpainter():
    """Interphase for using the transport inpainting function from """

    def __init__(self, **kwargs):
        self.dt = kwargs.pop("update_step_size", 0.01)
        self.rel_change_tolerance = kwargs.pop("rel_change_tolerance", 1e-8)
        self.max_iters = int(kwargs.pop("max_iters", 1e8))
        self.iters_inpainting = int(kwargs.pop("iters_inpainting", 40))
        self.iters_anisotropic = int(kwargs.pop("iters_anisotropic", 2))
        self.epsilon = kwargs.pop("epsilon", 1e-10)
        if self.dt <= 0:
            raise ValueError("update_step_size must be larger than zero")
        if self.rel_change_tolerance <= 0:
            raise ValueError("rel_change_tolerance must be larger than zero")
        if self.max_iters <= 0:
            raise ValueError("max_iters must be larger than zero")
        if self.iters_inpainting <= 0:
            raise ValueError("iters_inpainting must be larger than zero")
        if self.iters_anisotropic <= 0:
            raise ValueError("iters_anisotropic must be larger than zero")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be larger than zero")

    def inpaint(self, f, mask):
        return transport(f, mask, self.max_iters, self.rel_change_tolerance, self.dt, self.iters_inpainting, self.iters_anisotropic, self.epsilon)