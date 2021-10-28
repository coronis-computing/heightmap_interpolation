from heightmap_interpolation.inpainting.sobolev_inpainter import SobolevInpainter
from heightmap_interpolation.inpainting.tv_inpainter import TVInpainter
from heightmap_interpolation.inpainting.ccst_inpainter import CCSTInpainter
from heightmap_interpolation.inpainting.amle_inpainter import AMLEInpainter
from heightmap_interpolation.inpainting.opencv_inpainter import OpenCVInpainter


def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z


def default_options(method):
    options = {"update_step_size": 0.01,
               "rel_change_tolerance": 1e-8,
               "max_iters": 1e8,
               "relaxation": 0,
               "mgs_levels": 1,
               "print_progress": True,
               "print_progress_iters": 1000,
               "init_with": "zeros",
               "convolver": "opencv",
               "debug_dir": ""}
    if method.lower() == "sobolev":
        options["update_step_size"] = 0.8/4
        options["rel_change_tolerance"] = 1e-5
        options["max_iters"] = 1e5
    elif method.lower() == "tv":
        options["epsilon"] = 1
        options["update_step_size"] = .9*options["epsilon"]/4
        options["rel_change_tolerance"] = 1e-5
        options["max_iters"] = 1e5
        options["show_progress"] = False
    elif method.lower() == "ccst":
        options["update_step_size"] = 0.01
        # options["rel_change_tolerance"] = 1e-8
        options["rel_change_tolerance"] = 1e-8
        options["max_iters"] = 1e8
        # options["relaxation"] = 1.4
        options["tension"] = 0.3
    elif method.lower() == "amle":
        options["update_step_size"] = 0.01
        options["rel_change_tolerance"] = 1e-5
        options["max_iters"] = 1e8
    elif method.lower() == "navier-stokes":
        options = None # OpenCV's inpainters have no options!
    elif method.lower() == "telea":
        options = None # OpenCV's inpainters have no options!
    else:
        print("[ERROR] The required method (" + method + ") is unknown. Available options: sobolev, tv, ccst, amle, navier-stokes, telea")

    return options


def create_fd_pde_inpainter(method, custom_options=None):

    # Collect common parameters
    options = default_options(method)
    if options and custom_options:
        options = merge_two_dicts(options, custom_options)

    if method.lower() == "sobolev":
        inpainter = SobolevInpainter(**options)
    elif method.lower() == "tv":
        options["epsilon"] = 1
        options["update_step_size"] = .9*options["epsilon"]/4
        options["rel_change_tolerance"] = 1e-5
        options["max_iters"] = 1e5
        options["show_progress"] = False
        inpainter = TVInpainter(**options)
    elif method.lower() == "ccst":
        options["update_step_size"] = 0.01
        # options["rel_change_tolerance"] = 1e-8
        options["rel_change_tolerance"] = 1e-8
        options["max_iters"] = 1e8
        # options["relaxation"] = 1.4
        options["tension"] = 0.3
        inpainter = CCSTInpainter(**options)
    elif method.lower() == "amle":
        options["update_step_size"] = 0.01
        options["rel_change_tolerance"] = 1e-7
        options["max_iters"] = 1e8
        inpainter = AMLEInpainter(**options)
    elif method.lower() == "navier-stokes":
        # Convert mask to opencv's inpaint function expected format
        inpainter = OpenCVInpainter(method="navier-stokes", radius=25)
    elif method.lower() == "telea":
        inpainter = OpenCVInpainter(method="telea", radius=5)
    else:
        print("[ERROR] The required method (" + method + ") is unknown. Available options: sobolev, tv, ccst, amle, navier-stokes, telea")

    return inpainter