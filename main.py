"""
Software package for the estimation of hydrodynamic damping coefficients of floating and submerged objects from
triangular meshes in STL format.
Requires a working installation of OpenFOAM 10 (https://openfoam.org/download/10-ubuntu/)

Methods: 

- analyze_model: method to automatically build, run and analyze cases.
- run_model: method to automatically build and run cases and saving the raw output force data.
- load_force_data: method to load OpenFOAM force data into numpy arrays.
- build_case: method to build a case in a temp directory without running it.
- run_case: method to run a case in the temp directory.

"""

__author__ = 'Marius Seidl'
__date__ = '2025-04-11'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
import numpy as np
from stl import mesh
import os
import shutil
import subprocess

# constants
DOMAIN_SIZE_MULTIPLIER = 3. # multiple of bounding cube around model, must be >1. to fit model fully
FLOW_DIRECTION_SIZE_MULTIPLIER = 2. # domain extension in flow direction for linear cases, must be >1.
SIMULATION_TIME_MULTIPLIER = 3. # number of times the flow traverses the full simulation domain
BASE_RESOLUTION = int(30) # number of cells along the base (=non-extended) dimension, ~O(N^3) impact on runtime!
MAX_CFL_NUMBER = 20 # maximum Courant-Friedrichs-Lewy number, lower numbers cause longer runtime but higher quality results.

# evaluation points to estimate first and second order component of model
ANALYSIS_VELOCITIES = [np.array((1.5, .25, .25, .05, .05, .05)),
                       np.array((3., .5, .5, .1, .1, .1))]
CUTOFF_FRACTION = 0.1 # amount of datapoint from beginning of CFD run to discard

def analyze_model(stl_filepath: str,
                  z_constraint: str = 'submerged',
                  z_to_waterline: float = 0.,
                  motion_direction: str = 'forward'
                  ) -> dict:
    """
    Analyze a model provided as STL file to extract hydrodynamic damping coefficients.
    Note that runtimes of multiple hours are normal.

    :param stl_filepath: path to the STL file to be analyzed.
    :param z_constraint: type of situation to be analyzed, either "submerged" or "floating".
    :param z_to_waterline: distance from CG to waterline (negative for CG below free surface). Only used for floating objects.
    :param motion_direction: direction of motion to be analyzed, can be "forward", "backward" or "both".
    :return: nested dictionary containing hydrodynamic damping coefficients.
    Entries are result["coefficient_order"]["motion_direction"]["coefficients/uncertainty"].
    coefficient_order in ["linear","quadratic"]. motion_direction values are the same as input values.
    For each coefficient a matching uncertainty is provided.
    """

    # validate keyword inputs
    if z_constraint not in ['floating', 'submerged']:
        raise ValueError('z_constraint must be either floating or submerged.')
    if motion_direction not in ['forward', 'backward', 'both']:
        raise ValueError('motion_direction must be either forward, backward, or both.')

    # create potentially used arrays for forces/moments at given velocities/rates
    fm_vr1_forward_mean = np.zeros((6,6))
    fm_vr1_forward_std = np.zeros((6,6))
    fm_vr1_backward_mean = np.zeros((6,6))
    fm_vr1_backward_std = np.zeros((6,6))

    fm_vr2_forward_mean = np.zeros((6,6))
    fm_vr2_forward_std = np.zeros((6,6))
    fm_vr2_backward_mean = np.zeros((6,6))
    fm_vr2_backward_std = np.zeros((6,6))


    # run first velocity
    for idx1, motion in enumerate(['linear', 'rotational']):
        for idx2, axis in enumerate(['x', 'y', 'z']):
            if motion_direction in ['forward', 'both']:
                build_case(ANALYSIS_VELOCITIES[0][3*idx1+idx2], stl_filepath, axis, motion, z_constraint, z_to_waterline)
                run_case()
                time, forces, moments = load_force_data()
                fm_vr1_forward_mean[:3, 3 * idx1 + idx2] = np.mean(forces[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr1_forward_std[:3, 3 * idx1 + idx2] = np.std(forces[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr1_forward_mean[3:, 3 * idx1 + idx2] = np.mean(moments[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr1_forward_std[3:, 3 * idx1 + idx2] = np.std(moments[time > CUTOFF_FRACTION * time[-1]], axis=0)
            if motion_direction in ['backward', 'both']:
                build_case(-ANALYSIS_VELOCITIES[0][3*idx1+idx2], stl_filepath, axis, motion, z_constraint, z_to_waterline)
                run_case()
                time, forces, moments = load_force_data()
                fm_vr1_backward_mean[:3, 3 * idx1 + idx2] = np.mean(forces[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr1_backward_std[:3, 3 * idx1 + idx2] = np.std(forces[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr1_backward_mean[3:, 3 * idx1 + idx2] = np.mean(moments[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr1_backward_std[3:, 3 * idx1 + idx2] = np.std(moments[time > CUTOFF_FRACTION * time[-1]], axis=0)

    # run second velocity
    for idx1, motion in enumerate(['linear', 'rotational']):
        for idx2, axis in enumerate(['x', 'y', 'z']):
            if motion_direction in ['forward', 'both']:
                build_case(ANALYSIS_VELOCITIES[1][3*idx1+idx2], stl_filepath, axis, motion, z_constraint, z_to_waterline)
                run_case()
                time, forces, moments = load_force_data()
                fm_vr2_forward_mean[:3, 3 * idx1 + idx2] = np.mean(forces[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr2_forward_std[:3, 3 * idx1 + idx2] = np.std(forces[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr2_forward_mean[3:, 3 * idx1 + idx2] = np.mean(moments[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr2_forward_std[3:, 3 * idx1 + idx2] = np.std(moments[time > CUTOFF_FRACTION * time[-1]], axis=0)
            if motion_direction in ['backward', 'both']:
                build_case(-ANALYSIS_VELOCITIES[1][3*idx1+idx2], stl_filepath, axis, motion, z_constraint, z_to_waterline)
                run_case()
                time, forces, moments = load_force_data()
                fm_vr2_backward_mean[:3, 3 * idx1 + idx2] = np.mean(forces[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr2_backward_std[:3, 3 * idx1 + idx2] = np.std(forces[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr2_backward_mean[3:, 3 * idx1 + idx2] = np.mean(moments[time > CUTOFF_FRACTION * time[-1]], axis=0)
                fm_vr2_backward_std[3:, 3 * idx1 + idx2] = np.std(moments[time > CUTOFF_FRACTION * time[-1]], axis=0)

    # estimate hydrodynamic damping coefficients and return them
    result = dict()
    result['linear'] = dict()
    result['quadratic'] = dict()

    if motion_direction in ['forward', 'both']:
        fm_forward_mean_linear = (fm_vr1_forward_mean * (1 / ANALYSIS_VELOCITIES[0] - 1 / ANALYSIS_VELOCITIES[1])
                                  + fm_vr2_forward_mean * (ANALYSIS_VELOCITIES[0] ** 2 / ANALYSIS_VELOCITIES[1] ** 3
                                                           - ANALYSIS_VELOCITIES[0] / ANALYSIS_VELOCITIES[1] ** 2))
        fm_forward_mean_quadratic = (fm_vr2_forward_mean / ANALYSIS_VELOCITIES[1] ** 2
                                     - fm_forward_mean_linear / ANALYSIS_VELOCITIES[1])
        fm_forward_std_linear = np.sqrt(
            (fm_vr1_forward_std * (1 / ANALYSIS_VELOCITIES[0] - 1 / ANALYSIS_VELOCITIES[1])) ** 2
            + (fm_vr2_forward_std * (ANALYSIS_VELOCITIES[0] ** 2 / ANALYSIS_VELOCITIES[1] ** 3
                                     - ANALYSIS_VELOCITIES[0] / ANALYSIS_VELOCITIES[1] ** 2)) ** 2)
        fm_forward_std_quadratic = np.sqrt((fm_vr2_forward_std / ANALYSIS_VELOCITIES[1] ** 2) ** 2
                                           + (fm_forward_std_linear / ANALYSIS_VELOCITIES[1]) ** 2)
        result['linear']['forward'] = {'coefficients': fm_forward_mean_linear, 'uncertainty': fm_forward_std_linear}
        result['quadratic']['forward'] = {'coefficients': fm_forward_mean_quadratic, 'uncertainty': fm_forward_std_quadratic}

    if motion_direction in ['backward', 'both']:
        fm_backward_mean_linear = (fm_vr1_backward_mean * (1 / -ANALYSIS_VELOCITIES[0] - 1 / -ANALYSIS_VELOCITIES[1])
                                  + fm_vr2_backward_mean * (-ANALYSIS_VELOCITIES[0] ** 2 / -ANALYSIS_VELOCITIES[1] ** 3
                                                           + ANALYSIS_VELOCITIES[0] / -ANALYSIS_VELOCITIES[1] ** 2))
        fm_backward_mean_quadratic = (fm_vr2_backward_mean / -ANALYSIS_VELOCITIES[1] ** 2
                                     - fm_backward_mean_linear / -ANALYSIS_VELOCITIES[1])
        fm_backward_std_linear = np.sqrt(
            (fm_vr1_backward_std * (1 / -ANALYSIS_VELOCITIES[0] - 1 / -ANALYSIS_VELOCITIES[1])) ** 2
            + (fm_vr2_backward_std * (-ANALYSIS_VELOCITIES[0] ** 2 / -ANALYSIS_VELOCITIES[1] ** 3
                                     + ANALYSIS_VELOCITIES[0] / -ANALYSIS_VELOCITIES[1] ** 2)) ** 2)
        fm_backward_std_quadratic = np.sqrt((fm_vr2_backward_std / -ANALYSIS_VELOCITIES[1] ** 2) ** 2
                                           + (fm_backward_std_linear / -ANALYSIS_VELOCITIES[1]) ** 2)

        result['linear']['backward'] = {'coefficients': fm_backward_mean_linear, 'uncertainty': fm_backward_std_linear}
        result['quadratic']['backward'] = {'coefficients': fm_backward_mean_quadratic, 'uncertainty': fm_backward_std_quadratic}

    return result

def run_model(stl_filepath: str,
              save_filepath: str,
              velocities: list[tuple[float, float, float, float, float, float]],
              z_constraint: str = 'submerged',
              z_to_waterline: float = 0.
              ) -> None:
    """
    Low-level method to build and run cases without analyzing them but storing the raw output instead.

    :param stl_filepath: path to the STL file to be analyzed.
    :param save_filepath: path to directory in which results are stored.
    :param velocities: list of pairs of velocity and rotational rate as tuple. 6 cases are run per tuple.
    :param z_constraint: type of situation to be analyzed, either "submerged" or "floating".
    :param z_to_waterline: distance from CG to waterline (negative for CG below free surface). Only used for floating objects.
    """

    # validate inputs
    for velocities_ in velocities:
        for velocity in velocities_:
            if abs(velocity) < 1e-10:
                raise ValueError('only non-zero velocities can be used.')
    if z_constraint not in ['floating', 'submerged']:
        raise ValueError('z_constraint must be either floating or submerged.')

    # build and run cases for each velocity rate tuple
    for idx, velocities_, in enumerate(velocities):
        for velocity, axis, motion in zip(velocities_, ['x', 'y', 'z'] * 2, ['linear'] * 3 + ['rotational'] * 3):
            build_case(velocity, stl_filepath, axis, motion, z_constraint, z_to_waterline)
            run_case()
            shutil.copyfile('./temp/postProcessing/forces_object/0/forces.dat',
                            '{}/forces_{}_{}_{}.dat'.format(save_filepath, axis, motion, idx))

def load_force_data(filepath: str = './temp/postProcessing/forces_object/0/forces.dat'
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a OpenFOAM force file and returns content as numpy arrays.

    :param filepath: path to the force file to be loaded.
    :return: tuple of time, forces X, Y, Z (pressure + viscous), moments K, M, N (pressure + viscous) as numpy arrays.
    """

    data = np.loadtxt(filepath, skiprows=3, dtype='str')
    data = np.char.replace(np.char.replace(data,'(',''),')','').astype('float')
    return data[:,0], data[:,1:4] + data[:,4:7], data[:,7:10] + data[:,10:]

def build_case(velocity_rate: float,
               stl_filepath: str,
               axis: str = 'x',
               motion: str = 'linear',
               z_constraint: str = 'floating',
               z_to_waterline: float = 0.,
               ) -> None:
    """
    Builds a new case in the temp directory.

    :param velocity_rate: velocity or rotational rate at which the case will be run.
    :param stl_filepath: path to the STL file to be analyzed.
    :param axis: axis to be analyzed, can be "x", "y", or, "z".
    :param motion: motion to be analyzed, either "linear" or "rotational".
    :param z_constraint: type of situation to be analyzed, either "submerged" or "floating".
    :param z_to_waterline: distance from CG to waterline (negative for CG below free surface). Only used for floating objects.
    """

    #validate keyword inputs
    if axis not in ['x', 'y', 'z']:
        raise ValueError('axis must be either x, y or z.')
    if motion not in ['linear', 'rotational']:
        raise ValueError('motion must be either linear or rotational.')
    if z_constraint not in ['floating', 'submerged']:
        raise ValueError('z_constraint must be either floating or submerged.')

    # clear temp directory to build and run case in
    if os.path.isdir('./temp'):
        shutil.rmtree('./temp')

    # select case and set velocity/rate
    case_path = '{}_{}'.format(motion, axis)
    if motion == 'linear':
        case_path += '_pos' if velocity_rate > 0 else '_neg'
        shutil.copytree('./cases/{}/{}'.format(z_constraint, case_path), './temp/')
        with open('./temp/0/U', 'r') as u_file:
            u = u_file.read()
        with open('./temp/0/U', 'w') as u_file:
            # negative sign, because the flow is opposite to the virtual motion of the object
            u_file.write(u.format(velocity=-velocity_rate))
    else:
        shutil.copytree('./cases/{}/{}'.format(z_constraint, case_path), './temp/')
        with open('./temp/constant/MRFProperties', 'r') as mrf_file:
            mrf = mrf_file.read()
        with open('./temp/constant/MRFProperties', 'w') as mrf_file:
            # MRF zone is rotating counterclockwise
            mrf_file.write(mrf.format(angular_velocity=velocity_rate))

    # load stl and determine dimension (side length of a "bounding cube")
    stl_mesh = mesh.Mesh.from_file(stl_filepath)
    max_dimension = 0.
    for point in stl_mesh.points:
        max_dimension = max(max_dimension, np.max(np.abs(point)))
    max_dimension *= 2.
    # copy stl to case
    shutil.copy(stl_filepath, './temp/constant/triSurface/object.stl')

    # set limits and resolution
    with open('./temp/system/blockMeshDict', 'r') as block_mesh_file:
        block_mesh = block_mesh_file.read()
    with open('./temp/system/snappyHexMeshDict', 'r') as snappy_hex_mesh_file:
        snappy_hex_mesh = snappy_hex_mesh_file.read()

    domain_size = DOMAIN_SIZE_MULTIPLIER * max_dimension
    domain_size_half = domain_size / 2.
    lb = {'x': -domain_size_half, 'y': -domain_size_half, 'z': -domain_size_half}
    ub = {'x': domain_size_half, 'y': domain_size_half, 'z': domain_size_half}

    if motion == 'linear':
        if velocity_rate > 0.:
            lb[axis] -= (FLOW_DIRECTION_SIZE_MULTIPLIER - 1) * domain_size
        else:
            ub[axis] += (FLOW_DIRECTION_SIZE_MULTIPLIER - 1) * domain_size

        res = {'x': BASE_RESOLUTION, 'y': BASE_RESOLUTION, 'z': BASE_RESOLUTION}
        if z_constraint == 'floating':
            res['z'] = int(res['z']/2.)
        if axis != 'z' or z_constraint == 'submerged':# and velocity_rate >= 0.):
            res[axis] = int(res[axis] * FLOW_DIRECTION_SIZE_MULTIPLIER)
        else:
            if velocity_rate < 0.:
                res[axis] += int((FLOW_DIRECTION_SIZE_MULTIPLIER - 1) * BASE_RESOLUTION)

        with open('./temp/system/blockMeshDict', 'w') as block_mesh_file:
            block_mesh_file.write(block_mesh.format(
                x_lb=lb['x'], y_lb=lb['y'], z_lb=lb['z'], x_ub=ub['x'], y_ub=ub['y'], z_ub=ub['z'],
                x_resolution=res['x'], y_resolution=res['y'], z_resolution=res['z']))
    else:
        lb = {key: 0 if key != axis else lb[key] for key in lb}
        ub = {key: 0 if key != axis else ub[key] for key in ub}

        with open('./temp/system/blockMeshDict', 'w') as block_mesh_file:
            # write constant to avoid needing to do calculations in the openFOAM case file
            block_mesh_file.write(block_mesh.format(
                lb=lb[axis], ub=ub[axis], lb_inner=-domain_size_half/2., ub_inner=domain_size_half/2.,
                radius=domain_size_half, negative_radius=-domain_size_half,
                sin45_radius=0.707107*domain_size_half, negative_sin45_radius=-0.707107*domain_size_half,
                cos225_radius=0.923880*domain_size_half, negative_cos225_radius=-0.923880*domain_size_half,
                sin225_radius=0.382683*domain_size_half, negative_sin225_radius=-0.382683*domain_size_half,
                resolution=BASE_RESOLUTION, resolution_half=int(BASE_RESOLUTION/2),
                resolution_quarter=int(BASE_RESOLUTION/4)))

    with open('./temp/system/snappyHexMeshDict', 'w') as snappy_hex_mesh_file:
        snappy_hex_mesh_file.write(snappy_hex_mesh.format(
            x_in_mesh= .99 * lb['x'], y_in_mesh= .99 * lb['y'], z_in_mesh= .99 * lb['z']))

    # set time limit
    if motion == 'linear':
        t_max = (SIMULATION_TIME_MULTIPLIER * domain_size *
                 (1. if axis == 'z' and z_constraint == 'floating' and velocity_rate >= 0.
                  else FLOW_DIRECTION_SIZE_MULTIPLIER) / abs(velocity_rate))
    else:
        t_max = SIMULATION_TIME_MULTIPLIER * 2 * np.pi / abs(velocity_rate)
    with open('./temp/system/controlDict', 'r') as control_file:
        control = control_file.read()
    with open('./temp/system/controlDict', 'w') as control_file:
        control_file.write(control.format(t_max=t_max, max_CFL_number=MAX_CFL_NUMBER, z_to_waterline=z_to_waterline))

def run_case() -> None:
    """
    Runs case currently in temp directory. Wrapper for bash script.
    """

    subprocess.run(['./runFoam.sh'], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)



# script runner
if __name__ == "__main__":
    # run analysis to get vehicle coefficients
    print(analyze_model('./models/remus.stl', 'submerged', 0., 'second_only', 'forward'))
    print(analyze_model('./models/aries.stl', 'submerged', 0., 'second_only', 'forward'))
    print(analyze_model('./models/otter.stl', 'floating', 0.04, 'second_only', 'forward'))\
    
    # run sphere cases
    run_model('./models/sphere_10cm.stl', './results/sphere_10cm/submerged_positive', 
              [(v, v*0.1*np.pi) for v in 10**np.linspace(-4.,0.,41)], 'submerged', 0.)
    run_model('./models/sphere_10cm.stl', './results/sphere_10cm/submerged_negative', 
              [(-v, -v*0.1*np.pi) for v in 10**np.linspace(-4.,0.,41)], 'submerged', 0.)
    run_model('./models/sphere_10cm.stl', './results/sphere_10cm/floating_positive', 
              [(v, v*0.1*np.pi) for v in 10**np.linspace(-4.,0.,41)], 'floating', 0.)
    run_model('./models/sphere_10cm.stl', './results/sphere_10cm/floating_negative', 
              [(-v, -v*0.1*np.pi) for v in 10**np.linspace(-4.,0.,41)], 'floating', 0.)    
    run_model('./models/sphere_1m.stl', './results/sphere_1m/submerged_positive', 
              [(v, v*0.1*np.pi) for v in 10**np.linspace(-2.,2.,41)], 'submerged', 0.)
