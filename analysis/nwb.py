"""Analysis specific for NWB file format."""

import numpy as np


def explain_nwb_file(file):
    """Explain an NWB file.

    Args:
        file: NWB file handle
    """
    # file.units.electrodes.data[:] should be the electrode of each unit
    # info about units
    units = file.units
    print('Number of spiking units', units.spike_times_index.shape)
    print('These {:d} units have electrode indices {:s}'.format(
        units.electrodes_index.data[:].shape[0],
        str(np.unique(units.electrodes_index.data[:]))
    ))
    print(
        'These electrode indices belong to {:d} electrodes with {:d} unique indices'.format(
            file.units.electrodes.data[:].shape[0],
            len(np.unique(file.units.electrodes.data[:]))
        ))

    # file.electrodes.location.data[:] should be the location of each electrode
    electrodes = file.electrodes
    print('{:d} electrodes at unique locations {:s}'.format(
        electrodes.location.data[:].shape[0],
        str(np.unique(electrodes.location.data[:]))
    ))

    # Compute the location of each unit
    # -1 is needed because the indexing starts at 1 (matlab convention)
    unit_location = units.electrodes.data[:][
        file.units.electrodes_index.data[:] - 1]
    unit_location = electrodes.location.data[:][unit_location]
    print('Obtained unit locations', unit_location)
