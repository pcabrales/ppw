''' Functions to induce intratumoral heterogeneity inside the tumor by
creating regions inside it with different mean lives.
The code to create the regions is based on https://github.com/Cyril-Meyer/NumPyRandomShapes3D'''

import random
from random import randint
import array_api_compat.cupy as xp
from cupyx.scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
)
import cv2


def spheroid_coordinate(shape, center, radius, rotation=(None, None, None)):
    """
    return coordinates of point in the spheroid.
    rotation on the xy plan around the z axis.
    warning: rotation is not a perfectly verified feature, use at risk.
    """
    x_lim, y_lim, z_lim = xp.ogrid[
        0 : float(shape[0]), 0 : float(shape[1]), 0 : float(shape[2])
    ]
    x_org, y_org, z_org = center
    x_rad, y_rad, z_rad = radius
    x, y, z = (x_lim - x_org), (y_lim - y_org), (z_lim - z_org)

    if rotation == (None, None, None):
        d = (x / x_rad) ** 2 + (y / y_rad) ** 2 + (z / z_rad) ** 2
    else:
        r1, r2, r3 = rotation
        d = (
            (
                (
                    x * xp.cos(r2) * xp.cos(r3)
                    + z * xp.sin(r2)
                    - y * xp.cos(r2) * xp.sin(r3)
                )
                / x_rad
            )
            ** 2
            + (
                (
                    -z * xp.cos(r2) * xp.sin(r1)
                    + x
                    * (xp.cos(r3) * xp.sin(r1) * xp.sin(r2) + xp.cos(r1) * xp.sin(r3))
                    + y
                    * (xp.cos(r1) * xp.cos(r3) - xp.sin(r1) * xp.sin(r2) * xp.sin(r3))
                )
                / y_rad
            )
            ** 2
            + (
                (
                    z * xp.cos(r1) * xp.cos(r2)
                    + x
                    * (-xp.cos(r1) * xp.cos(r3) * xp.sin(r2) + xp.sin(r1) * xp.sin(r3))
                    + y
                    * (xp.cos(r3) * xp.sin(r1) + xp.cos(r1) * xp.sin(r2) * xp.sin(r3))
                )
                / z_rad
            )
            ** 2
        )

    return xp.nonzero(d < 1)


dtype = xp.uint8


def random_spheroid(
    array_shape, radius, shapes_max_size=None, rot_range=(-20, 20), center=None
):
    """return an array filled with a random spheroid."""
    array = xp.zeros(array_shape, dtype=dtype)
    if center is None:
        center = (
            randint(0, array.shape[0] - 1),
            randint(0, array.shape[1] - 1),
            randint(0, array.shape[2] - 1),
        )

    if radius is None and shapes_max_size is not None:
        radius = (
            randint(1, shapes_max_size[0]),
            randint(1, shapes_max_size[1]),
            randint(1, shapes_max_size[2]),
        )
    rot = (
        xp.deg2rad(randint(rot_range[0], rot_range[1])),
        xp.deg2rad(randint(rot_range[0], rot_range[1])),
        xp.deg2rad(randint(rot_range[0], rot_range[1])),
    )

    array[spheroid_coordinate(array.shape, center, radius, rot)] = 1

    return array


def erode_shape(array, erosion_size=16, iteration=10):
    """dilation and erosion with random structuring element to create random shapes from spheroid"""
    # We change the axis order because the operation are 2D
    for _ in range(iteration):
        kernel = xp.random.randint(
            2, size=(erosion_size, erosion_size, erosion_size), dtype=xp.uint8
        )
        array = binary_dilation(array, structure=kernel).astype(array.dtype)
        kernel = xp.random.randint(
            2, size=(erosion_size, erosion_size, erosion_size), dtype=xp.uint8
        )
        array = binary_erosion(array, structure=kernel).astype(array.dtype)

        array = xp.moveaxis(array, [0, 1], [1, 2])

    for _ in range(3 - (10 % 3)):
        array = xp.moveaxis(array, [0, 1], [1, 2])

    return array


def elastic_deformation(image, alpha=None, sigma=None, random_state=None):
    """
    apply random elastic deformation
    function based on _elastic function from https://github.com/ShuangXieIrene/ssds.pytorch
    """
    if alpha is None:
        alpha = image.shape[0] * random.uniform(0.5, 2)
    if sigma is None:
        sigma = int(image.shape[0] * random.uniform(0.5, 2))
    if random_state is None:
        random_state = xp.random.RandomState(None)

    for _ in range(3):
        image = xp.moveaxis(image, [0, 1], [1, 2])
        shape = image.shape[:2]

        dx, dy = [
            gaussian_filter((random_state.rand(*shape) * 2 - 1) * alpha, sigma)
            for _ in range(2)
        ]
        x, y = xp.meshgrid(xp.arange(shape[1]), xp.arange(shape[0]))
        x, y = xp.clip(x + dx, 0, shape[1] - 1).astype(xp.float32), xp.clip(
            y + dy, 0, shape[0] - 1
        ).astype(xp.float32)
        image = xp.array(
            cv2.remap(
                image.get(),
                x.get(),
                y.get(),
                interpolation=cv2.INTER_LINEAR,
                borderValue=0,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return image


def get_tumor_regions(
    tumor_mask,
    minimum_mean_lives,
    maximum_mean_lives,
    average_mean_lives,
    edges_region_types,
    voxel_size=(1, 1, 1),
    max_num_regions=5,
    isotope_list=["C11"],
    tumor_mask_dilated=None,
):
    """
    return the regions of the tumor with their induced mean lives.
    """
    if tumor_mask_dilated is None:
        tumor_mask_dilated = tumor_mask.copy()
    # first we find the biggest sphere that we can fit in the tumor
    distance_map = distance_transform_edt(tumor_mask)
    max_radius = xp.max(distance_map)
    # now we sample up to four regions (spheroids) inside the tumor by looking at least 25% of the max_radius inside the tumor
    indices = xp.where(distance_map > 0.1 * max_radius)
    # sample 5 random indices
    n_samples = xp.random.randint(0, max_num_regions + 1).item()
    sample_indices = xp.random.choice(len(indices[0]), n_samples, replace=False)
    regions_mask = tumor_mask.copy().astype(int)  # Convert to int to add the regions
    min_tumor_size_factor = 0.2
    max_tumor_size_factor = 2.0
    min_shape_vol = 0.1  # ml
    min_shape_voxels = int(min_shape_vol / xp.prod(xp.array(voxel_size) / 10)) + 1
    print(
        "Minimum number of voxels in the region to have a tumor of at least 0.1 mL (small):",
        min_shape_voxels,
    )

    for sample in range(n_samples):
        # get the coordinates of the sample
        sample_coord = (
            indices[0][sample_indices[sample]],
            indices[1][sample_indices[sample]],
            indices[2][sample_indices[sample]],
        )
        sum_shape_voxels = 0
        while (
            sum_shape_voxels < min_shape_voxels
            or sum_shape_voxels > 0.95 * tumor_mask.sum()
        ):  # we want the region to be at least 0.1 mL and at most 90% of the tumor
            radius = int(
                random.uniform(min_tumor_size_factor, max_tumor_size_factor)
                * max_radius
            )
            if radius == 0:  # avoid empty structures
                radius = 1
            shape = erode_shape(
                elastic_deformation(
                    random_spheroid(
                        tumor_mask.shape,
                        (
                            radius * voxel_size[0] / xp.max(voxel_size),
                            radius * voxel_size[1] / xp.max(voxel_size),
                            radius * voxel_size[2] / xp.max(voxel_size),
                        ),
                        center=sample_coord,
                    ),
                ),
                erosion_size=radius,
                iteration=4,
            ).astype(bool)
            shape = (
                shape & tumor_mask
            )  # keep only the part of the shape that is inside the tumor
            sum_shape_voxels = shape.sum()
        # if total_region is true
        regions_mask[shape] = sample + 2

    regions = []  # list of regions
    new_n_samples = 0
    for sample in range(n_samples):
        region_mask = regions_mask == sample + 2
        print("Number of voxels in region", sample, ":", region_mask.sum())
        num_subregions = label(region_mask.astype(int), structure=xp.ones((3, 3, 3)))[
            1
        ]  # if the region is not connected, as it has been split by other regions, skip it
        if region_mask.sum() > min_shape_voxels or num_subregions > 1:
            regions.append(region_mask.get())
            new_n_samples += 1
    print("Original number of regions:", n_samples)
    n_samples = new_n_samples
    print("Number of regions after removing too small regions:", n_samples)

    # +2 regions (tumor itself without regions and outside of the tumor from the dilation)
    mean_lives_regions = {
        isotope: xp.random.uniform(
            minimum_mean_lives[isotope], maximum_mean_lives[isotope], n_samples + 1
        )
        for isotope in isotope_list
    }

    tumor_background = regions_mask == 1
    regions.append(tumor_background.get())

    outside_tumor = regions_mask == 0
    regions.append(outside_tumor.get())
    for isotope in isotope_list:
        mean_lives_regions[isotope] = xp.append(
            mean_lives_regions[isotope], average_mean_lives[isotope]
        )

    clean_decay_map = {}
    region_types_map = {}
    for isotope in isotope_list:
        clean_decay_map[isotope] = xp.zeros_like(tumor_mask, dtype=xp.float32)
        region_types_map[isotope] = xp.zeros_like(tumor_mask, dtype=xp.uint8)
        region_types_isotope = (
            xp.digitize(mean_lives_regions[isotope], edges_region_types[isotope]) - 1
        )
        for region, mean_live_region, region_type in zip(
            regions, mean_lives_regions[isotope], region_types_isotope
        ):
            clean_decay_map[isotope][region] = (
                1 / mean_live_region
            )  # decay rate (lambda map), not mean lives
            region_types_map[isotope][region] = region_type

    return regions, mean_lives_regions, clean_decay_map, region_types_map
