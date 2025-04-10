from dataset.datasets import SSGFeatEncoder3D, SSGLWBFeat3D
from dataset.dataset_geo import SSGRelGeoFeatEncoder3D
from dataset.dataset_view import SSGRelGeoViewFeatEncoder3D
from utils.data_utils import read_scan_data, read_scan_data_with_edge_view

def build_dataset(config, split, device, ft=False, is_geo_only=False, is_edge_view=False):
    if (not ft) and is_edge_view:
        scan_data, relationship_json, objs_json, scans, o_obj_cls, o_rel_cls = \
            read_scan_data_with_edge_view(config, split, device)
        dataset = SSGRelGeoViewFeatEncoder3D(
            config, split, device,
            scan_data, relationship_json, objs_json, scans,
            o_obj_cls, o_rel_cls
        )
        return dataset
    scan_data, relationship_json, objs_json, scans, o_obj_cls, o_rel_cls = \
        read_scan_data(config, split, device)
    if (not ft) and (not is_geo_only):
        dataset = SSGFeatEncoder3D(
            config, split, device,
            scan_data, relationship_json, objs_json, scans,
            o_obj_cls, o_rel_cls
        )
    elif (not ft) and is_geo_only:
        dataset = SSGRelGeoFeatEncoder3D(
            config, split, device,
            scan_data, relationship_json, objs_json, scans,
            o_obj_cls, o_rel_cls
        )
    else:
        dataset = SSGLWBFeat3D(
            config, split, device,
            scan_data, relationship_json, objs_json, scans
        )
    return dataset