import os
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class HitAutobagger(BaseDataset):

  IMG_EXTENSIONS = ('.png',)

  '''CAREFUL: THESE WILL BE GIVEN INTEGER LABELS IN THE BELOW ORDER'''
  CLASSES = [
              '72_whistle_light_keychain',
              '23567_findlay_velvet_touch_keyring',
              '105_oval_led_keychain',
              '2629_mini_usb_charger_keychain',
              '20004_door_opener_stylus',
              '2555_pops_keychain_w_bottleopener',
              '25211_bamboo_charger',
              '62_tapeamatic_keytag',
              '197_binder_flip_clip',
              '170_slim_bottle_opener',
              '130_aluminum_led_flashlight_w_bottleopener',
              'c2w_swivel_usb_drive',
              '4790_leatherette_keytag',
              '2503_mini_aluminum_led_flashlight',
              '157_led_keychain',
              '20002_stylus_keychain',
              '2932_aluminum_cellphone_ring',
              '189_foam_phone_stand',
              '103_slim_led_light',
              '108_whistle_light_compass_keychain',
              '2282_aluminum_carabiner_w_triple_split_ring',
              '7344_reflective_tape_measure',
              '172_bottle_opener_key_light',
              '4770_circular_metal_spinner_keytag',
              '4713_circular_metal_keytag',
              '4796_color_block_mirrored_keytag',
              '4704_rectangle_metal_keytag',
              '2094_aluminum_keytag_w_strap',
              '7204_aluminum_multitool_w_carabiner',
              '2532_mini_cylinder_led_flashlight',
              '2565_everset_ruler_carabiner',
              '1652_magnifier_led_keychain',
              '7211_led_light_w_pen',
              '154_hardhat_led_keychain',
              '20001_brass_door_opener',
              '5555_multi_purpose_knife',
              '2526_aluminum_keychain_flashlight',
              '2600_onthego_car_adapter',
              '155_reflector_keylight_w_carabiner',
              '4706_athens_keychain',
              '8704_4in1_mini_nailfile',
              '144_rectangular_led_keychain',
              '8724_easy_grip_nailclippers'
            ]


  def load_annotations(self):
      if isinstance(self.ann_file, str):
          with open(self.ann_file) as f:
              samples = [x.strip().split(' ') for x in f.readlines()]
      else:
          raise TypeError('ann_file must be a str or None')
      self.samples = samples

      data_infos = []
      for filename, gt_label in self.samples:
          info = {'img_prefix': self.data_prefix}
          info['img_info'] = {'filename': filename}
          info['gt_label'] = np.array(gt_label, dtype=np.int64)
          data_infos.append(info)
      return data_infos
