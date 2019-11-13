import os

class BySceneAndPatchOrganiser:
    """Organiser for output files based on scenes and patch ids."""
    def __init__(self):
        pass
    def __call__(self,out_path,scene_id,patch_ids):
        """Organises output directory structure for all samples taken from a scene.

        Parameters
        ----------
        out_path : str
            Parent directory for outputted dataset.
        scene_id : str
            Identifier for scene.
        patch_ids : list
            Identifiers for all samples taken from scene.

        Returns
        -------
        list
            Output directories corresponding to each entry in patch_ids.
        """
        return [os.path.join(out_path,scene_id,patch_id) for patch_id in patch_ids]
