# -*- coding: utf-8 -*-
"""Base config class for storing and reading parameters.

Taken from Transport for the North's NorMITs-Demand repository.
Source:
https://github.com/Transport-for-the-North/NorMITs-Demand/blob/master/normits_demand/utils/config_base.py
"""

##### IMPORTS #####
import json
import pathlib

import pydantic
import strictyaml


##### CLASSES #####
class BaseConfig(pydantic.BaseModel):
    r"""Base class for storing model parameters.

    Contains functionality for reading / writing parameters to
    config files in the YAML format.

    See Also
    --------
    [pydantic docs](https://pydantic-docs.helpmanual.io/):
        for more information about using pydantic's model classes.
    `pydantic.BaseModel`: which handles converting data to Python types.
    `pydantic.validator`: which allows additional custom validation methods.
    """

    @classmethod
    def from_yaml(cls, text: str):
        """Parse class attributes from YAML `text`.

        Parameters
        ----------
        text : str
            YAML formatted string, with parameters for
            the class attributes.

        Returns
        -------
        Instance of self
            Instance of class with attributes filled in from
            the YAML data.
        """
        data = strictyaml.load(text).data
        return cls.parse_obj(data)

    @classmethod
    def load_yaml(cls, path: pathlib.Path):
        """Read YAML file and load the data using `from_yaml`.

        Parameters
        ----------
        path : pathlib.Path
            Path to YAML file containing parameters.

        Returns
        -------
        Instance of self
            Instance of class with attributes filled in from
            the YAML data.
        """
        with open(path, "rt") as file:
            text = file.read()
        return cls.from_yaml(text)

    def to_yaml(self) -> str:
        """Convert attributes from self to YAML string.

        Returns
        -------
        str
            YAML formatted string with the data from
            the class attributes.
        """
        # Use pydantic to convert all types to json compatiable,
        # then convert this back to a dictionary to dump to YAML
        json_dict = json.loads(self.json())

        # Strictyaml cannot handle None so excluding from output
        json_dict = _remove_none_dict(json_dict)

        return strictyaml.as_document(json_dict).as_yaml()

    def save_yaml(self, path: pathlib.Path) -> None:
        """Write data from self to a YAML file.

        Parameters
        ----------
        path : pathlib.Path
            Path to YAML file to output.
        """
        with open(path, "wt") as file:
            file.write(self.to_yaml())


##### FUNCTIONS #####
def _remove_none_dict(data: dict) -> dict:
    """Remove items recursively from dictionary which are None."""
    filtered = {}

    for key, value in data.items():
        if value is None:
            continue

        if isinstance(value, dict):
            value = _remove_none_dict(value)

        filtered[key] = value

    return filtered
